#pragma once
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <optimization/util/pack.h>
#include <random>
#include <memory>
#include <iostream>
#include <functional>
#include <utility>
#include <tuple>
#include <bit>

namespace optimization
{

namespace detail
{
using namespace Eigen;
using namespace std;

template<size_t N, typename T>
auto sample()
{
    static random_device rd;
    static mt19937_64 rand(rd());
    static normal_distribution<T> dist(0., 1.);

    Vector<T, N> x;
    for (auto i = 0; i < N; ++i) {
        x(i) = dist(rand);
    }
    return x;
}
}

template<size_t MAXITERS, typename T, typename ... Args>
auto cma_es(std::function<T(Args...)> objective, T stepSize)
{
    using namespace std;
    using namespace Eigen;
    using namespace Eigen::placeholders;
    constexpr const size_t N = sizeof...(Args);
    constexpr const size_t LAMBDA = 4 + 3 * (bit_width(N) / 1.4426950408890);
    constexpr const size_t MU = LAMBDA / 2;

    auto mean = make_unique<Vector<T, N>>();
    auto prevMean = make_unique<Vector<T, N>>();
    auto cov = make_unique<Matrix<T, N, N>>();

    mean->setZero();
    cov->setIdentity();

    SelfAdjointEigenSolver<Matrix<T, N, N>> es(*cov);
    Matrix<T, N, N> B = es.eigenvectors();
    DiagonalMatrix<T, N, N> D = es.eigenvalues().cwiseSqrt().asDiagonal();
    Matrix<T, N, N> invsqrtCov;
    invsqrtCov.noalias() = B * D.inverse() * B.transpose();

    Vector<T, MU> w;
    for (auto i = 0; i < MU; ++i) {
        w(i) = log(MU + 0.5) - log(i+1);
    }
    w /= w.sum();

    T mueff = 1. / w.dot(w);

    // adaptation parameters
    T cc = (4. + mueff / N) / (N + 4. + 2. * mueff/N);
    T cs = (mueff + 2.) / (N + mueff + 5.);
    T c1 = 2. / (pow(N + 1.3, 2) + mueff);
    T cmu = min(1.-c1, 2. * (mueff - 2. + 1./mueff) / (pow(N+2, 2.) + mueff));
    T damps = 1. + 2. * max(0., sqrt((mueff-1.)/(N + 1.))-1.) + cs;
    Vector<T, N> ps, pc;

    ps.setZero();
    pc.setZero();

    const T chiN = sqrt(N) * (1. - 1. /(4. * N) + 1./(21.*pow(N,2.)));
    auto counteval = 0u, eigeneval = 0u;

    Matrix<T, N, LAMBDA> samples, deltas;
    vector<T> values(LAMBDA);
    vector<uint32_t> idx(LAMBDA);

    T best = std::numeric_limits<T>::infinity();
    Vector<T, N> bestx;

    for (auto i = 0u; i < MAXITERS; ++i) {
        for (auto k = 0u; k < LAMBDA; ++k, ++counteval) {
            Vector<T, N> delta = B * D * detail::sample<N, T>();
            Vector<T, N> sample = *mean + stepSize * delta;
            auto packed = detail::pack(sample, make_index_sequence<N>{});
            values[k] = std::apply(objective, packed);
            samples(all, k) = std::move(sample);
            deltas(all, k) = std::move(delta);
            idx[k] = k;
        }

        sort(idx.begin(), idx.end(), [&](auto x, auto y) {
            return values[x] < values[y]; });
        vector<uint32_t> ind(idx.cbegin(), idx.cbegin() + MU);

        if (abs(values[idx.front()] - best) < 1e-14) {
            break;
        }

        if (values[idx.front()] < best) {
            best = values[idx.front()];
            bestx = samples(all, idx.front());
        }

        *prevMean = *mean;
        *mean = samples(all, ind) * w;
        Vector<T, N> deltaMu = (*mean - *prevMean);
        ps = (1. - cs) * ps + sqrt(cs * (2. - cs) * mueff) * invsqrtCov * deltaMu / stepSize;
        pc = (1. - cc) * pc;
        if (ps.norm() / sqrt(1 - pow (1 - cs, 2 * counteval)) < (1.4 + 2.0 / (N + 1)) * chiN) {
            pc += sqrt(cc * (2. - cc) * mueff) * deltaMu/stepSize;
            *cov = (1. - c1 - cmu) * *cov + c1 * pc * pc.transpose();
        } else {
            *cov = (1. - c1 - cmu) * *cov + c1 * (pc * pc.transpose()) + cc * (2. - cc) * *cov;
        }

        for (auto k = 0; k < MU; ++k) {
            *cov += cmu * w(k) * deltas(all, k) * deltas(all, k).transpose();
        }

        stepSize *= exp(cs * (ps.norm() / chiN - 1) / damps);

        if ((counteval - eigeneval) * 10 * N * (c1 + cmu) > 1) {
            eigeneval = counteval;
            es.compute(*cov);
            B = es.eigenvectors();
            D = es.eigenvalues().cwiseSqrt().asDiagonal();
            invsqrtCov.noalias() = B * D.inverse() * B.transpose();
        }
    }

    return bestx;
}

}
