#pragma once

#include <Eigen/Dense>
#include <Eigen/Householder>
#include <Eigen/QR>
#include <Eigen/Jacobi>
#include <random>
#include <memory>
#include <iostream>
#include <functional>
#include <utility>
#include <tuple>
#include <iomanip>
#include <memory>
#include <cxxabi.h>
#include <cassert>
#include <optimization/util/apply.h>

namespace optimization
{

enum class ResultCode
{
    OK,
    NUMERICAL_ISSUE,
    ITERATION_LIMIT,
};
struct LMResults
{
    Eigen::VectorXd x;
    ResultCode code;
    double value;
};

namespace detail
{

template<typename T, int N>
T phi(T delta,
    const Eigen::Vector<T, N>& p,
    const Eigen::DiagonalMatrix<T, N, N>& D)
{
    return (D * p).norm() - delta;
}

template<typename T, int N>
T dphi(const Eigen::Vector<T, N>& q,
    const Eigen::DiagonalMatrix<T, N, N>& D,
    auto& R,
    auto& perm)
{
    auto qhat = q / q.stableNorm();
    auto Dqhat = D.diagonal().cwiseProduct(qhat);
    Eigen::VectorXd piDqhat = perm.transpose() * Dqhat;
    //auto tmp = R.transpose().solve(perm.transpose() * D.diagonal().cwiseProduct(q) / q.stableNorm());
    Eigen::VectorXd tmp = R.transpose().solve(piDqhat);
    return - q.norm() * tmp.squaredNorm();
}
}

template<size_t MAXITERS, typename T>
auto levenbergMarquardt(
    const Eigen::Vector<T, Eigen::Dynamic>& y,
    const Eigen::Vector<T, Eigen::Dynamic>& arg,
    auto fn, auto jacobian,
    Eigen::Vector<T, Eigen::Dynamic>& x0,
    const T zeroTol = 1e-12,
    const T objTol = 1e-10)
{
    using namespace std;
    using namespace Eigen;
    using namespace Eigen::placeholders;
    using detail::pack;

    const size_t M = y.size();
    const size_t N = x0.size();

    Matrix<T, Dynamic, Dynamic> J;

    T obj, objnew, prevobj = 0;
    VectorXd f(M), work(M + N);
    Vector<T, Dynamic> x = x0, step, p, xnew;
    Vector<T, Dynamic> b = Vector<T, Dynamic>::Zero(M+N);
    T lambda = 0., delta = .1, sigma = 0.01;
    DiagonalMatrix<T, Dynamic, Dynamic> D;

    Matrix<T, Dynamic, Dynamic> A(M+N, N); A.setZero();
    // QR decomposition of a rectangular MxN matrix will give Q: MxM and R: MxN

    assert(f.rows() == y.rows());
    f = fn(arg, x0);
    obj = 0.5 * (f - y).squaredNorm();
    if (std::isnan(obj)) {
        return LMResults{ .x = x0, .code = ResultCode::NUMERICAL_ISSUE, .value = std::numeric_limits<T>::quiet_NaN() };
    }
    assert(isfinite(obj) && !std::isnan(obj));

    // compute the Jacobian at the current point x = x0
    J = jacobian(arg, x0);

    D.diagonal() = VectorXd::Ones(N);
//    D.diagonal() = (J.transpose() * J).diagonal();
//    D.diagonal() = J.colwise().norm();
    b(seq(0, M-1)) = f - y;

    // QR-decompose the Jacobian
    ColPivHouseholderQR<MatrixXd> qr(J);

//    cout << "J=" << endl << J << endl;

    MatrixXd QTI(M + N, M + N); QTI.setZero();
    QTI(seq(0, M-1), seq(0, M-1)) = qr.matrixQ().transpose();
    QTI(seq(M, last), seq(M, last)) = MatrixXd::Identity(N, N);

    // this triangular view corresponds to triangular matrix R in the QR decomposition.
    // it's dimensions are M x N
    //auto triu0 = qr.matrixR().topLeftCorner(qr.rank(), qr.rank()).template triangularView<Upper>();

    work.setZero();
    if (qr.rank() > 0) {
        work(seq(0, N - 1)) = qr.solve(-(f - y));//triu.solve(tmp);
    }
    step = /*qr.colsPermutation() * */ work(seq(0, N-1));

    T stepNorm = D.diagonal().cwiseProduct(step).stableNorm();
    size_t i = 0;
    for (auto k = 0; (i < MAXITERS) && (i == 0 || (stepNorm > zeroTol && (k == 0 || (k > 1 && (prevobj - obj)/prevobj > objTol)) )) && (delta > zeroTol); ++i) {
        xnew = x;
        auto triu = qr.matrixR().topLeftCorner(qr.rank(), qr.rank()).template triangularView<Upper>();

        if (stepNorm <= (1. + sigma) * delta && stepNorm > 0) {
            xnew = x + step;
            lambda = 0.;
        } else {
            T l = 0.;
            if (qr.rank() == N) {
                // todo: implement
                //l = (D*step).norm() / detail::dphi(D.diagonal().cwiseProduct(step).eval(), D, triu, qr.colsPermutation());
                //l = max(l, 0.);
            }
            T u = ((J * D.inverse()).transpose() * f).norm() / delta;
            if (!isfinite(u)) {
                //return LMResults{ .code = ResultCode::NUMERICAL_ISSUE };
                u = 100;
            }
            assert(isfinite(u));
            if (lambda == 0.) {
                lambda = 1e-3 * u;
                assert(!std::isnan(lambda));
                assert(isfinite(lambda));
            }

            JacobiRotation<T> G;
            Vector<T, Dynamic> rhs0 = Vector<T, Dynamic>::Zero(M+N), rhs;
            rhs0(seq(0, M-1)) = (f-y);
            rhs0.applyOnTheLeft(-QTI);

            for (auto j = 0; j < 10; ++j) {
//                if (lambda < l || lambda > u) {
//                    lambda = max(1e-3 * u, sqrt(l * u));
//                }

                // triu is an R upper triangular coming from the factorization of J.P = Q.R
                A(seq(0, qr.rank()-1), seq(0, qr.rank()-1)) = triu;
                ////////A(seq(M, last), seq(0, last)) = sqrt(lambda) * D; <--- old variant
                A(seq(M, last), seq(0, last)) = sqrt(lambda) * DiagonalMatrix<T, Dynamic>(qr.colsPermutation() * D.diagonal());
                //ColPivHouseholderQR<Matrix<T, M + N, N>> qr2(A);

//                cout << "A=" << endl << A << endl;
//                cout << "rhs=" << rhs0.transpose() << endl;
                rhs = rhs0;

                for (size_t w = 0, v = N-1u; w < N; ++w, --v) {
                    G.makeGivens(A(v,v), A(M+v, v));
                    A.applyOnTheLeft(v, M+v, G.adjoint());
                    rhs.applyOnTheLeft(v, M+v, G.adjoint());

                    for (auto r = v+1; r < N; ++r) {
                        G.makeGivens(A(r, r), A(M+v, r));
                        A.applyOnTheLeft(r, M+v, G.adjoint());
                        rhs.applyOnTheLeft(r, M+v, G.adjoint());
                    }
                }

                auto triu2 = A.topRightCorner(N, N).template triangularView<Upper>();

                work.setZero();
                work(seq(0, N-1)) = triu2.solve(rhs(seq(0, N-1)));
                p = qr.colsPermutation() * work(seq(0, N-1));
                //cout << (j+1) << ": candidate step p=" << p.transpose() << endl;
                auto tmp = D.diagonal().cwiseProduct(p).stableNorm();
                if (tmp <= (1. + sigma) * delta) {
                    step = p;
                    stepNorm = D.diagonal().cwiseProduct(step).stableNorm();
                    xnew.noalias() = x + step;
                    break;
                }

                decltype(p) q = D * p;

                using detail::phi, detail::dphi;
                T phi_k = phi(delta, p, D);
                T dphi_k = dphi(q, D, triu2, qr.colsPermutation());
                if (phi_k < 0) {
                    u = lambda;
                }
                l = max(l, lambda - phi_k / dphi_k);
                lambda = lambda - (phi_k + delta)/delta * phi_k/dphi_k;
                assert(!std::isnan(lambda));
            }
        }

        work(seq(0, M-1)) = fn(arg, xnew);

        objnew = .5 * (work(seq(0, M-1)) - y).squaredNorm();
        T ared = obj - objnew;
        if (ared < -zeroTol) {
            delta *= .5;
            //lambda *= 2.;
        } else {
            T gain = 0;
            T pred = .5 * step.transpose() * (lambda * step - J.transpose() * (f - y));
            if (pred > zeroTol) {
                gain = ared / pred;
            }

            if (gain <= 0.25) {
                delta *= 0.5;// (gain / 0.25) * (0.5 * delta - 0.1 * delta);
                //lambda *= 2.;
            } else if (gain >= 0.75 || (lambda == 0. && gain >= .25 && gain <= .75)) {
                //auto Dp = D.diagonal().cwiseProduct(step).stableNorm();
                delta *= 4;// * Dp;
                //lambda *= .25;
            }

            if (gain > zeroTol) {
                x.noalias() = xnew;
                f = fn(arg, x);
                prevobj = obj;
                obj = 0.5 * (f - y).squaredNorm();
                assert(!std::isnan(obj) && !std::isinf(obj));
                if (std::isnan(obj) || std::isinf(obj)) {
                    return LMResults{.code = ResultCode::NUMERICAL_ISSUE, .value = numeric_limits<T>::quiet_NaN()};
                }
                ++k;
                //cout << i << "/" << k << ": f(" << x.transpose() << ") = " << obj << endl;

                // compute the Jacobian at the current point x
                J = jacobian(arg, x);
                b(seq(0,M-1)) = f - y;

                // recompute QR of the Jacobian
                qr.compute(J);
                QTI(seq(0, M-1), seq(0, M-1)) = qr.matrixQ().transpose();

                // recompute Newton step
                //auto triu3 = qr.matrixR().topLeftCorner(qr.rank(), qr.rank()).template triangularView<Upper>();
                work.setZero();
                work(seq(0, N-1)) = qr.solve(-(f-y));// * -b(seq(0, qr.rank()-1)));
                step = work(seq(0, N-1));
                stepNorm = D.diagonal().cwiseProduct(step).stableNorm();

//                VectorXd norms = J.colwise().stableNorm();
//                D.diagonal() = norms.cwiseMax(D.diagonal());
            }
        }

        delta = max(0., min(delta, 1e12));
    }

    LMResults results{.code = ResultCode::NUMERICAL_ISSUE, .value = numeric_limits<T>::quiet_NaN() };
    if (i == MAXITERS && isfinite(obj)) {
        results.code = ResultCode::ITERATION_LIMIT;
        results.value = obj;
        results.x = x;
    }
    if (std::isnan(obj) || std::isinf(obj)) {
        results.code = ResultCode::NUMERICAL_ISSUE;
        results.value = numeric_limits<T>::quiet_NaN();
    } else {
        results.value = obj;
        results.x = x;
        results.code = ResultCode::OK;
    }
    return results;
}

}
