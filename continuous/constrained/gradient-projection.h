//
// Created by Mateusz Berezecki on 3/7/23.
//

#pragma once

#include <numeric>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <concepts>
#include <algorithm>
#include <iostream>
#include <iterator>

namespace optimization
{

namespace detail
{
using namespace Eigen;

template<std::floating_point T>
auto cauchyPoint(
    const auto& x0,
    const auto& G,
    const auto& c,
    const auto& l,
    const auto& u)
{
    using namespace std;
    constexpr T inf = std::numeric_limits<T>::infinity();
    const auto N = x0.size();

    VectorXd x = x0, p = VectorXd::Zero(N), xnext = x0;
    VectorXd g = G * x + c;

    // identify breakpoints of a piecewise linear search direction projected onto the box walls
    vector<T> tbreak(N), tbar(N);
    for (auto i = 0; i < N; ++i) {
        assert(x(i) >= l(i) && x(i) <= u(i));
        if (g(i) < 0 && u(i) < inf) {
            tbar[i] = tbreak[i] = (x(i) - u(i)) / g(i);
        } else if (g(i) > 0 && l(i) > -inf) {
            tbar[i] = tbreak[i] = (x(i) - l(i)) / g(i);
        } else {
            tbar[i] = tbreak[i] = inf;
        }
    }
    // sort the vector
    sort(tbreak.begin(), tbreak.end());

    // remove zeros
    auto first = std::upper_bound(tbreak.begin(), tbreak.end(), T(0));
    tbreak.erase(tbreak.begin(), first);

    // remove duplicates
    auto last = std::unique(tbreak.begin(), tbreak.end());
    tbreak.erase(last, tbreak.end());

    T tprev = 0;
    for (auto i = 0; i < tbreak.size(); ++i) {
        x = xnext;
        // compute new search direction
        const auto tdelta = tbreak[i] - tprev;
        for (auto j = 0; j < N; ++j) {
            if (tprev < tbar[j]) {
                p(j) = -g(j);
            } else {
                p(j) = 0;
            }
        }

        auto fprime = c.dot(p) + x.dot(G * p);
        if (fprime > 0) {
            break; // found a Cauchy point
        }

        //auto f = c.dot(x) + .5 * x.dot(G * x);
        auto fbis = p.dot(G * p);

        if (fbis != 0) {
            auto tdopt = -fprime / fbis;
            if (tdopt >= 0 && tdopt < tdelta) {
                // local minimizer at tprev + tdopt
                x += tdopt * p;
                break;
            }
        }

        tprev = tbreak[i];
        xnext = x + tdelta * p;
    }
    return x;
}

}

// Solves a quadratic program, min. f(x) = x^T.G.x + c^T.x , subject to box constraints l <= x <= u
// this method will introduce slacks as appropriate. do not include them in the problem description
template<std::size_t MAXITERS, std::floating_point T = double>
auto gradientProjectionQP(auto& x0, const auto& Garg, const auto& c, const auto& lbArg, const auto& ubArg)
{
    using namespace std;
    using namespace detail;
    using namespace Eigen;
    using namespace Eigen::placeholders;

    using VT = Eigen::Vector<T, Dynamic>;
    using MT = Eigen::Matrix<T, Dynamic, Dynamic>;

    VT lb = lbArg, ub = ubArg;
    MT G = Garg;
    assert(lb.size() == ub.size());
    const auto N = lb.size();

    VT x = x0;

    // compute the Cauchy point
    VT xcp = cauchyPoint<T>(x, G, c, lb, ub);

    // determine the active set and matrix P (and Z)
    vector<int> activeset(N), perm(N), iperm(N);
    int aix = 0, inix = N-1;
    for (int i = 0, k = N; i < k; ++i) {
        if (abs(xcp(i)) < 1e-8) {
            xcp(i) = 0;
        }
        if (xcp(i) == lb(i) || xcp(i) == ub(i)) {
            perm[aix] = i; iperm[i] = aix;
            activeset[aix++] = i;
        } else {
            perm[inix] = i; iperm[i] = inix;
            activeset[inix--] = i;
        }
    }

    if (aix == N) {
        return xcp;
    }

    // permute bounds and x
    x = xcp(perm);
    lb = lbArg(perm);
    ub = ubArg(perm);

    MT Z = MT::Zero(N, N - inix - 1);
    Z(seq(aix, last), all) = MT::Identity(N - inix - 1, N - inix - 1);

    auto P = Z * (Z.transpose() * Z).inverse() * Z.transpose();

    MT PGP = G(perm,perm);
    VT r = PGP * x + c(perm), g = P * r, d = -g;

    for (auto i = 0; i < MAXITERS; ++i) {
        auto rg = r.dot(g);
        if (abs(rg) < 1e-8) {
            break;
        }
        auto alpha = rg / (d.dot(PGP * d));
        x += alpha * d;

        r += alpha * PGP * d;
        g = P * r;
        auto beta = r.dot(g) / rg;
        VectorXd dprime = -g + beta * d;
        d.noalias() = dprime;
    }

    // project back to bounds
    for (auto i = 0; i < x.size(); ++i) {
        if (x(i) < lb(i)) {
            x(i) = lb(i);
        } else if (x(i) > ub(i)) {
            x(i) = ub(i);
        }
    }
    VT result = x(iperm);
    return result;
}
}
