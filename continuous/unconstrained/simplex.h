#pragma once
#include <Eigen/Dense>
#include <optimization/util/apply.h>
#include <random>
#include <memory>
#include <iostream>
#include <functional>
#include <utility>
#include <tuple>
#include <bit>
#include <ranges>

namespace optimization
{

template<size_t MAXITERS, typename T, typename ... Args>
auto simplex(std::function<T(Args...)> objective,
    std::vector<Eigen::Vector<T, sizeof...(Args)>> x,
    T alpha = 1.0,
    T beta = 2,
    T gamma = 0.5)
{
    using namespace std;
    using namespace Eigen;
    using namespace Eigen::placeholders;
    constexpr const size_t N = sizeof...(Args);
    using V = Vector<T, N>;

    vector<pair<V, T>> xy(N+1);
    V centroid;

    T sdev = std::numeric_limits<T>::max();

    // evaluate objective at simplex points x_n
    transform(x.cbegin(), x.cend(), xy.begin(), [&](V arg) {
        T val = apply(objective, arg);
        return make_pair(arg, val);
    });

    for (auto i = 0; i < MAXITERS && sdev > 1e-8; ++i) {
        // argsort
        sort(xy.begin(), xy.end(), [](auto arg1, auto arg2) { return arg1.second < arg2.second; });
        // compute centroid
        centroid = V::Zero();
        for (auto k = 0; k < N; ++k) {
            centroid.noalias() += xy[k].first;
        }
        centroid /= N;

        auto& [xbest, ybest] = xy.front();
        auto& [xworst, yworst] = xy.back();
        auto& [xworst2, yworst2] = *(xy.cend() - 2);

        V xreflect = centroid + alpha * (centroid - xworst);
        auto yreflect = apply(objective, xreflect);

        if (yreflect < ybest) {
            V xexpand = centroid + beta * (xreflect - centroid);
            auto yexpand = apply(objective, xexpand);
            if (yexpand < yreflect) {
                xworst = xexpand;
                yworst = yexpand;
            } else {
                xworst = xreflect;
                yworst = yreflect;
            }
        } else if (yreflect > yworst2) {
            if (yreflect <= yworst) {
                xworst = xreflect;
                yworst = yreflect;
            }
            V xcontract = centroid + gamma * (xworst - centroid);
            auto ycontract = apply(objective, xcontract);
            if (ycontract > yworst) {
                for (auto k = 2; k < N + 1; ++k) {
                    xy[k].first = (xy[0].first + xy[k].first) / 2.;
                    xy[k].second = apply(objective, xy[k].first);
                }
            } else {
                xworst = std::move(xcontract);
                yworst = std::move(ycontract);
            }
        } else {
            xworst = std::move(xreflect);
            yworst = std::move(yreflect);
        }

        T mean = 0;
        for (auto ii = 0; ii < xy.size(); ++ii) {
            mean += xy[ii].second;
        }
        mean /= xy.size();

        sdev = 0;
        for (auto ii = 0; ii < xy.size(); ++ii) {
            sdev += pow(xy[ii].second - mean, 2.);
        }
        sdev /= xy.size();
    }

    return xy.front().first;
}

}
