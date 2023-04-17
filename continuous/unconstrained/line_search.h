#pragma once
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <random>
#include <memory>
#include <iostream>
#include <functional>
#include <utility>
#include <tuple>
#include <optimization/util/pack.h>

namespace optimization
{

template<std::size_t MAXITERS, typename T>
auto wolfe_line_search(auto f,
    auto gradient,
    const Eigen::Vector<T, Eigen::Dynamic>& x0,
    const Eigen::Vector<T, Eigen::Dynamic>& d,
    T c1 = 1e-4, T c2 = 0.1)
{
    using namespace Eigen;
    using namespace std;
    using detail::pack;
    using VectorT = Eigen::Vector<T, Eigen::Dynamic>;

    VectorT x = x0;
    const T y0 = f(x);
    const T g0 = gradient(x0).dot(d);
    T aprev = 0, a = 1., lo, hi, yprev, g;

    for (auto i = 1; i < MAXITERS; ++i) {
        VectorXd xcandidate = x + a * d;
//        cout << "xcand=" << xcandidate.transpose() << endl;
        T y = f(xcandidate);//apply(f, pack(x + a * d, make_index_sequence<N>{}));
//        if (isnan(y)) {
//            cout << "zomg" << endl;
//            f(xcandidate);
//        }
//        cout << "y=" << y << ", y0=" << y0 << ", thresh=" << (y0 + c1 * a * g0) << endl;
        if ((y > y0 + c1 * a * g0) || (i > 1 && y >= yprev)) {
            lo = aprev;
            hi = a;
            break;
        }

//        cout << "dir=" << d.transpose() << endl;
//        cout << "new x=" << (x + a * d).transpose() << endl;
//        cout << "grad = " << gradient(x + a * d).transpose() << endl;
        g = gradient(x + a * d).dot(d); //apply(gradient, pack(x + a * d, make_index_sequence<N>{})).dot(d);
//        cout << "abs(g)=" << abs(g) << " vs " << (-c2 * g0) << endl;
        if (abs(g) <= - c2 * g0) {
            return a;
        }
        if (g >= 0) {
            lo = a;
            hi = aprev;
            break;
        }
        yprev = y;
        aprev = a;
        a = 2. * a;
//        cout << "new a = 2*" << a << "=" << 2*a << endl;
    }

    T ylo = f(x + lo * d);//apply(f, pack(x + lo * d, make_index_sequence<N>{}));
    for (auto i = 1; i < MAXITERS; ++i) {
        a = (lo + hi) / 2.; // bisection step (can be replaced with cubic)
//        cout << "new a=(" << lo << "+" << hi << ")/2=" << a << endl;

        T y = f(x + a * d);//apply(f, pack(x + a * d, make_index_sequence<N>{}));
        if (y > y0 + c1 * a * g0 || y >= ylo) {
            hi = a;
        } else {
            g = gradient(x + a * d).dot(d);//apply(gradient, pack(x + a * d, make_index_sequence<N>{})).dot(d);
            if (abs(g) <= - c2 * g0) {
                return a;
            } else if (g * (hi - lo) >= 0) {
                hi = lo;
            }
            lo = a;
        }
    }
    return a;
}
}