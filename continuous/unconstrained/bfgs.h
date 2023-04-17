#pragma once

#include <Eigen/Dense>
#include <random>
#include <memory>
#include <iostream>
#include <functional>
#include <utility>
#include <tuple>
#include <optimization/util/apply.h>
#include <optimization/continuous/unconstrained/line_search.h>

namespace optimization
{

template<size_t MAXITERS, typename T = double, typename ArgT, typename ... ArgTs>
auto bfgs(auto objective, auto gradient, ArgT arg1, T tolerance, ArgTs&& ... args)
{
    using namespace std;
    using namespace Eigen;
    using namespace Eigen::placeholders;
    using detail::pack;

    using VectorT = Eigen::Vector<T, Eigen::Dynamic>;
    using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    const auto N = gradient(arg1, args...).size();
    MatrixT H(N, N);
    H.setIdentity();

    Vector<T, Eigen::Dynamic> x = arg1;
    Vector<T, Eigen::Dynamic> g = gradient(x);

    for (auto i = 0ull; i < MAXITERS && g.stableNorm() > tolerance; ++i) {
        const VectorT p = -H * g;
//        cout << "-H = " << endl << H << endl;
//        cout << "g=" << g.transpose() << ", gnorm = " << g.stableNorm() << endl;
//        cout << "direction: " << p.transpose() << ", dir norm=" << p.stableNorm() << endl;
        const T alpha = wolfe_line_search<100>(objective, gradient, x, p);
        const VectorT xprime = x + alpha * p;
        const VectorT gprime = gradient(xprime, args...);

        const auto s = xprime - x;
        const auto y = gprime - g;

//        if (isnan(s.dot(y))) {
//            cout << "alpha=" << alpha << ",p=" << p.transpose() << endl;
//            cout << "xprime=" << xprime.transpose() << ", x=" << x.transpose() << endl;
//            cout << "gprime=" << gprime.transpose() << ", g=" << g.transpose() << endl;
//        }
        if (abs(s.dot(y)) < 1e-10) {
            break;
        }
        auto a = s.dot(y) + y.transpose() * H * y;
        auto b = (s * s.transpose())  / (1e-10 + pow(s.dot(y),2.0));
        auto c = (H * y * s.transpose() + s * y.transpose() * H)/(1e-10 + s.dot(y));

//        cout << "s.y=" << s.dot(y) << endl;
//        cout << "#1 " << a << endl;
//        cout << "#2 " << b << endl;
//        cout << "#3 " << c << endl;

        H += a * b - c;
//        H += HH +  * (s * s.transpose()) * 1./pow(s.dot(y),2.0)
//            - (HH * y * s.transpose() + s * y.transpose() * HH)/s.dot(y);
        x = xprime;
        g = gprime;
//        cout << "new x = " << x.transpose() << endl;
    }

    return x;
}

}
