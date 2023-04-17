//
// Created by Mateusz Berezecki on 3/6/23.
//

#pragma once

#include <optimization/continuous/unconstrained/lm.h>
#include <optimization/continuous/constrained/gradient-projection.h>
#include <aad/expr.h>
#include <aad/tape.h>
#include <vector>

namespace optimization
{

template<size_t MAXITERS, typename T = double>
auto augmentedLagrangian(
    auto objective,
    auto constrF,
    auto aadObjective,
    auto aadConstrF,
    const Eigen::Vector<T, Eigen::Dynamic>& lowerBounds,
    const Eigen::Vector<T, Eigen::Dynamic>& upperBounds,
    const Eigen::Vector<T, Eigen::Dynamic>& x0)
{
    using namespace Eigen;
    using namespace std;
    using namespace aad;

    using VT = Vector<T, Dynamic>;
    using MT = Matrix<T, Dynamic, Dynamic>;

    MT B = MT::Identity(x0.size(), x0.size()); // initial Hessian approximation
    T mu = 1.0, omega = 1. / mu, eta = 1. / pow(mu, 0.1), etastar = 1e-6, omegastar = 1e-6, delta = 1, deltahat = 1e4;
    VT g(x0.size()), gprev = VT::Zero(x0.size()), x = x0;

    const auto CONSTRAINT_NUM = constrF(x0).size();

    VT lambda = VT::Zero(CONSTRAINT_NUM);
    VT trl = lowerBounds, tru = upperBounds;
    auto ones = VT::Ones(upperBounds.size());

    Vector<Number<T>, Dynamic> theta(x0.size());
    for (auto k = 0; k < theta.size(); ++k) {
        theta(k) = T(x(k));
    }

    Number<T>::tape->mark();
    for (auto i = 0; i < MAXITERS && delta > 1e-10; ++i) {
        // compute the gradient of the augmented lagrangian
        Number<T>::tape->rewindToMark();
        Number<T>::tape->reset();

        auto c = aadConstrF(theta);
        Number<T> clambda(0);
        for (auto k = 0; k < c.size(); ++k) {
            clambda += c(k) * lambda(k);
        }
        Number<T> L = aadObjective(theta) - clambda + .5 * mu * c.dot(c);
        L.propagateToStart();
        for (auto k = 0; k < theta.size(); ++k) {
            gprev(k) = theta(k).adjoint();
        }
        // end of gradient computation

        // construct trust region bounds
        trl = (x - delta * ones).cwiseMax(lowerBounds);
        tru = (x + delta * ones).cwiseMin(upperBounds);
//        cout << "tr lb=" << trl.transpose() << endl
//            << "tr ub=" << tru.transpose() << endl;

//        cout << "g(x) violation = " << clambda.value() << endl;
        // solve quadratic program with box constraints using trust region bounds
        auto xnext = gradientProjectionQP<1000>(x, B, gprev, trl, tru);

        // compare the actual reduction vs projected reduction
        auto cc = constrF(xnext);
        T ared = L.value() - (objective(xnext) - lambda.dot(cc) + .5 * mu * cc.dot(cc));
        if (ared < 0) {
            delta /= 4;
            continue;
        }
        auto dx = xnext - x;
        T pred = - gprev.dot(dx) - .5 * dx.dot(B * dx);
        T rho = 0;

        if (pred > 0) {
            rho = ared / pred;
        }

//        cout << "g=" << gprev.transpose() << endl << "p=" << dx.transpose() << endl;
//        cout << "g.p = " << gprev.dot(dx) << endl;
//        cout << "at x=" << x.transpose() << " approximation yields "
//             << (L.value() + gprev.dot(dx) + .5 * dx.dot(B * dx))
//             << " vs actual value of " << (objective(xnext) - lambda.dot(cc) + .5 * mu * cc.dot(cc)) << endl;

        if (dx.norm() < 1e-10) {
            break;
        }

        if (rho < 0.01) {
            delta /= 4;
            mu *= 2;
            continue;
        } else {
            if (rho > .75 && dx.norm() == delta) {
                delta = min(2 * delta, deltahat);
            }
        }

        if (dx.norm() < 1e-10) {
            continue;
        }

//        cout << "x'=" << xnext.transpose() << endl;
        for (auto k = 0; k < theta.size(); ++k) {
            theta(k) = xnext(k);
        }
        //Number<T>::tape->reset();
        Number<T>::tape->rewindToMark();
        Number<T>::tape->reset();
        c = aadConstrF(theta);
        clambda = T(0);
        for (auto k = 0; k < c.size(); ++k) {
            clambda += c(k) * lambda(k);
        }
        L = aadObjective(theta) - clambda + .5 * mu * c.squaredNorm();
//        cout << "augmented lagrangian value at x'=" << L.value() << endl;
        L.propagateToStart();
        for (auto k = 0; k < theta.size(); ++k) {
            g(k) = theta(k).adjoint();
        }
//        cout << "g' = " << g.transpose() << endl;

        // update Hessian approximation B
        const auto s = xnext - x;
        const auto y = g - gprev;

        T dampedBfgsTheta = 1.0;
        auto sBs = s.dot(B * s);
        auto sy = s.dot(y);

        if (sy < .2 * sBs) {
            dampedBfgsTheta = (0.8 * sBs) / (sBs - sy);
        }
        auto rr = dampedBfgsTheta * y + (1 - dampedBfgsTheta) * (B * s);

        auto Bs = B * s;
        auto sB = s.transpose() * B;
        auto rrt = rr * rr.transpose();
        auto srr = s.dot(rr);

        B -= (Bs * sB) / sBs + rrt / srr;

#if 0
        if (i == 0) {
            //B *= y.dot(y) / y.dot(s);
            //B = gprev * gprev.transpose();
        }
#endif
//        cout << "s=" << s.transpose() << endl;
//        cout << "y=" << y.transpose() << endl;
//        auto aa = s.dot(y) + y.transpose() * B * y;
//        auto bb = (s * s.transpose())  / (1e-20 + pow(s.dot(y),2.0));
//        auto ccc = (B * y * s.transpose() + s * y.transpose() * B)/(1e-20 + s.dot(y));

//        cout << "B=" << endl << B << endl;
        //B += aa * bb - ccc;
//        cout << "B'=" << endl << B << endl;

        // update z (lambda), and mu
        const auto cd = c.template cast<double>();
        auto cnext = constrF(xnext);

//        cout << "cd = " << cd << endl;
//        cout << "cnext = " << cnext << endl;
        lambda -= mu * cnext;
        if (cnext.stableNorm() >= (.25 * cd.stableNorm())) {
            mu *= 2;
        } else {

        }

        // convergence test
        if (cd.stableNorm() < eta) {
            if ((xnext - x).stableNorm() < omegastar && cnext.stableNorm() < etastar) {
                x = xnext;
                break;
            } else {
                lambda -= mu * cd;
                eta = eta / pow(mu, 0.9);
                omega = omega / mu;
            }
        } else {
            mu *= 100;
            eta = 1 / pow(mu, .1);
            omega = 1 / mu;
        }

        // save xnext into x
        x = xnext;
    }

    return x;
}

template<size_t MAXITERS, typename T = double>
auto augmentedLagrangianLM(
    auto obj, auto objAAD,
    auto constraints, auto constraintsAAD,
    const auto& arg, const auto& y,
    const Eigen::Vector<T, Eigen::Dynamic>& x0)
{
    using namespace Eigen;
    using namespace std;
    using namespace optimization;
    using namespace aad;

    using VT = Vector<T, Dynamic>;
    using MT = Matrix<T, Dynamic, Dynamic>;

    double mu = 1;
    const auto M = constraints(x0).size();
    const auto N = obj(arg, x0).size();
    VT x = x0, z = VectorXd::Zero(M);

    auto FAAD = [&](const auto& arg, const Vector<Number<T>, Dynamic>& x) {
        auto f = objAAD(arg, x);
        auto g = constraintsAAD(x);
        Vector<Number<T>, Dynamic> r(f.size() + g.size());
        for (auto i = 0; i < f.size(); ++i) {
            r(i) = f(i);
        }
        for (auto i = 0; i < g.size(); ++i) {
            r(f.size() + i) = sqrt(mu) * g(i) + z(i)/(2.0 * sqrt(mu));
        }
        return r;
    };
    auto F = [&](const auto& arg, const VectorXd& x) {
        auto f = obj(arg, x);
        auto g = constraints(x);
        VectorXd r(f.size() + g.size());
        for (auto i = 0; i < f.size(); ++i) {
            r(i) = f(i);
        }
        for (auto i = 0; i < g.size(); ++i) {
            r(f.size() + i) = sqrt(mu) * g(i) + z(i)/(2.0 * sqrt(mu));
        }
        return r;
    };
    auto jac = [&](const VectorXd& arg, const VectorXd& p) {
        Vector<Number<double>, Dynamic> params(p.size());
        for (auto k = 0; k < p.size(); ++k) {
            params(k) = p(k);
        }
        auto yhat = FAAD(arg, params);
        for (auto k = 0; k < yhat.size(); ++k) {
            yhat(k).adjoint(k) = 1;
        }
        Number<T>::propagateMulti(std::prev(Number<T>::tape->end()), Number<T>::tape->begin());
        Matrix<T, Eigen::Dynamic, Eigen::Dynamic> J(yhat.size(), params.size());
        for (auto k = 0; k < yhat.size(); ++k) {
            for (auto j = 0; j < params.size(); ++j) {
                J(k, j) = params(j).adjoint(k);
            }
        }
        return J;
    };

    auto resetter = aad::setAdjointMultiplicity<double>(true, M+N);

    Number<T>::tape->reset();
    Vector<Number<T>, Dynamic> params(x.size());
    for (auto i = 0; i < x.size(); ++i) {
        params(i) = Number<T>(x(i));
    }
    Number<T>::tape->mark();

    for (auto i = 0u; i < MAXITERS; ++i) {
        Number<T>::tape->rewindToMark();
        Number<T>::tape->reset();

        // evaluate the function to compute its Jacobian
        auto xhat = levenbergMarquardt<1000>(y, arg, F, jac, x, 1e-8, 1e-8);

        auto gx = constraints(x).norm();
        auto gxnext = constraints(xhat.x).norm();

        z -= mu * constraints(xhat.x);
        if (gxnext >= .25 * gx) {
            mu *= 2;
        }
        x = xhat.x;
    }


//

    return x;
}

}