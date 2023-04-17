#pragma once

#include <tuple>
#include <functional>
#include <Eigen/Dense>
#include <optimization/util/pack.h>

namespace optimization
{

template<typename R, typename T>
auto apply(std::function<R(const Eigen::VectorXd&)> f, const T& args)
{
    return f(args);
}

template<typename T, typename DT, typename R, typename ... Args>
auto apply(std::function<R(DT, Args...)> f, const DT& data, Eigen::Vector<T, sizeof...(Args)> args)
{
    using namespace std;
    using detail::pack;
    return std::apply(f, std::forward<tuple<Args...>>(pack(args, std::make_index_sequence<sizeof...(Args)>{})));
}

#if 0
template<typename T, typename R, typename ... Args1, typename ... Args2>
auto apply(std::function<R(Args1..., Args2...)> f,
    Eigen::Vector<T, sizeof...(Args1)> args1,
    Eigen::Vector<T, sizeof...(Args2)> args2)
{
    using namespace std;
    using detail::pack;

    static constexpr const size_t N1 = sizeof...(Args1);
    static constexpr const size_t N2 = sizeof...(Args2);

    return std::apply(f,
        tuple_cat(
            pack(args1, std::make_index_sequence<N1>{}),
            pack(args2, std::make_index_sequence<N2>{})));
}
#endif
}
