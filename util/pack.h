#pragma once

#include <cstdint>
#include <cstddef>
#include <utility>
#include <tuple>

namespace optimization
{

namespace detail {

template<typename T, std::size_t ... Is>
auto pack(const T &v, std::index_sequence<Is...>)
{
    return std::make_tuple(v(Is)...);
}
}

}
