#pragma once

#include <cstddef>
#include <type_traits>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MinMax.hpp>
#include <Kokkos_SIMD.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>

template <std::size_t Dir, class T>
KOKKOS_FUNCTION T get(std::integral_constant<std::size_t, Dir> /*unused*/, EulerPrim<T> const& prim)
{
    static_assert(Dir < 3);
    if constexpr (Dir == 0) {
        return prim.ux0;
    } else if constexpr (Dir == 1) {
        return prim.ux1;
    } else if constexpr (Dir == 2) {
        return prim.ux2;
    }
}

template <std::size_t Dir, class T>
KOKKOS_FUNCTION T get(std::integral_constant<std::size_t, Dir> /*unused*/, EulerCons<T> const& cons)
{
    static_assert(Dir < 3);
    if constexpr (Dir == 0) {
        return cons.mx0;
    } else if constexpr (Dir == 1) {
        return cons.mx1;
    } else if constexpr (Dir == 2) {
        return cons.mx2;
    }
}

template <std::size_t Dir, class T>
KOKKOS_FUNCTION T get(std::integral_constant<std::size_t, Dir> /*unused*/, EulerFlux<T> const& flux)
{
    static_assert(Dir < 3);
    if constexpr (Dir == 0) {
        return flux.mx0;
    } else if constexpr (Dir == 1) {
        return flux.mx1;
    } else if constexpr (Dir == 2) {
        return flux.mx2;
    }
}

namespace detail {

// scalar
template <class T>
KOKKOS_FUNCTION T select(bool cond, T const& a, T const& b)
{
    return cond ? a : b;
}

// simd
template <class Mask, class T>
KOKKOS_FUNCTION T select(Mask const& mask, T const& a, T const& b)
{
    return Kokkos::Experimental::condition(mask, a, b);
}

} // namespace detail

struct hllc
{
    template <std::size_t Dir, class T, class U>
    KOKKOS_FUNCTION EulerFlux<T> operator()(
            std::integral_constant<std::size_t, Dir> dir,
            PerfectGas<U> const& eos,
            EulerPrim<T> const& q_L,
            EulerPrim<T> const& q_R) const noexcept
    {
        static_assert(Dir < 3);

        using detail::select;

        T const un_L = get(dir, q_L);
        T const un_R = get(dir, q_R);

        T const c_L = eos.speed_of_sound(q_L.d, q_L.p);
        T const c_R = eos.speed_of_sound(q_R.d, q_R.p);

        T const S_L = Kokkos::min(un_L, un_R) - Kokkos::max(c_L, c_R);
        T const S_R = Kokkos::max(un_L, un_R) + Kokkos::max(c_L, c_R);

        T const rc_L = q_L.d * (S_L - un_L);
        T const rc_R = q_R.d * (S_R - un_R);

        // Compute acoustic star states
        T const ustar = (q_R.p - q_L.p + (rc_L * un_L) - (rc_R * un_R)) / (rc_L - rc_R);
        T const pstar = 0.5 * (q_L.p + q_R.p + (rc_L * (ustar - un_L)) + (rc_R * (ustar - un_R)));

        // Conditions (scalar -> bool, SIMD -> mask)
        auto const cond_ustar = ustar > 0;
        auto const cond_SR = S_L * S_R > 0;

        // Select wave speed and state
        T const S = select(cond_ustar, S_L, S_R);

        EulerPrim<T> q {};
        q.d = select(cond_ustar, q_L.d, q_R.d);
        q.p = select(cond_ustar, q_L.p, q_R.p);
        q.ux0 = select(cond_ustar, q_L.ux0, q_R.ux0);
        q.ux1 = select(cond_ustar, q_L.ux1, q_R.ux1);
        q.ux2 = select(cond_ustar, q_L.ux2, q_R.ux2);

        T const un = get(dir, q);
        T const etot = eos.internal_energy(q.d, q.p) + kinetic_energy(q);

        // Output states
        T const un_o = select(cond_SR, un, ustar);
        T const ptot_o = select(cond_SR, q.p, pstar);

        T const d_o = (S - un) / (S - un_o) * q.d;

        T const etot_o
                = ((S - un) / (S - un_o) * etot) + (((ptot_o * un_o) - (q.p * un)) / (S - ustar));

        EulerFlux<T> flux {};

        flux.d = d_o * un_o;
        flux.e = (etot_o + ptot_o) * un_o;

        flux.mx0 = d_o * un_o * q.ux0;
        flux.mx1 = d_o * un_o * q.ux1;
        flux.mx2 = d_o * un_o * q.ux2;

        if constexpr (Dir == 0) {
            flux.mx0 = (d_o * un_o * un_o) + ptot_o;
        } else if constexpr (Dir == 1) {
            flux.mx1 = (d_o * un_o * un_o) + ptot_o;
        } else {
            flux.mx2 = (d_o * un_o * un_o) + ptot_o;
        }

        return flux;
    }
};
