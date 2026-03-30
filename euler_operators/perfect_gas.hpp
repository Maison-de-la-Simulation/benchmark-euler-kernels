#pragma once

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

template <class T>
class PerfectGas
{
private:
    T m_gamma;
    T gamma_minus_one_inv;

public:
    explicit PerfectGas(T const gamma) : m_gamma(gamma), gamma_minus_one_inv(1 / (gamma - 1)) {}


    KOKKOS_FUNCTION T speed_of_sound(T const density, T const pressure) const noexcept
    {
        return Kokkos::sqrt(m_gamma * pressure / density);
    }

    template <class S>
    KOKKOS_FUNCTION S internal_energy(S const /*density*/, S const pressure) const noexcept
    {
        // return pressure / (m_gamma - 1);
        return pressure * gamma_minus_one_inv;
    }

    KOKKOS_FUNCTION T pressure(T const /*density*/, T const int_e) const noexcept
    {
        return (m_gamma - 1) * int_e;
    }
};
