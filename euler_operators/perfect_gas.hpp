#pragma once

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>

template <class T>
class PerfectGas
{
private:
    T m_gamma;

public:
    explicit PerfectGas(T const gamma) : m_gamma(gamma) {}

    KOKKOS_FUNCTION T gamma() const noexcept
    {
        return m_gamma;
    }

    KOKKOS_FUNCTION T speed_of_sound(T const density, T const pressure) const noexcept
    {
        return Kokkos::sqrt(m_gamma * pressure / density);
    }

    KOKKOS_FUNCTION T internal_energy(T const /*density*/, T const pressure) const noexcept
    {
        return pressure / (m_gamma - 1);
    }

    KOKKOS_FUNCTION T pressure(T const /*density*/, T const int_e) const noexcept
    {
        return (m_gamma - 1) * int_e;
    }
};
