#pragma once

#include <Kokkos_Macros.hpp>

template <class T>
class UniformMesh3d
{
private:
    T m_dx0;

    T m_dx1;

    T m_dx2;

public:
    UniformMesh3d(T dx0, T dx1, T dx2) noexcept : m_dx0(dx0), m_dx1(dx1), m_dx2(dx2) {}

    KOKKOS_FUNCTION T dx0() const noexcept
    {
        return m_dx0;
    }

    KOKKOS_FUNCTION T dx1() const noexcept
    {
        return m_dx1;
    }

    KOKKOS_FUNCTION T dx2() const noexcept
    {
        return m_dx2;
    }

    KOKKOS_FUNCTION T ds0() const noexcept
    {
        return m_dx1 * m_dx2;
    }

    KOKKOS_FUNCTION T ds1() const noexcept
    {
        return m_dx2 * m_dx0;
    }

    KOKKOS_FUNCTION T ds2() const noexcept
    {
        return m_dx0 * m_dx1;
    }

    KOKKOS_FUNCTION T dv() const noexcept
    {
        return m_dx0 * m_dx1 * m_dx2;
    }
};
