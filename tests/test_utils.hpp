#pragma once

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>

#include "uniform_mesh.hpp"

template <class T>
bool compare(EulerPrimArrays<T> const& ref, EulerPrimArrays<T> const& vec, double tol, int idx)
{
    if (std::abs(ref.d(idx) - vec.d(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.p(idx) - vec.p(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.ux0(idx) - vec.ux0(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.ux1(idx) - vec.ux1(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.ux2(idx) - vec.ux2(idx)) > tol) {
        return false;
    }

    return true;
}

template <class T>
bool compare(EulerConsArrays<T> const& ref, EulerConsArrays<T> const& vec, double tol, int idx)
{
    if (std::abs(ref.d(idx) - vec.d(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.e(idx) - vec.e(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.mx0(idx) - vec.mx0(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.mx1(idx) - vec.mx1(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.mx2(idx) - vec.mx2(idx)) > tol) {
        return false;
    }

    return true;
}

template <class T>
auto copy_to_host(EulerConsArrays<T> const& src)
{
    return EulerConsArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.d),
            .e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.e),
            .mx0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.mx0),
            .mx1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.mx1),
            .mx2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.mx2)};
}

template <class T>
auto copy_to_host(EulerPrimArrays<T> const& src)
{
    return EulerPrimArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.d),
            .p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.p),
            .ux0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.ux0),
            .ux1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.ux1),
            .ux2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.ux2)};
}

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void init_ramp_state(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        UniformMesh3d<T> const& mesh)
{
    T const dx0 = mesh.dx0();
    T const dx1 = mesh.dx1();
    T const dx2 = mesh.dx2();

    T const ux0 = 0.5;
    T const ux1 = -0.2;
    T const ux2 = 0.1;

    T const d_base = 1.0;
    T const p_base = 1.0;
    T const d_ramp = 0.02;
    T const p_ramp = 0.01;

    Kokkos::parallel_for(
            "init_ramp_state",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {0, 0, 0},
                    {prim_arrays.d.extent(0), prim_arrays.d.extent(1), prim_arrays.d.extent(2)}),

            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                // map index to physical coordinate
                T const x0 = (((i + 0.5) * dx0) - 0.5);
                // map x0 to [0,1]
                T const xi = x0 + 0.5;

                // use smoothstep cubic Hermite interpolation
                T const ramp = xi * xi * (3 - (2 * xi)); // smoothstep

                T const d = 1.0 + (d_ramp * ramp);
                T const p = 1.0 + (p_ramp * ramp);

                EulerPrim<T> const prim {.d = d, .p = p, .ux0 = ux0, .ux1 = ux1, .ux2 = ux2};

                store(prim, prim_arrays, i, j, k);
            });
}
