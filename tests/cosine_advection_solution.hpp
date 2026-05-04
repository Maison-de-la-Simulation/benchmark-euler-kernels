#pragma once

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <uniform_mesh.hpp>

template <class T>
KOKKOS_FUNCTION EulerPrim<T> cosine_advection_solution_1d(T const x, T const time)
{
    T const u = 1.;
    T const d = 1. + (0.1 * Kokkos::sin(2 * Kokkos::numbers::pi * (x - (u * time))));
    T const p = 10.;
    return {.d = d, .p = p, .ux0 = u, .ux1 = 0, .ux2 = 0};
}

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void cosine_advection_solution(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        UniformMesh3d<T> const& mesh,
        T const time,
        int const dir)
{
    T const dx0 = mesh.dx0();
    T const dx1 = mesh.dx1();
    T const dx2 = mesh.dx2();
    Kokkos::parallel_for(
            "cosine_advection_solution",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {0, 0, 0},
                    {prim_arrays.d.extent(0), prim_arrays.d.extent(1), prim_arrays.d.extent(2)}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                T const c_x0 = ((i - 1) + 0.5) * dx0;
                T const c_x1 = ((j - 1) + 0.5) * dx1;
                T const c_x2 = ((k - 1) + 0.5) * dx2;
                T const c_x = ((dir == 0) * c_x0) + ((dir == 1) * c_x1) + ((dir == 2) * c_x2);
                EulerPrim prim = cosine_advection_solution_1d(c_x, time);
                if (dir == 1) {
                    Kokkos::kokkos_swap(prim.ux0, prim.ux1);
                } else if (dir == 2) {
                    Kokkos::kokkos_swap(prim.ux0, prim.ux2);
                }
                store(prim, prim_arrays, i, j, k);
            });
}
