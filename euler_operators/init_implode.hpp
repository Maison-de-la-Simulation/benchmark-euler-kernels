#pragma once

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <uniform_mesh.hpp>

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void init_implode(
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
    T const density_in = 1.;
    T const density_out = 0.125;
    T const pressure_in = 1.;
    T const pressure_out = 0.14;
    T const radius2 = 0.25 * 0.25;
    Kokkos::parallel_for(
            "implode_nd",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {0, 0, 0},
                    {prim_arrays.d.extent(0), prim_arrays.d.extent(1), prim_arrays.d.extent(2)}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                T const c_x0 = (((i - 1) + 0.5) * dx0) - 0.5;
                T const c_x1 = (((j - 1) + 0.5) * dx1) - 0.5;
                T const c_x2 = (((k - 1) + 0.5) * dx2) - 0.5;
                T const r2 = (c_x0 * c_x0) + (c_x1 * c_x1) + (c_x2 * c_x2);
                EulerPrim<T> const
                        cons {.d = r2 < radius2 ? density_in : density_out,
                              .p = r2 < radius2 ? pressure_in : pressure_out,
                              .ux0 = 0,
                              .ux1 = 0,
                              .ux2 = 0};
                store(cons, prim_arrays, i, j, k);
            });
}
