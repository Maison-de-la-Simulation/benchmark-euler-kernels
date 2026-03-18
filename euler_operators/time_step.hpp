#pragma once

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>
#include <uniform_mesh.hpp>

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
T time_step(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        PerfectGas<T> const& eos,
        UniformMesh3d<T> const& mesh)
{
    T const invdx0 = 1 / mesh.dx0();
    T const invdx1 = 1 / mesh.dx1();
    T const invdx2 = 1 / mesh.dx2();
    T dt {};
    Kokkos::parallel_reduce(
            "time_step_exp1_structured",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {0, 0, 0},
                    {prim_arrays.d.extent(0), prim_arrays.d.extent(1), prim_arrays.d.extent(2)}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k, T& dt_loc) {
                EulerPrim const prim = load(prim_arrays, i, j, k);
                T const cs = eos.speed_of_sound(prim.d, prim.p);
                T const cx0 = cs + Kokkos::abs(prim.ux0);
                T const cx1 = cs + Kokkos::abs(prim.ux1);
                T const cx2 = cs + Kokkos::abs(prim.ux2);
                T const invdt = (cx0 * invdx0) + (cx1 * invdx1) + (cx2 * invdx2);
                dt_loc = Kokkos::min(dt_loc, 1 / invdt);
            },
            Kokkos::Min<T>(dt));
    return dt;
}
