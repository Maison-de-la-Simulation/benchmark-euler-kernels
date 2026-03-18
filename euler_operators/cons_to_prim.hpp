#pragma once

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void cons_to_prim(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerConsArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        EulerPrimArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        PerfectGas<T> const& eos)
{
    Kokkos::parallel_for(
            "cons_to_prim",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {0, 0, 0},
                    {cons_arrays.d.extent(0), cons_arrays.d.extent(1), cons_arrays.d.extent(2)}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                EulerCons const cons = load(cons_arrays, i, j, k);
                EulerPrim const prim = to_prim(cons, eos.pressure(cons.d, internal_energy(cons)));
                store(prim, prim_arrays, i, j, k);
            });
}
