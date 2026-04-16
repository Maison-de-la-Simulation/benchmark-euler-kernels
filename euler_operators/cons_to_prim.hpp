#pragma once

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>

#include "utils.hpp"

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

template <class SimdType, class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void cons_to_prim_kernel(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerConsArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        EulerPrimArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        IndexType nx_begin,
        IndexType nx_end,
        PerfectGas<T> const& eos)
{
    constexpr IndexType width = SimdType::size();
    IndexType const nx_blocks = (nx_end - nx_begin) / width;
    IndexType const nx = cons_arrays.d.extent(0);
    IndexType const ny = cons_arrays.d.extent(1);
    IndexType const nz = cons_arrays.d.extent(2);

    T* pd = prim_arrays.d.data_handle();
    T* pp = prim_arrays.p.data_handle();
    T* pu0 = prim_arrays.ux0.data_handle();
    T* pu1 = prim_arrays.ux1.data_handle();
    T* pu2 = prim_arrays.ux2.data_handle();

    auto const prim_ptrs = EulerPrimArrays<T*> {pd, pp, pu0, pu1, pu2};

    Kokkos::parallel_for(
            "cons_to_prim_kernel",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(exec_space, {0, 0, 0}, {nx_blocks, ny, nz}),

            KOKKOS_LAMBDA(IndexType bi, IndexType j, IndexType k) {
                IndexType const i = nx_begin + bi * width;
                IndexType const base = cons_arrays.d.mapping()(i, j, k);

                EulerCons<SimdType> const cons = load<SimdType>(cons_arrays, base);

                EulerPrim<SimdType> prim
                        = to_prim(cons, eos.pressure(cons.d, internal_energy(cons)));
                store<SimdType>(prim, prim_ptrs, base);
            }

    );
}


template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void cons_to_prim_vec(
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
    namespace KE = Kokkos::Experimental;
    using simd_t = KE::simd<T>;
    using simd_scalar_t = KE::basic_simd<T, KE::simd_abi::scalar>;

    IndexType const nx = cons_arrays.d.extent(0);
    IndexType const vec_end = (nx / simd_t::size()) * simd_t::size();

    cons_to_prim_kernel<simd_t>(exec_space, cons_arrays, prim_arrays, IndexType(0), vec_end, eos);
    if (vec_end < nx) {
        cons_to_prim_kernel<simd_scalar_t>(exec_space, cons_arrays, prim_arrays, vec_end, nx, eos);
    }
}
