#pragma once

#include <cstddef>

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void prim_to_cons(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        PerfectGas<T> const& eos)
{
    Kokkos::parallel_for(
            "prim_to_cons",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {0, 0, 0},
                    {prim_arrays.d.extent(0), prim_arrays.d.extent(1), prim_arrays.d.extent(2)}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                EulerPrim const prim = load(prim_arrays, i, j, k);
                EulerCons const cons = to_cons(prim, eos.internal_energy(prim.d, prim.p));
                store(cons, cons_arrays, i, j, k);
            });
}

template <class SimdType, class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void prim_to_cons_kernel(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        IndexType n0_begin,
        IndexType n0_end,
        PerfectGas<T> const& eos)
{
    constexpr IndexType width = SimdType::size();
    IndexType const n0_blocks = (n0_end - n0_begin) / width;
    IndexType const n1 = prim_arrays.d.extent(1);
    IndexType const n2 = prim_arrays.d.extent(2);

    Kokkos::layout_left::mapping const common_mapping = prim_arrays.d.mapping();
    EulerPrimArrays const prim_ptrs = data_handle(prim_arrays);
    EulerConsArrays const cons_ptrs = data_handle(cons_arrays);

    Kokkos::parallel_for(
            "prim_to_cons_kernel",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(exec_space, {0, 0, 0}, {n0_blocks, n1, n2}),
            KOKKOS_LAMBDA(IndexType bi, IndexType j, IndexType k) {
                IndexType const base = common_mapping(n0_begin + (bi * width), j, k);
                EulerPrim const prim = load<SimdType>(prim_ptrs, base);
                EulerCons const cons = to_cons(prim, eos.internal_energy(prim.d, prim.p));
                store<SimdType>(cons, cons_ptrs, base);
            });
}

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void prim_to_cons_vec(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        PerfectGas<T> const& eos)
{
    namespace KE = Kokkos::Experimental;
    using simd_t = KE::simd<T>;
    using simd_scalar_t = KE::basic_simd<T, KE::simd_abi::scalar>;

    IndexType const n0 = prim_arrays.d.extent(0);
    IndexType const vec_end = (n0 / simd_t::size()) * simd_t::size();

    prim_to_cons_kernel<simd_t>(exec_space, prim_arrays, cons_arrays, IndexType(0), vec_end, eos);

    if (vec_end < n0) {
        prim_to_cons_kernel<simd_scalar_t>(exec_space, prim_arrays, cons_arrays, vec_end, n0, eos);
    }
}
