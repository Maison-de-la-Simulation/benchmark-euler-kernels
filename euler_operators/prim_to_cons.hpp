#pragma once

#include <cstddef>
#include <span>

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
                Kokkos::Experimental::layout_left_padded<std::dynamic_extent>>> const& prim_arrays,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::Experimental::layout_left_padded<std::dynamic_extent>>> const& cons_arrays,
        PerfectGas<T> const& eos)
{
    constexpr IndexType width = SimdType::size();
    IndexType const n0_blocks = prim_arrays.d.extent(0) / width;
    IndexType const n1 = prim_arrays.d.extent(1);
    IndexType const n2 = prim_arrays.d.extent(2);

    auto const common_mapping = prim_arrays.d.mapping();
    EulerPrimArrays const prim_ptrs = data_handle(prim_arrays);
    EulerConsArrays const cons_ptrs = data_handle(cons_arrays);

    Kokkos::parallel_for(
            "prim_to_cons_kernel",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(exec_space, {0, 0, 0}, {n0_blocks, n1, n2}),
            KOKKOS_LAMBDA(IndexType bi, IndexType j, IndexType k) {
                IndexType const base = common_mapping(bi * width, j, k);
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

    Kokkos::full_extent_t const slice1;
    Kokkos::full_extent_t const slice2;
    {
        Kokkos::pair const slice0(0, vec_end);
        EulerPrimArrays const sub_prim_arrays = subspan(prim_arrays, slice0, slice1, slice2);
        EulerConsArrays const sub_cons_arrays = subspan(cons_arrays, slice0, slice1, slice2);
        prim_to_cons_kernel<simd_t>(exec_space, sub_prim_arrays, sub_cons_arrays, eos);
    }
    if (vec_end < n0) {
        Kokkos::pair const slice0(vec_end, n0);
        EulerPrimArrays const sub_prim_arrays = subspan(prim_arrays, slice0, slice1, slice2);
        EulerConsArrays const sub_cons_arrays = subspan(cons_arrays, slice0, slice1, slice2);
        prim_to_cons_kernel<simd_scalar_t>(exec_space, sub_prim_arrays, sub_cons_arrays, eos);
    }
}
