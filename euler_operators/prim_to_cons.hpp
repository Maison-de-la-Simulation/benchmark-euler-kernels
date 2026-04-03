#pragma once

#include <cstdio>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <perfect_gas.hpp>

#include "euler_arrays.hpp"

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
        IndexType nx_begin,
        IndexType nx_end,
        PerfectGas<T> const& eos)
{
    constexpr IndexType width = SimdType::size();
    IndexType const nx_blocks = (nx_end - nx_begin) / width;
    IndexType const nx = prim_arrays.d.extent(0);
    IndexType const ny = prim_arrays.d.extent(1);
    IndexType const nz = prim_arrays.d.extent(2);



    T* cd = cons_arrays.d.data_handle();
    T* ce = cons_arrays.e.data_handle();
    T* cm0 = cons_arrays.mx0.data_handle();
    T* cm1 = cons_arrays.mx1.data_handle();
    T* cm2 = cons_arrays.mx2.data_handle();

    auto const cons_ptrs = EulerConsArrays<T*> {cd, ce, cm0, cm1, cm2};


    Kokkos::parallel_for(
            "prim_to_cons_kernel",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(exec_space, {0, 0, 0}, {nx_blocks, ny, nz}),
            KOKKOS_LAMBDA(IndexType bi, IndexType j, IndexType k) {
                IndexType const base = prim_arrays.d.mapping()(nx_begin + bi * width, j, k);

                EulerPrim<SimdType> const prim = load<SimdType>(prim_arrays, base);

                EulerCons<SimdType> const cons = to_cons(prim, eos.internal_energy(prim.d, prim.p));


                store<SimdType>(cons, cons_ptrs, base);
            });
}

template <class SimdType, class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void prim_to_cons_kernel_inline(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        IndexType nx_begin,
        IndexType nx_end,
        PerfectGas<T> const& eos)
{
    namespace KE = Kokkos::Experimental;
    constexpr IndexType width = SimdType::size();
    IndexType const nx_blocks = (nx_end - nx_begin) / width;
    IndexType const nx = prim_arrays.d.extent(0);
    IndexType const ny = prim_arrays.d.extent(1);
    IndexType const nz = prim_arrays.d.extent(2);

    T const* pd = prim_arrays.d.data_handle();
    T const* pp = prim_arrays.p.data_handle();
    T const* pu0 = prim_arrays.ux0.data_handle();
    T const* pu1 = prim_arrays.ux1.data_handle();
    T const* pu2 = prim_arrays.ux2.data_handle();
    T* cd = cons_arrays.d.data_handle();
    T* ce = cons_arrays.e.data_handle();
    T* cm0 = cons_arrays.mx0.data_handle();
    T* cm1 = cons_arrays.mx1.data_handle();
    T* cm2 = cons_arrays.mx2.data_handle();

    T gamma_inv_minus_one = 1 / (1.4 - 1);

    // Before the parallel_for
    Kokkos::parallel_for(
            "prim_to_cons_kernel",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(exec_space, {0, 0, 0}, {nx_blocks, ny, nz}),
            KOKKOS_LAMBDA(IndexType bi, IndexType j, IndexType k) {
                IndexType const base = (nx_begin + bi * width) + nx * j + nx * ny * k;
                SimdType d = pd[base];
                SimdType p = pp[base];
                SimdType ux0 = pu0[base];
                SimdType ux1 = pu1[base];
                SimdType ux2 = pu2[base];
                SimdType m0 = d * ux0;
                SimdType m1 = d * ux1;
                SimdType m2 = d * ux2;
                SimdType e_int = p * SimdType(gamma_inv_minus_one);
                SimdType e_kin = (m0 * ux0 + m1 * ux1 + m2 * ux2) * SimdType(0.5);
                SimdType e_tot = e_int + e_kin;

                KE::simd_unchecked_store(d, cd + base, KE::simd_flag_default);

                KE::simd_unchecked_store(e_tot, ce + base, KE::simd_flag_default);
                KE::simd_unchecked_store(m0, cm0 + base, KE::simd_flag_default);
                KE::simd_unchecked_store(m1, cm1 + base, KE::simd_flag_default);
                KE::simd_unchecked_store(m2, cm2 + base, KE::simd_flag_default);
            });
    exec_space.fence();
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

    IndexType const nx = prim_arrays.d.extent(0);
    IndexType const vec_end = (nx / simd_t::size()) * simd_t::size();


    prim_to_cons_kernel_inline<
            simd_t>(exec_space, prim_arrays, cons_arrays, IndexType(0), vec_end, eos);

    if (vec_end < nx) {
        prim_to_cons_kernel_inline<
                simd_scalar_t>(exec_space, prim_arrays, cons_arrays, vec_end, nx, eos);
    }
}
