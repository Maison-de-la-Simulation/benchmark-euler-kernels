#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <Kokkos_SIMD_Common.hpp>
#include <Kokkos_SIMD_Scalar.hpp>
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
    using simd_t = KE::simd<double>;
    constexpr IndexType simd_width = simd_t::size();

    IndexType const nx = prim_arrays.d.extent(0);
    IndexType const ny = prim_arrays.d.extent(1);
    IndexType const nz = prim_arrays.d.extent(2);
    IndexType const simd_end = (nx / simd_width) * simd_width;

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

    simd_t const gamma_minus_one_inv = 1 / (eos.gamma() - 1); // hoist EOS constant
    Kokkos::parallel_for(
            "prim_to_cons_simd",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(exec_space, {0, 0, 0}, {nx / simd_width, ny, nz}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                IndexType const base = (i * simd_width) + (nx * j) + (nx * ny * k);

                simd_t const d(pd + base, KE::simd_flag_default);
                simd_t const p(pp + base, KE::simd_flag_default);
                simd_t const ux0(pu0 + base, KE::simd_flag_default);
                simd_t const ux1(pu1 + base, KE::simd_flag_default);
                simd_t const ux2(pu2 + base, KE::simd_flag_default);
                auto c_simd = {d, p, ux0, ux1, ux2};

                simd_t const int_e = p * gamma_minus_one_inv;

                simd_t const e_kin = d * (ux0 * ux0 + ux1 * ux1 + ux2 * ux2) / 2;

                d.copy_to(cd + base, KE::simd_flag_default);
                (e_kin + int_e).copy_to(ce + base, KE::simd_flag_default);
                (d * ux0).copy_to(cm0 + base, KE::simd_flag_default);
                (d * ux1).copy_to(cm1 + base, KE::simd_flag_default);
                (d * ux2).copy_to(cm2 + base, KE::simd_flag_default);
                // KE::simd_unchecked_store(d, cd + base, KE::simd_flag_default);
                // KE::simd_unchecked_store(e_kin + int_e, ce + base, KE::simd_flag_default);
                // KE::simd_unchecked_store(d * ux0, cm0 + base, KE::simd_flag_default);
                // KE::simd_unchecked_store(d * ux1, cm1 + base, KE::simd_flag_default);
                // KE::simd_unchecked_store(d * ux2, cm2 + base, KE::simd_flag_default);
            });

    // scalar remainder for when nx % simd_width != 0
    // Kokkos::parallel_for(
    //         "prim_to_cons_remainder",
    //         Kokkos::MDRangePolicy<
    //                 Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
    //                 Kokkos::IndexType<IndexType>>(exec_space, {0, 0}, {ny, nz}),
    //         KOKKOS_LAMBDA(IndexType const j, IndexType const k) {
    //             for (IndexType i = simd_end; i < nx; ++i) {
    //                 IndexType const base = i + (nx * j) + (nx * ny * k);
    //                 T const d = pd[base];
    //                 T const p = pp[base];
    //                 T const ux0 = pu0[base];
    //                 T const ux1 = pu1[base];
    //                 T const ux2 = pu2[base];
    //                 T const int_e = p * gamma_minus_one_inv;
    //                 T const e_kin = d * (ux0 * ux0 + ux1 * ux1 + ux2 * ux2) / 2;
    //                 cd[base] = d;
    //                 ce[base] = e_kin + int_e;
    //                 cm0[base] = d * ux0;
    //                 cm1[base] = d * ux1;
    //                 cm2[base] = d * ux2;
    //             }
    //         });
}
