#pragma once

#include <cstddef>
#include <span>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <euler_arrays.hpp>
#include <hllc.hpp>
#include <perfect_gas.hpp>
#include <uniform_mesh.hpp>

template <std::size_t N>
using dir_t = std::integral_constant<std::size_t, N>;

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void godunov(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        PerfectGas<T> const& eos,
        UniformMesh3d<T> const& mesh,
        hllc const& riemann_solver,
        T const dt)
{
    Kokkos::Array<T, 3> const ds = {mesh.ds0(), mesh.ds1(), mesh.ds2()};
    T const dtodv = dt / mesh.dv();

    Kokkos::parallel_for(
            "godunov",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {1, 1, 1},
                    {1 + (prim_arrays.d.extent(0) - 2),
                     1 + (prim_arrays.d.extent(1) - 2),
                     1 + (prim_arrays.d.extent(2) - 2)}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                EulerPrim const prim = load(prim_arrays, i, j, k);
                EulerFlux<T> flux {};

                {
                    EulerPrim const prim_L = load(prim_arrays, i - 1, j, k);
                    EulerPrim const prim_R = load(prim_arrays, i + 1, j, k);
                    EulerFlux const flux_L = riemann_solver(dir_t<0>(), eos, prim_L, prim);
                    EulerFlux const flux_R = riemann_solver(dir_t<0>(), eos, prim, prim_R);
                    flux.d += ds[0] * (flux_R.d - flux_L.d);
                    flux.e += ds[0] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[0] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[0] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[0] * (flux_R.mx2 - flux_L.mx2);
                }
                {
                    EulerPrim const prim_L = load(prim_arrays, i, j - 1, k);
                    EulerPrim const prim_R = load(prim_arrays, i, j + 1, k);
                    EulerFlux const flux_L = riemann_solver(dir_t<1>(), eos, prim_L, prim);
                    EulerFlux const flux_R = riemann_solver(dir_t<1>(), eos, prim, prim_R);
                    flux.d += ds[1] * (flux_R.d - flux_L.d);
                    flux.e += ds[1] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[1] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[1] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[1] * (flux_R.mx2 - flux_L.mx2);
                }
                {
                    EulerPrim const prim_L = load(prim_arrays, i, j, k - 1);
                    EulerPrim const prim_R = load(prim_arrays, i, j, k + 1);
                    EulerFlux const flux_L = riemann_solver(dir_t<2>(), eos, prim_L, prim);
                    EulerFlux const flux_R = riemann_solver(dir_t<2>(), eos, prim, prim_R);
                    flux.d += ds[2] * (flux_R.d - flux_L.d);
                    flux.e += ds[2] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[2] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[2] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[2] * (flux_R.mx2 - flux_L.mx2);
                }

                EulerCons cons = load(cons_arrays, i, j, k);
                cons.d -= dtodv * flux.d;
                cons.e -= dtodv * flux.e;
                cons.mx0 -= dtodv * flux.mx0;
                cons.mx1 -= dtodv * flux.mx1;
                cons.mx2 -= dtodv * flux.mx2;
                store(cons, cons_arrays, i, j, k);
            });
}

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void godunov_opti(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        PerfectGas<T> const& eos,
        UniformMesh3d<T> const& mesh,
        hllc const& riemann_solver,
        T const dt)
{
    Kokkos::Array<T, 3> const ds = {mesh.ds0(), mesh.ds1(), mesh.ds2()};
    T const dtodv = dt / mesh.dv();

    Kokkos::layout_left::mapping const common_mapping = prim_arrays.d.mapping();
    IndexType const stride_1 = prim_arrays.d.extent(0);
    IndexType const stride_2 = prim_arrays.d.extent(0) * prim_arrays.d.extent(1);

    EulerPrimArrays const prim_ptrs = data_handle(prim_arrays);
    EulerConsArrays const cons_ptrs = data_handle(cons_arrays);

    Kokkos::parallel_for(
            "godunov_opti",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {1, 1, 1},
                    {1 + (prim_arrays.d.extent(0) - 2),
                     1 + (prim_arrays.d.extent(1) - 2),
                     1 + (prim_arrays.d.extent(2) - 2)}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                IndexType const base = common_mapping(i, j, k);
                EulerPrim const prim = load(prim_ptrs, base);
                EulerFlux<T> flux {};

                {
                    EulerPrim const prim_L = load(prim_ptrs, base - 1);
                    EulerPrim const prim_R = load(prim_ptrs, base + 1);
                    EulerFlux const flux_L = riemann_solver(dir_t<0>(), eos, prim_L, prim);
                    EulerFlux const flux_R = riemann_solver(dir_t<0>(), eos, prim, prim_R);
                    flux.d += ds[0] * (flux_R.d - flux_L.d);
                    flux.e += ds[0] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[0] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[0] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[0] * (flux_R.mx2 - flux_L.mx2);
                }
                {
                    EulerPrim const prim_L = load(prim_ptrs, base - stride_1);
                    EulerPrim const prim_R = load(prim_ptrs, base + stride_1);
                    EulerFlux const flux_L = riemann_solver(dir_t<1>(), eos, prim_L, prim);
                    EulerFlux const flux_R = riemann_solver(dir_t<1>(), eos, prim, prim_R);
                    flux.d += ds[1] * (flux_R.d - flux_L.d);
                    flux.e += ds[1] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[1] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[1] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[1] * (flux_R.mx2 - flux_L.mx2);
                }
                {
                    EulerPrim const prim_L = load(prim_ptrs, base - stride_2);
                    EulerPrim const prim_R = load(prim_ptrs, base + stride_2);
                    EulerFlux const flux_L = riemann_solver(dir_t<2>(), eos, prim_L, prim);
                    EulerFlux const flux_R = riemann_solver(dir_t<2>(), eos, prim, prim_R);
                    flux.d += ds[2] * (flux_R.d - flux_L.d);
                    flux.e += ds[2] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[2] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[2] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[2] * (flux_R.mx2 - flux_L.mx2);
                }

                EulerCons cons = load(cons_ptrs, base);
                cons.d -= dtodv * flux.d;
                cons.e -= dtodv * flux.e;
                cons.mx0 -= dtodv * flux.mx0;
                cons.mx1 -= dtodv * flux.mx1;
                cons.mx2 -= dtodv * flux.mx2;
                store(cons, cons_ptrs, base);
            });
}

template <class SimdType, class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void godunov_kernel(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::Experimental::layout_left_padded<std::dynamic_extent>>> const& prim_arrays,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::Experimental::layout_left_padded<std::dynamic_extent>>> const& cons_arrays,

        PerfectGas<T> const& eos,
        UniformMesh3d<T> const& mesh,

        hllc const& riemann_solver,
        T const dt)
{
    constexpr IndexType width = SimdType::size();

    // do not include most left and right neighbors in loop
    IndexType const n0_blocks = (prim_arrays.d.extent(0) - 2) / width;
    IndexType const n1 = prim_arrays.d.extent(1);
    IndexType const n2 = prim_arrays.d.extent(2);

    Kokkos::Array<T, 3> const ds = {mesh.ds0(), mesh.ds1(), mesh.ds2()};
    T const dtodv = dt / mesh.dv();

    auto const common_mapping = prim_arrays.d.mapping();
    IndexType const stride_1 = common_mapping.stride(1);
    IndexType const stride_2 = common_mapping.stride(2);
    EulerPrimArrays const prim_ptrs = data_handle(prim_arrays);
    EulerConsArrays const cons_ptrs = data_handle(cons_arrays);

    Kokkos::parallel_for(
            "godunov_kernel",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<
                            IndexType>>(exec_space, {0, 1, 1}, {n0_blocks, n1 - 1, n2 - 1}),
            KOKKOS_LAMBDA(IndexType const bi, IndexType const j, IndexType const k) {
                IndexType const base = common_mapping(1 + (bi * width), j, k);
                EulerPrim<SimdType> const prim = load<SimdType>(prim_ptrs, base);
                EulerFlux<SimdType> flux {};

                {
                    EulerPrim const prim_L = load<SimdType>(prim_ptrs, base - 1);
                    EulerPrim const prim_R = load<SimdType>(prim_ptrs, base + 1);
                    EulerFlux const flux_L = riemann_solver(dir_t<0>(), eos, prim_L, prim);
                    EulerFlux const flux_R = riemann_solver(dir_t<0>(), eos, prim, prim_R);
                    flux.d += ds[0] * (flux_R.d - flux_L.d);
                    flux.e += ds[0] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[0] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[0] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[0] * (flux_R.mx2 - flux_L.mx2);
                }
                {
                    EulerPrim const prim_L = load<SimdType>(prim_ptrs, base - stride_1);
                    EulerPrim const prim_R = load<SimdType>(prim_ptrs, base + stride_1);
                    EulerFlux const flux_L = riemann_solver(dir_t<1>(), eos, prim_L, prim);
                    EulerFlux const flux_R = riemann_solver(dir_t<1>(), eos, prim, prim_R);
                    flux.d += ds[1] * (flux_R.d - flux_L.d);
                    flux.e += ds[1] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[1] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[1] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[1] * (flux_R.mx2 - flux_L.mx2);
                }
                {
                    EulerPrim const prim_L = load<SimdType>(prim_ptrs, base - stride_2);
                    EulerPrim const prim_R = load<SimdType>(prim_ptrs, base + stride_2);
                    EulerFlux const flux_L = riemann_solver(dir_t<2>(), eos, prim_L, prim);
                    EulerFlux const flux_R = riemann_solver(dir_t<2>(), eos, prim, prim_R);
                    flux.d += ds[2] * (flux_R.d - flux_L.d);
                    flux.e += ds[2] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[2] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[2] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[2] * (flux_R.mx2 - flux_L.mx2);
                }

                EulerCons cons = load<SimdType>(cons_ptrs, base);
                cons.d -= dtodv * flux.d;
                cons.e -= dtodv * flux.e;
                cons.mx0 -= dtodv * flux.mx0;
                cons.mx1 -= dtodv * flux.mx1;
                cons.mx2 -= dtodv * flux.mx2;
                store<SimdType>(cons, cons_ptrs, base);
            });
}

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void godunov_vec(
        Kokkos::DefaultExecutionSpace const& exec_space,
        EulerPrimArrays<Kokkos::mdspan<
                T const,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& prim_arrays,
        EulerConsArrays<Kokkos::mdspan<
                T,
                Kokkos::extents<IndexType, E0, E1, E2>,
                Kokkos::layout_left>> const& cons_arrays,
        PerfectGas<T> const& eos,
        UniformMesh3d<T> const& mesh,
        hllc const& riemann_solver,
        T const dt)
{
    namespace KE = Kokkos::Experimental;
    using simd_t = KE::simd<T>;
    using simd_scalar_t = KE::basic_simd<T, KE::simd_abi::scalar>;

    // interior x-range is [1, n0-1)
    IndexType const n0 = prim_arrays.d.extent(0);
    IndexType const n0_begin = 1;
    IndexType const n0_inner = n0 - 2; // number of interior cells
    IndexType const vec_end = n0_begin + ((n0_inner / simd_t::size()) * simd_t::size());

    IndexType const n0_end = n0 - 1;

    Kokkos::full_extent_t const slice1;
    Kokkos::full_extent_t const slice2;
    {
        Kokkos::pair const slice0(0, vec_end + 1); // include right most neighbor
        EulerPrimArrays const sub_prim_arrays = subspan(prim_arrays, slice0, slice1, slice2);
        EulerConsArrays const sub_cons_arrays = subspan(cons_arrays, slice0, slice1, slice2);
        godunov_kernel<simd_t>(
                exec_space,
                sub_prim_arrays,
                sub_cons_arrays,
                eos,
                mesh,
                riemann_solver,
                dt);
    }

    if (vec_end < n0_end) {
        Kokkos::pair const slice0(vec_end - 1, n0); // include left most neighbor
        EulerPrimArrays const sub_prim_arrays = subspan(prim_arrays, slice0, slice1, slice2);
        EulerConsArrays const sub_cons_arrays = subspan(cons_arrays, slice0, slice1, slice2);

        godunov_kernel<simd_scalar_t>(
                exec_space,
                sub_prim_arrays,
                sub_cons_arrays,
                eos,
                mesh,
                riemann_solver,
                dt);
    }
}
