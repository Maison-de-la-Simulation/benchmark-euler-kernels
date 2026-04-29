#pragma once

#include <cstddef>
#include <iostream>
#include <type_traits>

#include <Kokkos_Core.hpp>
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
    namespace KE = Kokkos::Experimental;
    using simd_t = KE::simd<T>;

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
template <class SimdType, class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void godunov_kernel(
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
        PerfectGas<T> const& eos,
        UniformMesh3d<T> const& mesh,
        hllc const& riemann_solver,
        T const dt)
{
    constexpr IndexType width = SimdType::size();
    IndexType const nx_blocks = (nx_end - nx_begin) / width;
    IndexType const ny = prim_arrays.d.extent(1);
    IndexType const nz = prim_arrays.d.extent(2);

    // layout_left strides: stride in y = extent(0), stride in z = extent(0)*extent(1)
    IndexType const stride_y = prim_arrays.d.extent(0);
    IndexType const stride_z = prim_arrays.d.extent(0) * prim_arrays.d.extent(1);

    Kokkos::Array<T, 3> const ds = {mesh.ds0(), mesh.ds1(), mesh.ds2()};
    T const dtodv = dt / mesh.dv();

    T* cd = cons_arrays.d.data_handle();
    T* ce = cons_arrays.e.data_handle();
    T* cm0 = cons_arrays.mx0.data_handle();
    T* cm1 = cons_arrays.mx1.data_handle();
    T* cm2 = cons_arrays.mx2.data_handle();

    auto const cons_ptrs = EulerConsArrays<T*> {cd, ce, cm0, cm1, cm2};

    Kokkos::parallel_for(
            "godunov_kernel",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {0, 1, 1},
                    {nx_blocks, ny - 1, nz - 1}), // nx_begin already acouting for ghost cells
            KOKKOS_LAMBDA(IndexType const bi, IndexType const j, IndexType const k) {
                IndexType const base = prim_arrays.d.mapping()(nx_begin + (bi * width), j, k);

                EulerPrim<SimdType> const prim = load<SimdType>(prim_arrays, base);
                EulerFlux<SimdType> flux {};

                {
                    EulerPrim<SimdType> const prim_L = load<SimdType>(prim_arrays, base - 1);
                    EulerPrim<SimdType> const prim_R = load<SimdType>(prim_arrays, base + 1);
                    EulerFlux<SimdType> const flux_L
                            = riemann_solver(dir_t<0>(), eos, prim_L, prim);
                    EulerFlux<SimdType> const flux_R
                            = riemann_solver(dir_t<0>(), eos, prim, prim_R);
                    flux.d += ds[0] * (flux_R.d - flux_L.d);
                    flux.e += ds[0] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[0] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[0] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[0] * (flux_R.mx2 - flux_L.mx2);
                }
                {
                    EulerPrim<SimdType> const prim_L = load<SimdType>(prim_arrays, base - stride_y);
                    EulerPrim<SimdType> const prim_R = load<SimdType>(prim_arrays, base + stride_y);
                    EulerFlux<SimdType> const flux_L
                            = riemann_solver(dir_t<1>(), eos, prim_L, prim);
                    EulerFlux<SimdType> const flux_R
                            = riemann_solver(dir_t<1>(), eos, prim, prim_R);
                    flux.d += ds[1] * (flux_R.d - flux_L.d);
                    flux.e += ds[1] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[1] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[1] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[1] * (flux_R.mx2 - flux_L.mx2);
                }
                {
                    EulerPrim<SimdType> const prim_L = load<SimdType>(prim_arrays, base - stride_z);
                    EulerPrim<SimdType> const prim_R = load<SimdType>(prim_arrays, base + stride_z);
                    EulerFlux<SimdType> const flux_L
                            = riemann_solver(dir_t<2>(), eos, prim_L, prim);
                    EulerFlux<SimdType> const flux_R
                            = riemann_solver(dir_t<2>(), eos, prim, prim_R);
                    flux.d += ds[2] * (flux_R.d - flux_L.d);
                    flux.e += ds[2] * (flux_R.e - flux_L.e);
                    flux.mx0 += ds[2] * (flux_R.mx0 - flux_L.mx0);
                    flux.mx1 += ds[2] * (flux_R.mx1 - flux_L.mx1);
                    flux.mx2 += ds[2] * (flux_R.mx2 - flux_L.mx2);
                }

                EulerCons<SimdType> cons = load<SimdType>(cons_ptrs, base);
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

    // interior x-range is [1, nx-1)
    IndexType const nx = prim_arrays.d.extent(0);
    IndexType const nx_begin = 1;
    IndexType const nx_inner = nx - 2; // number of interior cells
    IndexType const vec_end = nx_begin + ((nx_inner / simd_t::size()) * simd_t::size());
    IndexType const nx_end = nx - 1;

    godunov_kernel<simd_t>(
            exec_space,
            prim_arrays,
            cons_arrays,
            nx_begin,
            vec_end,
            eos,
            mesh,
            riemann_solver,
            dt);

    if (vec_end < nx_end) {
        godunov_kernel<simd_scalar_t>(
                exec_space,
                prim_arrays,
                cons_arrays,
                vec_end,
                nx_end,
                eos,
                mesh,
                riemann_solver,
                dt);
    }
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
        hllc_opti const& riemann_solver,
        T const dt)
{
    T const ds0 = mesh.ds0();
    T const ds1 = mesh.ds1();
    T const ds2 = mesh.ds2();

    T const dtodv = dt / mesh.dv();

    Kokkos::parallel_for(
            "godunov",
            Kokkos::MDRangePolicy<
                    Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Left>,
                    Kokkos::IndexType<IndexType>>(
                    exec_space,
                    {1, 1, 1},
                    {prim_arrays.d.extent(0) - 1,
                     prim_arrays.d.extent(1) - 1,
                     prim_arrays.d.extent(2) - 1}),
            KOKKOS_LAMBDA(IndexType const i, IndexType const j, IndexType const k) {
                EulerPrim<T> const c = load(prim_arrays, i, j, k);

                T fd = T(0);
                T fe = T(0);
                T fx = T(0);
                T fy = T(0);
                T fz = T(0);

                {
                    auto const L = load(prim_arrays, i - 1, j, k);
                    auto const R = load(prim_arrays, i + 1, j, k);

                    auto const FL = riemann_solver(dir_t<0> {}, eos, L, c);
                    auto const FR = riemann_solver(dir_t<0> {}, eos, c, R);

                    fd += ds0 * (FR.d - FL.d);
                    fe += ds0 * (FR.e - FL.e);
                    fx += ds0 * (FR.mx0 - FL.mx0);
                    fy += ds0 * (FR.mx1 - FL.mx1);
                    fz += ds0 * (FR.mx2 - FL.mx2);
                }

                {
                    auto const L = load(prim_arrays, i, j - 1, k);
                    auto const R = load(prim_arrays, i, j + 1, k);

                    auto const FL = riemann_solver(dir_t<1> {}, eos, L, c);
                    auto const FR = riemann_solver(dir_t<1> {}, eos, c, R);

                    fd += ds1 * (FR.d - FL.d);
                    fe += ds1 * (FR.e - FL.e);
                    fx += ds1 * (FR.mx0 - FL.mx0);
                    fy += ds1 * (FR.mx1 - FL.mx1);
                    fz += ds1 * (FR.mx2 - FL.mx2);
                }

                {
                    auto const L = load(prim_arrays, i, j, k - 1);
                    auto const R = load(prim_arrays, i, j, k + 1);

                    auto const FL = riemann_solver(dir_t<2> {}, eos, L, c);
                    auto const FR = riemann_solver(dir_t<2> {}, eos, c, R);

                    fd += ds2 * (FR.d - FL.d);
                    fe += ds2 * (FR.e - FL.e);
                    fx += ds2 * (FR.mx0 - FL.mx0);
                    fy += ds2 * (FR.mx1 - FL.mx1);
                    fz += ds2 * (FR.mx2 - FL.mx2);
                }

                EulerCons<T> u = load(cons_arrays, i, j, k);

                u.d -= dtodv * fd;
                u.e -= dtodv * fe;
                u.mx0 -= dtodv * fx;
                u.mx1 -= dtodv * fy;
                u.mx2 -= dtodv * fz;

                store(u, cons_arrays, i, j, k);
            });
}
