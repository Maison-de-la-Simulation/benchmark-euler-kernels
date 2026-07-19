#pragma once

#include <cstddef>
#include <span>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <Kokkos_SIMD.hpp>
#include <Kokkos_SIMD_Extended.hpp>
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

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void godunov_vec2(
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

    using scalar_simd_t = KE::basic_simd<T, KE::simd_abi::scalar>;

    using simd_type = KE::simd<double>;

    using simd_t = KE::basic_simd<
            double,
            KE::simd_abi::extended_abi<simd_type::abi_type, simd_type::size() * 2>>;

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

template <class T, class IndexType, std::size_t E0, std::size_t E1, std::size_t E2>
void godunov_vec_unrolled(
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
    using scalar_simd_t = KE::basic_simd<T, KE::simd_abi::scalar>;

    constexpr IndexType simd_width = static_cast<IndexType>(simd_t::size());
    constexpr IndexType unroll_factor = 2;
    constexpr IndexType cells_per_iteration = unroll_factor * simd_width;

    IndexType const n0 = static_cast<IndexType>(prim_arrays.d.extent(0));
    IndexType const n0_begin = 1;
    IndexType const n0_end = n0 - 1;
    IndexType const n0_inner = n0 - 2;

    IndexType const unrolled_cells = (n0_inner / cells_per_iteration) * cells_per_iteration;

    IndexType const unrolled_end = n0_begin + unrolled_cells;

    Kokkos::full_extent_t const slice1;
    Kokkos::full_extent_t const slice2;

    if (unrolled_cells > 0) {
        Kokkos::pair const slice0(IndexType {0}, unrolled_end + IndexType {1});

        EulerPrimArrays const sub_prim_arrays = subspan(prim_arrays, slice0, slice1, slice2);

        EulerConsArrays const sub_cons_arrays = subspan(cons_arrays, slice0, slice1, slice2);

        EulerPrimArrays<T const*> const prim_ptrs {
                sub_prim_arrays.d.data_handle(),
                sub_prim_arrays.p.data_handle(),
                sub_prim_arrays.ux0.data_handle(),
                sub_prim_arrays.ux1.data_handle(),
                sub_prim_arrays.ux2.data_handle()};

        EulerConsArrays<T*> const cons_ptrs {
                sub_cons_arrays.d.data_handle(),
                sub_cons_arrays.e.data_handle(),
                sub_cons_arrays.mx0.data_handle(),
                sub_cons_arrays.mx1.data_handle(),
                sub_cons_arrays.mx2.data_handle()};

        IndexType const extent1 = static_cast<IndexType>(sub_prim_arrays.d.extent(1));
        IndexType const extent2 = static_cast<IndexType>(sub_prim_arrays.d.extent(2));

        std::size_t const stride0 = sub_prim_arrays.d.stride(0);
        std::size_t const stride1 = sub_prim_arrays.d.stride(1);
        std::size_t const stride2 = sub_prim_arrays.d.stride(2);

        IndexType const number_of_unrolled_blocks = unrolled_cells / cells_per_iteration;

        simd_t const dt_dx(dt / mesh.dx0());
        simd_t const dt_dy(dt / mesh.dx1());
        simd_t const dt_dz(dt / mesh.dx2());

        using policy_type = Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::Rank<3>>;

        policy_type const
                policy(exec_space,
                       {IndexType {0}, IndexType {1}, IndexType {1}},
                       {number_of_unrolled_blocks,
                        extent1 - IndexType {1},
                        extent2 - IndexType {1}});

        Kokkos::parallel_for(
                "godunov_vec_unrolled",
                policy,
                KOKKOS_LAMBDA(IndexType const block, IndexType const j, IndexType const k) {
                    std::size_t const i0
                            = static_cast<std::size_t>(IndexType {1} + block * cells_per_iteration);

                    std::size_t const base0 = i0 * stride0 + static_cast<std::size_t>(j) * stride1
                                              + static_cast<std::size_t>(k) * stride2;

                    std::size_t const base1
                            = base0 + static_cast<std::size_t>(simd_width) * stride0;

                    {
                        EulerPrim<simd_t> const prim = load<simd_t>(prim_ptrs, base0);

                        EulerPrim<simd_t> const prim_xm = load<simd_t>(prim_ptrs, base0 - stride0);

                        EulerPrim<simd_t> const prim_xp = load<simd_t>(prim_ptrs, base0 + stride0);

                        EulerPrim<simd_t> const prim_ym = load<simd_t>(prim_ptrs, base0 - stride1);

                        EulerPrim<simd_t> const prim_yp = load<simd_t>(prim_ptrs, base0 + stride1);

                        EulerPrim<simd_t> const prim_zm = load<simd_t>(prim_ptrs, base0 - stride2);

                        EulerPrim<simd_t> const prim_zp = load<simd_t>(prim_ptrs, base0 + stride2);

                        EulerFlux<simd_t> const flux_xm
                                = riemann_solver(dir_t<0>(), eos, prim_xm, prim);

                        EulerFlux<simd_t> const flux_xp
                                = riemann_solver(dir_t<0>(), eos, prim, prim_xp);

                        EulerFlux<simd_t> const flux_ym
                                = riemann_solver(dir_t<1>(), eos, prim_ym, prim);

                        EulerFlux<simd_t> const flux_yp
                                = riemann_solver(dir_t<1>(), eos, prim, prim_yp);

                        EulerFlux<simd_t> const flux_zm
                                = riemann_solver(dir_t<2>(), eos, prim_zm, prim);

                        EulerFlux<simd_t> const flux_zp
                                = riemann_solver(dir_t<2>(), eos, prim, prim_zp);

                        EulerCons<simd_t> cons = load<simd_t>(cons_ptrs, base0);

                        cons.d -= dt_dx * (flux_xp.d - flux_xm.d) + dt_dy * (flux_yp.d - flux_ym.d)
                                  + dt_dz * (flux_zp.d - flux_zm.d);

                        cons.e -= dt_dx * (flux_xp.e - flux_xm.e) + dt_dy * (flux_yp.e - flux_ym.e)
                                  + dt_dz * (flux_zp.e - flux_zm.e);

                        cons.mx0 -= dt_dx * (flux_xp.mx0 - flux_xm.mx0)
                                    + dt_dy * (flux_yp.mx0 - flux_ym.mx0)
                                    + dt_dz * (flux_zp.mx0 - flux_zm.mx0);

                        cons.mx1 -= dt_dx * (flux_xp.mx1 - flux_xm.mx1)
                                    + dt_dy * (flux_yp.mx1 - flux_ym.mx1)
                                    + dt_dz * (flux_zp.mx1 - flux_zm.mx1);

                        cons.mx2 -= dt_dx * (flux_xp.mx2 - flux_xm.mx2)
                                    + dt_dy * (flux_yp.mx2 - flux_ym.mx2)
                                    + dt_dz * (flux_zp.mx2 - flux_zm.mx2);

                        store<simd_t>(cons, cons_ptrs, base0);
                    }

                    {
                        EulerPrim<simd_t> const prim = load<simd_t>(prim_ptrs, base1);

                        EulerPrim<simd_t> const prim_xm = load<simd_t>(prim_ptrs, base1 - stride0);

                        EulerPrim<simd_t> const prim_xp = load<simd_t>(prim_ptrs, base1 + stride0);

                        EulerPrim<simd_t> const prim_ym = load<simd_t>(prim_ptrs, base1 - stride1);

                        EulerPrim<simd_t> const prim_yp = load<simd_t>(prim_ptrs, base1 + stride1);

                        EulerPrim<simd_t> const prim_zm = load<simd_t>(prim_ptrs, base1 - stride2);

                        EulerPrim<simd_t> const prim_zp = load<simd_t>(prim_ptrs, base1 + stride2);

                        EulerFlux<simd_t> const flux_xm
                                = riemann_solver(dir_t<0>(), eos, prim_xm, prim);

                        EulerFlux<simd_t> const flux_xp
                                = riemann_solver(dir_t<0>(), eos, prim, prim_xp);

                        EulerFlux<simd_t> const flux_ym
                                = riemann_solver(dir_t<1>(), eos, prim_ym, prim);

                        EulerFlux<simd_t> const flux_yp
                                = riemann_solver(dir_t<1>(), eos, prim, prim_yp);

                        EulerFlux<simd_t> const flux_zm
                                = riemann_solver(dir_t<2>(), eos, prim_zm, prim);

                        EulerFlux<simd_t> const flux_zp
                                = riemann_solver(dir_t<2>(), eos, prim, prim_zp);

                        EulerCons<simd_t> cons = load<simd_t>(cons_ptrs, base1);

                        cons.d -= dt_dx * (flux_xp.d - flux_xm.d) + dt_dy * (flux_yp.d - flux_ym.d)
                                  + dt_dz * (flux_zp.d - flux_zm.d);

                        cons.e -= dt_dx * (flux_xp.e - flux_xm.e) + dt_dy * (flux_yp.e - flux_ym.e)
                                  + dt_dz * (flux_zp.e - flux_zm.e);

                        cons.mx0 -= dt_dx * (flux_xp.mx0 - flux_xm.mx0)
                                    + dt_dy * (flux_yp.mx0 - flux_ym.mx0)
                                    + dt_dz * (flux_zp.mx0 - flux_zm.mx0);

                        cons.mx1 -= dt_dx * (flux_xp.mx1 - flux_xm.mx1)
                                    + dt_dy * (flux_yp.mx1 - flux_ym.mx1)
                                    + dt_dz * (flux_zp.mx1 - flux_zm.mx1);

                        cons.mx2 -= dt_dx * (flux_xp.mx2 - flux_xm.mx2)
                                    + dt_dy * (flux_yp.mx2 - flux_ym.mx2)
                                    + dt_dz * (flux_zp.mx2 - flux_zm.mx2);

                        store<simd_t>(cons, cons_ptrs, base1);
                    }
                });
    }

    if (unrolled_end < n0_end) {
        Kokkos::pair const slice0(unrolled_end - IndexType {1}, n0);

        EulerPrimArrays const sub_prim_arrays = subspan(prim_arrays, slice0, slice1, slice2);

        EulerConsArrays const sub_cons_arrays = subspan(cons_arrays, slice0, slice1, slice2);

        godunov_kernel<scalar_simd_t>(
                exec_space,
                sub_prim_arrays,
                sub_cons_arrays,
                eos,
                mesh,
                riemann_solver,
                dt);
    }
}
