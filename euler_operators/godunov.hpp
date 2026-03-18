#pragma once

#include <cstddef>
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
