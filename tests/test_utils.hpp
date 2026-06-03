#pragma once

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>

template <class T>
bool compare_prim(EulerPrimArrays<T> const& ref, EulerPrimArrays<T> const& vec, double tol, int idx)
{
    if (std::abs(ref.d(idx) - vec.d(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.p(idx) - vec.p(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.ux0(idx) - vec.ux0(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.ux1(idx) - vec.ux1(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.ux2(idx) - vec.ux2(idx)) > tol) {
        return false;
    }

    return true;
}

template <class T>
bool compare_cons(EulerConsArrays<T> const& ref, EulerConsArrays<T> const& vec, double tol, int idx)
{
    if (std::abs(ref.d(idx) - vec.d(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.e(idx) - vec.e(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.mx0(idx) - vec.mx0(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.mx1(idx) - vec.mx1(idx)) > tol) {
        return false;
    }
    if (std::abs(ref.mx2(idx) - vec.mx2(idx)) > tol) {
        return false;
    }

    return true;
}

template <class T>
EulerConsArrays<T> CopyToHost(EulerConsArrays<T> const& src)
{
    return EulerConsArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.d),
            .e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.e),
            .mx0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.mx0),
            .mx1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.mx1),
            .mx2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.mx2)};
}

template <class T>
EulerPrimArrays<T> CopyToHost(EulerPrimArrays<T> const& src)
{
    return EulerPrimArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.d),
            .p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.p),
            .ux0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.ux0),
            .ux1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.ux1),
            .ux2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src.ux2)};
}

template <class PrimArrays, class ConsArrays, class EOS>
void init_ramp_state(
        Kokkos::DefaultExecutionSpace const& exec,
        PrimArrays const& P,
        ConsArrays const& U,
        EOS const& eos)
{
    auto Pd = P.d;
    auto Pp = P.p;
    auto Pux0 = P.ux0;
    auto Pux1 = P.ux1;
    auto Pux2 = P.ux2;

    auto Ud = U.d;
    auto Ue = U.e;
    auto Umx0 = U.mx0;
    auto Umx1 = U.mx1;
    auto Umx2 = U.mx2;

    int const n0 = P.d.extent(0);
    int const n1 = P.d.extent(1);
    int const n2 = P.d.extent(2);

    Kokkos::parallel_for(
            "init_ramp_state",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {n0, n1, n2}),
            KOKKOS_LAMBDA(int i, int j, int k) {
                // deterministic smooth variation in x-direction only
                double const rho = 1.0 + (0.01 * i);
                double const p = 1.0 + (0.005 * i);

                double const ux0 = 0.5;
                double const ux1 = -0.2;
                double const ux2 = 0.1;

                Pd(i, j, k) = rho;
                Pp(i, j, k) = p;
                Pux0(i, j, k) = ux0;
                Pux1(i, j, k) = ux1;
                Pux2(i, j, k) = ux2;

                double const eint = eos.internal_energy(rho, p);

                EulerPrim<double> prim {.d = rho, .p = p, .ux0 = ux0, .ux1 = ux1, .ux2 = ux2};

                EulerCons const cons = to_cons(prim, eint);

                Ud(i, j, k) = cons.d;
                Ue(i, j, k) = cons.e;
                Umx0(i, j, k) = cons.mx0;
                Umx1(i, j, k) = cons.mx1;
                Umx2(i, j, k) = cons.mx2;
            });

    exec.fence();
}
