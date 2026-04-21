#include <cstddef>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cons_to_prim.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>

#include "utils.hpp"

TEST(ConsToPrimRemainder, ScalarVsVectorized)
{
    using real_t = double;
    using index_t = int;
    int const n = 23;
    Kokkos::DefaultExecutionSpace const exec_space;
    PerfectGas<real_t> const eos(1.4);

    auto cons_alloc
            = create_cons_arrays_1d<real_t>(exec_space, static_cast<std::size_t>(n * n * n));
    // --- allocate base ---
    auto prims_alloc_ref
            = create_prim_arrays_1d<real_t>(exec_space, static_cast<std::size_t>(n * n * n));
    // --- allocate vectorized ---
    auto prims_alloc_vec
            = create_prim_arrays_1d<real_t>(exec_space, static_cast<std::size_t>(n * n * n));

    auto cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);
    auto prim_ref = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc_ref, n, n, n);
    auto prim_vec = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc_vec, n, n, n);

    // --- initialize with non-trivial conserved state ---
    EulerCons<real_t> const cons {.d = 1.0, .e = 2.5, .mx0 = 0.5, .mx1 = -0.3, .mx2 = 0.1};
    init_from_state(exec_space, cons_arrays, cons);
    exec_space.fence();

    // --- run both ---
    cons_to_prim(exec_space, as_const(cons_arrays), prim_ref, eos);
    cons_to_prim_vec(exec_space, as_const(cons_arrays), prim_vec, eos);
    exec_space.fence();

    auto ref_h = EulerPrimArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_ref.d),
            .p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_ref.p),
            .ux0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_ref.ux0),
            .ux1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_ref.ux1),
            .ux2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_ref.ux2)};
    auto vec_h = EulerPrimArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_vec.d),
            .p = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_vec.p),
            .ux0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_vec.ux0),
            .ux1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_vec.ux1),
            .ux2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), prims_alloc_vec.ux2)};

    double const tol = 1e-12;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                int const idx = i + (n * (j + (n * k))); // layout_left flattening
                ASSERT_NEAR(ref_h.d(idx), vec_h.d(idx), tol);
                ASSERT_NEAR(ref_h.p(idx), vec_h.p(idx), tol);
                ASSERT_NEAR(ref_h.ux0(idx), vec_h.ux0(idx), tol);
                ASSERT_NEAR(ref_h.ux1(idx), vec_h.ux1(idx), tol);
                ASSERT_NEAR(ref_h.ux2(idx), vec_h.ux2(idx), tol);
            }
        }
    }
}
