#include <cstddef>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>
#include <prim_to_cons.hpp>
#include <utils.hpp>

#include "test_utils.hpp"

TEST(PrimToConsRemainder, ScalarVsVectorized)
{
    using real_t = double;
    using index_t = int;

    int const n = 23;

    Kokkos::DefaultExecutionSpace const exec_space;
    PerfectGas<real_t> const eos(1.4);

    auto nn = static_cast<std::size_t>(n);
    std::size_t const n3 = nn * nn * nn;

    auto prims_alloc = create_prim_arrays_1d<real_t>(exec_space, n3);
    // --- allocate base ---
    auto cons_alloc_ref = create_cons_arrays_1d<real_t>(exec_space, n3);

    // --- allocate vectorized ---
    auto cons_alloc_vec = create_cons_arrays_1d<real_t>(exec_space, n3);

    auto prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);

    auto cons_ref = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc_ref, n, n, n);

    auto cons_vec = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc_vec, n, n, n);

    // --- initialize with non-trivial state ---
    EulerPrim<real_t> const prim {.d = 1.0, .p = 1.0, .ux0 = 0.5, .ux1 = -0.3, .ux2 = 0.1};

    init_from_state(exec_space, prim_arrays, prim);
    exec_space.fence();

    // --- run both ---
    prim_to_cons(exec_space, as_const(prim_arrays), cons_ref, eos);
    prim_to_cons_vec(exec_space, as_const(prim_arrays), cons_vec, eos);
    exec_space.fence();

    auto ref_h = EulerConsArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.d),
            .e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.e),
            .mx0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.mx0),
            .mx1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.mx1),
            .mx2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_ref.mx2)};

    auto vec_h = EulerConsArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.d),
            .e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.e),
            .mx0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.mx0),
            .mx1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.mx1),
            .mx2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_alloc_vec.mx2)};

    double const tol = 1e-12;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                int const idx = i + (n * (j + (n * k))); // layout_left flattening
                compare(ref_h, vec_h, tol, idx);
            }
        }
    }
}
