#include <cmath>
#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>
#include <prim_to_cons.hpp>
#include <utils.hpp>

#include "test_utils.hpp"

TEST(PrimToCons, ScalarVsVectorized)
{
    using real_t = double;
    using index_t = int;

    Kokkos::DefaultExecutionSpace const exec;
    PerfectGas<real_t> const eos(1.4);

    int const max_n = 33;
    for (int n = 3; n <= max_n; ++n) {
        std::size_t const n3 = std::size_t(n) * n * n;

        auto prims_alloc = create_prim_arrays_1d<real_t>(exec, n3);

        auto cons_ref_alloc = create_cons_arrays_1d<real_t>(exec, n3);
        auto cons_vec_alloc = create_cons_arrays_1d<real_t>(exec, n3);

        auto P = to_mdspan<Kokkos::mdspan<
                real_t,
                Kokkos::dextents<index_t, 3>,
                Kokkos::layout_left>>(prims_alloc, n, n, n);

        auto U_ref = to_mdspan<Kokkos::mdspan<
                real_t,
                Kokkos::dextents<index_t, 3>,
                Kokkos::layout_left>>(cons_ref_alloc, n, n, n);

        auto U_vec = to_mdspan<Kokkos::mdspan<
                real_t,
                Kokkos::dextents<index_t, 3>,
                Kokkos::layout_left>>(cons_vec_alloc, n, n, n);

        init_ramp_state(exec, P, U_ref, eos);

        Kokkos::deep_copy(cons_vec_alloc.d, cons_ref_alloc.d);
        Kokkos::deep_copy(cons_vec_alloc.e, cons_ref_alloc.e);
        Kokkos::deep_copy(cons_vec_alloc.mx0, cons_ref_alloc.mx0);
        Kokkos::deep_copy(cons_vec_alloc.mx1, cons_ref_alloc.mx1);
        Kokkos::deep_copy(cons_vec_alloc.mx2, cons_ref_alloc.mx2);

        exec.fence();

        prim_to_cons(exec, as_const(P), U_ref, eos);
        prim_to_cons_vec(exec, as_const(P), U_vec, eos);

        exec.fence();

        auto ref_h = CopyToHost(cons_ref_alloc);

        auto vec_h = CopyToHost(cons_vec_alloc);

        double const tol = 1e-12;
        for (int idx = 0; idx < n * n * n; ++idx) {
            ASSERT_TRUE(compare_cons(ref_h, vec_h, tol, idx)) << "Mismatch at:" << idx << "\n";
        }
    }
}
