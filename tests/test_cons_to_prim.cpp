#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cons_to_prim.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>
#include <utils.hpp>

#include "test_utils.hpp"

TEST(ConsToPrim, ScalarVsVectorized)
{
    using real_t = double;
    using index_t = int;
    Kokkos::DefaultExecutionSpace const exec_space;
    PerfectGas<real_t> const eos(1.4);

    int const max_n = 33;
    for (int n = 3; n <= max_n; ++n) {
        auto nn = static_cast<std::size_t>(n);
        std::size_t const n3 = nn * nn * nn;

        auto cons_alloc = create_cons_arrays_1d<real_t>(exec_space, n3);
        auto prims_alloc_ref = create_prim_arrays_1d<real_t>(exec_space, n3);
        auto prims_alloc_vec = create_prim_arrays_1d<real_t>(exec_space, n3);

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

        EulerCons<real_t> const cons {.d = 1.0, .e = 2.5, .mx0 = 0.5, .mx1 = -0.3, .mx2 = 0.1};
        init_from_state(exec_space, cons_arrays, cons);
        exec_space.fence();

        cons_to_prim(exec_space, as_const(cons_arrays), prim_ref, eos);
        cons_to_prim_vec(exec_space, as_const(cons_arrays), prim_vec, eos);
        exec_space.fence();

        auto ref_h = CopyToHost(prims_alloc_ref);

        auto vec_h = CopyToHost(prims_alloc_vec);

        double const tol = 1e-12;

        for (int idx = 0; idx < n * n * n; ++idx) {
            ASSERT_TRUE(compare_prim(ref_h, vec_h, tol, idx)) << "Mismatch at:" << idx << "\n";
        }
    }
}
