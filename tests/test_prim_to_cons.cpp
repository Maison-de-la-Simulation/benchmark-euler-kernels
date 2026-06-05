#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>
#include <prim_to_cons.hpp>

#include "test_utils.hpp"

TEST(PrimToCons, ScalarVsVectorized)
{
    using real_t = double;
    using index_t = int;

    Kokkos::DefaultExecutionSpace const exec_space;
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);

    int const max_n = 33;
    for (int n = 3; n <= max_n; ++n) {
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

        init_ramp_state(exec_space, prim_arrays, mesh);
        exec_space.fence();

        // --- run both ---
        prim_to_cons(exec_space, as_const(prim_arrays), cons_ref, eos);
        prim_to_cons_vec(exec_space, as_const(prim_arrays), cons_vec, eos);
        exec_space.fence();

        auto ref_h = copy_to_host(cons_alloc_ref);

        auto vec_h = copy_to_host(cons_alloc_vec);

        double const tol = 1e-12;
        for (int idx = 0; idx < n * n * n; ++idx) {
            EXPECT_TRUE(compare(ref_h, vec_h, tol, idx)) << "Mismatch at:" << idx << "\n";
        }
    }
}
