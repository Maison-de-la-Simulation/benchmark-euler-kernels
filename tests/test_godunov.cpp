#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <godunov.hpp>
#include <hllc.hpp>
#include <perfect_gas.hpp>
#include <uniform_mesh.hpp>

#include "prim_to_cons.hpp"
#include "test_utils.hpp"

namespace {

using real_t = double;
using index_t = int;

template <class Kernel>
auto run_case_impl(
        Kokkos::DefaultExecutionSpace const& exec,
        int n,
        PerfectGas<real_t> const& eos,
        UniformMesh3d<real_t> const& mesh,
        double dt,
        Kernel kernel)
{
    auto nn = static_cast<std::size_t>(n);
    std::size_t const n3 = nn * nn * nn;

    auto prims_alloc = create_prim_arrays_1d<real_t>(exec, n3);
    auto cons_alloc = create_cons_arrays_1d<real_t>(exec, n3);

    auto prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);

    auto cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);

    init_ramp_state(exec, prim_arrays, mesh);
    prim_to_cons(exec, as_const(prim_arrays), cons_arrays, eos);

    kernel(exec, as_const(prim_arrays), cons_arrays, eos, mesh, hllc {}, dt);

    exec.fence();

    return cons_alloc;
}

} // namespace

TEST(Godunov, ScalarVsVectorized)
{
    Kokkos::DefaultExecutionSpace const exec;

    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);

    double const dt = 1e-5;
    int const max_n = 33;
    for (int n = 3; n <= max_n; ++n) {
        auto ref = run_case_impl(
                exec,
                n,
                eos,
                mesh,
                dt,
                [](auto const& exec,
                   auto const& P,
                   auto& U,
                   auto const& eos,
                   auto const& mesh,
                   auto solver,
                   double dt) { godunov(exec, P, U, eos, mesh, solver, dt); });

        auto vec = run_case_impl(
                exec,
                n,
                eos,
                mesh,
                dt,
                [](auto const& exec,
                   auto const& P,
                   auto& U,
                   auto const& eos,
                   auto const& mesh,
                   auto solver,
                   double dt) { godunov_vec(exec, P, U, eos, mesh, solver, dt); });

        auto ref_h = copy_to_host(ref);

        auto vec_h = copy_to_host(vec);

        double const tol = 1e-12;

        for (int idx = 0; idx < n * n * n; ++idx) {
            EXPECT_TRUE(compare(ref_h, vec_h, tol, idx)) << "Mismatch at idx = " << idx;
        }
    }
}
