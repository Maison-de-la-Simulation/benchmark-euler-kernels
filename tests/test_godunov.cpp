#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <godunov.hpp>
#include <hllc.hpp>
#include <perfect_gas.hpp>
#include <uniform_mesh.hpp>
#include <utils.hpp>

#include "test_utils.hpp"

namespace {

using real_t = double;
using index_t = int;

struct Case
{
    int n;
    double dt;
    EulerPrim<double> prim;
};

class GodunovTest : public ::testing::TestWithParam<Case>
{
};

constexpr EulerPrim<double> uniform_state {.d = 1.0, .p = 1.0, .ux0 = 0.5, .ux1 = -0.3, .ux2 = 0.1};

constexpr EulerPrim<double> shock_state {.d = 4.0, .p = 10.0, .ux0 = 1.5, .ux1 = -0.8, .ux2 = 0.3};

constexpr double dt_default = 1e-9;
constexpr double dt_small = 1e-10;

template <class Kernel, class Solver>
auto run(
        Kokkos::DefaultExecutionSpace const& exec,
        int n,
        EulerPrim<double> const& prim,
        PerfectGas<double> const& eos,
        UniformMesh3d<double> const& mesh,
        double dt,
        Kernel kernel,
        Solver solver)
{
    auto const nn = static_cast<std::size_t>(n);
    std::size_t const n3 = nn * nn * nn;

    auto prims_alloc = create_prim_arrays_1d<real_t>(exec, n3);
    auto cons_alloc = create_cons_arrays_1d<real_t>(exec, n3);

    auto P = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);

    auto U = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);

    init_from_state(exec, P, prim);
    init_from_state(exec, U, to_cons(prim, eos.internal_energy(prim.d, prim.p)));

    exec.fence();
    kernel(exec, as_const(P), U, eos, mesh, solver, dt);
    exec.fence();

    return cons_alloc;
}

void run_case(int n, double dt, EulerPrim<double> const& prim)
{
    Kokkos::DefaultExecutionSpace const exec;
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);

    auto cons_ref = run(
            exec,
            n,
            prim,
            eos,
            mesh,
            dt,
            [](auto const& exec,
               auto const& P,
               auto& U,
               auto const& eos,
               auto const& mesh,
               auto solver,
               double dt) { godunov(exec, P, U, eos, mesh, solver, dt); },
            hllc {});

    auto cons_vec = run(
            exec,
            n,
            prim,
            eos,
            mesh,
            dt,
            [](auto const& exec,
               auto const& P,
               auto& U,
               auto const& eos,
               auto const& mesh,
               auto solver,
               double dt) { godunov_vec(exec, P, U, eos, mesh, solver, dt); },
            hllc {});

    auto ref_h = EulerConsArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_ref.d),
            .e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_ref.e),
            .mx0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_ref.mx0),
            .mx1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_ref.mx1),
            .mx2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_ref.mx2)};

    auto vec_h = EulerConsArrays {
            .d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_vec.d),
            .e = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_vec.e),
            .mx0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_vec.mx0),
            .mx1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_vec.mx1),
            .mx2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cons_vec.mx2)};

    double const tol = 1e-12;

    for (int idx = 0; idx < n * n * n; ++idx) {
        compare(ref_h, vec_h, tol, idx);
    }
}

} // namespace

TEST_P(GodunovTest, ScalarVectorizedAgree)
{
    auto const& c = GetParam();
    run_case(c.n, c.dt, c.prim);
}

INSTANTIATE_TEST_SUITE_P(
        All,
        GodunovTest,
        ::testing::Values(
                Case {23, dt_default, uniform_state},
                Case {32, dt_default, uniform_state},
                Case {33, dt_small, shock_state}));
