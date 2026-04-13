#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <godunov.hpp>
#include <hllc.hpp>
#include <perfect_gas.hpp>
#include <uniform_mesh.hpp>

#include "utils.hpp"

namespace {

struct Results
{
    std::vector<double> d, e, mx0, mx1, mx2;
};

template <class Mdspan>
Results to_host(EulerConsArrays<Mdspan> const& a)
{
    std::size_t const n = a.d.mapping().required_span_size();

    auto copy = [&](auto* ptr) {
        Kokkos::View<
                double*,
                Kokkos::DefaultExecutionSpace,
                Kokkos::MemoryTraits<Kokkos::Unmanaged>>
                v(ptr, n);
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace {}, v);
        return std::vector<double>(h.data(), h.data() + n);
    };

    return {copy(a.d.data_handle()),
            copy(a.e.data_handle()),
            copy(a.mx0.data_handle()),
            copy(a.mx1.data_handle()),
            copy(a.mx2.data_handle())};
}

template <class Kernel>
Results run(
        Kokkos::DefaultExecutionSpace& exec,
        int n,
        EulerPrim<double> const& prim,
        PerfectGas<double> const& eos,
        UniformMesh3d<double> const& mesh,
        double dt,
        Kernel kernel)
{
    auto prims = create_prim_arrays_1d<double>(exec, n * n * n);
    auto cons = create_cons_arrays_1d<double>(exec, n * n * n);

    auto P = to_mdspan<
            Kokkos::mdspan<double, Kokkos::dextents<int, 3>, Kokkos::layout_left>>(prims, n, n, n);
    auto U = to_mdspan<
            Kokkos::mdspan<double, Kokkos::dextents<int, 3>, Kokkos::layout_left>>(cons, n, n, n);

    init_from_state(exec, P, prim);
    init_from_state(exec, U, to_cons(prim, eos.internal_energy(prim.d, prim.p)));

    exec.fence();
    kernel(exec, as_const(P), U, eos, mesh, hllc {}, dt);
    exec.fence();

    return to_host(U);
}

void assert_near(Results const& a, Results const& b, double tol)
{
    ASSERT_EQ(a.d.size(), b.d.size());
    for (std::size_t i = 0; i < a.d.size(); ++i) {
        EXPECT_NEAR(a.d[i], b.d[i], tol);
        EXPECT_NEAR(a.e[i], b.e[i], tol);
        EXPECT_NEAR(a.mx0[i], b.mx0[i], tol);
        EXPECT_NEAR(a.mx1[i], b.mx1[i], tol);
        EXPECT_NEAR(a.mx2[i], b.mx2[i], tol);
    }
}

void run_case(int n, double dt, EulerPrim<double> const& prim)
{
    Kokkos::DefaultExecutionSpace exec;
    PerfectGas<double> eos(1.4);
    UniformMesh3d<double> mesh(1., 1., 1.);

    // auto ref = run(exec, n, prim, eos, mesh, dt, godunov);
    Results ref
            = run(exec,
                  n,
                  prim,
                  eos,
                  mesh,
                  dt,
                  [](auto& exec,
                     auto const& P,
                     auto& U,
                     auto const& eos,
                     auto const& mesh,
                     auto solver,
                     double dt) { godunov(exec, P, U, eos, mesh, solver, dt); });

    Results vec
            = run(exec,
                  n,
                  prim,
                  eos,
                  mesh,
                  dt,
                  [](auto& exec,
                     auto const& P,
                     auto& U,
                     auto const& eos,
                     auto const& mesh,
                     auto solver,
                     double dt) { godunov_vec(exec, P, U, eos, mesh, solver, dt); });

    assert_near(ref, vec, 1e-12);
}

} // namespace


struct Case
{
    int n;
    double dt;
    EulerPrim<double> prim;
};

class GodunovTest : public ::testing::TestWithParam<Case>
{
};

TEST_P(GodunovTest, ScalarVsVectorized)
{
    auto const& c = GetParam();
    run_case(c.n, c.dt, c.prim);
}

INSTANTIATE_TEST_SUITE_P(
        All,
        GodunovTest,
        ::testing::Values(
                Case {23,
                      1e-9,
                      {.d = 1.0,
                       .p = 1.0,
                       .ux0 = 0.5,
                       .ux1 = -0.3,
                       .ux2 = 0.1}}, // remainder stress
                Case {32,
                      1e-9,
                      {.d = 1.0, .p = 1.0, .ux0 = 0.5, .ux1 = -0.3, .ux2 = 0.1}}, // clean SIMD
                Case {33, 1e-10, {.d = 4.0, .p = 10.0, .ux0 = 1.5, .ux1 = -0.8, .ux2 = 0.3}}
                // shock-like
                ));
