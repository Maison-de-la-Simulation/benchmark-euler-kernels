#include <cstddef>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>
#include <time_step.hpp>
#include <utils.hpp>

TEST(TimeStepRemainderWorstRem, ScalarVsVectorized)
{
    using real_t = double;
    using index_t = int;

    int const n = 23; // non-multiple of SIMD width to exercise remainder path

    auto nn = static_cast<std::size_t>(n);
    std::size_t const n3 = nn * nn * nn;
    Kokkos::DefaultExecutionSpace const exec_space;
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);

    auto prims_alloc = create_prim_arrays_1d<real_t>(exec_space, n3);
    auto prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);

    EulerPrim<real_t> const prim {.d = 1.0, .p = 1.0, .ux0 = 0.5, .ux1 = -0.3, .ux2 = 0.1};
    init_from_state(exec_space, prim_arrays, prim);
    exec_space.fence();

    real_t const dt_ref = time_step(exec_space, as_const(prim_arrays), eos, mesh);
    real_t const dt_vec = time_step_vec(exec_space, as_const(prim_arrays), eos, mesh);
    std::cout << "dt_ref = " << dt_ref << '\n';
    std::cout << "dt_vec = " << dt_vec << '\n';
    exec_space.fence();

    ASSERT_NEAR(dt_ref, dt_vec, 1e-12);
}

TEST(TimeStep, ScalarVsVectorized)
{
    using real_t = double;
    using index_t = int;

    int const n = 32;
    auto nn = static_cast<std::size_t>(n);
    std::size_t const n3 = nn * nn * nn;
    Kokkos::DefaultExecutionSpace const exec_space;
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);

    auto prims_alloc = create_prim_arrays_1d<real_t>(exec_space, n3);
    auto prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);

    EulerPrim<real_t> const prim {.d = 1.0, .p = 1.0, .ux0 = 0.5, .ux1 = -0.3, .ux2 = 0.1};
    init_from_state(exec_space, prim_arrays, prim);
    exec_space.fence();

    real_t const dt_ref = time_step(exec_space, as_const(prim_arrays), eos, mesh);
    real_t const dt_vec = time_step_vec(exec_space, as_const(prim_arrays), eos, mesh);
    std::cout << "dt_ref = " << dt_ref << '\n';
    std::cout << "dt_vec = " << dt_vec << '\n';

    exec_space.fence();

    ASSERT_NEAR(dt_ref, dt_vec, 1e-12);
}
