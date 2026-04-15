#include <cmath>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <cons_to_prim.hpp>
#include <euler_arrays.hpp>
#include <godunov.hpp>
#include <hllc.hpp>
#include <init_implode.hpp>
#include <perfect_gas.hpp>
#include <periodic_boundary_conditions.hpp>
#include <prim_to_cons.hpp>
#include <time_step.hpp>
#include <uniform_mesh.hpp>

#include "compute_l1_error.hpp"
#include "cosine_advection_solution.hpp"

namespace {

double compute_error(int const nx, int const ny, int const nz, int const dir)
{
    using index_t = int;
    using real_t = double;

    int const nt = 1'000'000;
    real_t const tend = 1.;
    real_t const cfl_factor = 0.9;
    real_t const gamma = 1.4;

    real_t const dx = 1. / nx;
    real_t const dy = 1. / ny;
    real_t const dz = 1. / nz;
    PerfectGas<real_t> const eos(gamma);
    UniformMesh3d<real_t> const mesh(dx, dy, dz);
    hllc const riemann_solver;
    Kokkos::DefaultExecutionSpace const exec_space;
    EulerPrimArrays const prims_alloc
            = create_prim_arrays_1d<real_t>(exec_space, (nx + 2) * (ny + 2) * (nz + 2));
    EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, nx + 2, ny + 2, nz + 2);
    EulerConsArrays const cons_alloc
            = create_cons_arrays_1d<real_t>(exec_space, (nx + 2) * (ny + 2) * (nz + 2));
    EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, nx + 2, ny + 2, nz + 2);

    int it = 0;
    real_t t = 0;

    cosine_advection_solution(exec_space, prim_arrays, mesh, t, dir);
    prim_to_cons(exec_space, as_const(prim_arrays), cons_arrays, eos);

    exec_space.fence();
    bool exit = false;
    while (!exit) {
        real_t dt = time_step(exec_space, as_const(prim_arrays), eos, mesh);
        dt *= cfl_factor;
        if (t + dt >= tend) {
            dt = tend - t;
            exit = true;
        }
        if (it + 1 >= nt) {
            exit = true;
        }

        godunov(exec_space, as_const(prim_arrays), cons_arrays, eos, mesh, riemann_solver, dt);

        boundary_conditions_periodic(exec_space, cons_arrays, 1);

        cons_to_prim(exec_space, as_const(cons_arrays), prim_arrays, eos);

        t += dt;
        ++it;
    }

    EulerPrimArrays const sol_alloc
            = create_prim_arrays_1d<real_t>(exec_space, (nx + 2) * (ny + 2) * (nz + 2));
    EulerPrimArrays const sol_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(sol_alloc, nx + 2, ny + 2, nz + 2);
    cosine_advection_solution(exec_space, sol_arrays, mesh, t, dir);
    return compute_l1_error(exec_space, as_const(prim_arrays).d, as_const(sol_arrays).d);
}

} // namespace

TEST(Euler, Convergence2dX)
{
    std::vector<int> const sizes {32, 64, 128, 256, 512, 1024};
    std::vector<double> errors(sizes.size());
    for (std::size_t i = 0; i < errors.size(); ++i) {
        errors[i] = compute_error(sizes[i], sizes[i], 1, 0);
    }
    std::vector<double> orders(sizes.size() - 1);
    for (std::size_t i = 0; i < orders.size(); ++i) {
        orders[i] = -std::log10(errors[i + 1] / errors[i]) / std::log10(sizes[i + 1] / sizes[i]);
    }

    double const expected_order = 1;
    EXPECT_GT(orders.back(), expected_order * 0.95);
    for (std::size_t i = 0; i < orders.size() - 1; ++i) {
        EXPECT_LT(orders[i], orders[i + 1]) << i;
    }
}

TEST(Euler, Convergence2dY)
{
    std::vector<int> const sizes {32, 64, 128, 256, 512, 1024};
    std::vector<double> errors(sizes.size());
    for (std::size_t i = 0; i < errors.size(); ++i) {
        errors[i] = compute_error(1, sizes[i], sizes[i], 1);
    }
    std::vector<double> orders(sizes.size() - 1);
    for (std::size_t i = 0; i < orders.size(); ++i) {
        orders[i] = -std::log10(errors[i + 1] / errors[i]) / std::log10(sizes[i + 1] / sizes[i]);
    }

    double const expected_order = 1;
    EXPECT_GT(orders.back(), expected_order * 0.95);
    for (std::size_t i = 0; i < orders.size() - 1; ++i) {
        EXPECT_LT(orders[i], orders[i + 1]) << i;
    }
}

TEST(Euler, Convergence2dZ)
{
    std::vector<int> const sizes {32, 64, 128, 256, 512, 1024};
    std::vector<double> errors(sizes.size());
    for (std::size_t i = 0; i < errors.size(); ++i) {
        errors[i] = compute_error(sizes[i], 1, sizes[i], 2);
    }
    std::vector<double> orders(sizes.size() - 1);
    for (std::size_t i = 0; i < orders.size(); ++i) {
        orders[i] = -std::log10(errors[i + 1] / errors[i]) / std::log10(sizes[i + 1] / sizes[i]);
    }

    double const expected_order = 1;
    EXPECT_GT(orders.back(), expected_order * 0.95);
    for (std::size_t i = 0; i < orders.size() - 1; ++i) {
        EXPECT_LT(orders[i], orders[i + 1]) << i;
    }
}
