#include <chrono>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>

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

#include "save_npy.hpp"

int main(int argc, char** argv)
{
    using index_t = int;
    using real_t = double;

    int const nx = 128;
    int const nt = 200;
    int const output_freq = 10;
    real_t const cfl_factor = 0.49;
    real_t const gamma = 1.4;

    real_t const dx = 1. / nx;
    Kokkos::ScopeGuard const scope(argc, argv);
    PerfectGas<real_t> const eos(gamma);
    UniformMesh3d<real_t> const mesh(dx, dx, dx);
    hllc const riemann_solver;
    Kokkos::DefaultExecutionSpace const exec_space;
    EulerPrimArrays const prims_alloc
            = create_prim_arrays_1d<real_t>(exec_space, (nx + 2) * (nx + 2) * (nx + 2));
    EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, nx + 2, nx + 2, nx + 2);
    EulerConsArrays const cons_alloc
            = create_cons_arrays_1d<real_t>(exec_space, (nx + 2) * (nx + 2) * (nx + 2));
    EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, nx + 2, nx + 2, nx + 2);

    init_implode(exec_space, prim_arrays, mesh);
    prim_to_cons(exec_space, as_const(prim_arrays), cons_arrays, eos);

    exec_space.fence();
    auto const start = std::chrono::steady_clock::now();
    int it = 0;
    while (it < nt) {
        real_t const dt = time_step(exec_space, as_const(prim_arrays), eos, mesh);

        godunov(exec_space,
                as_const(prim_arrays),
                cons_arrays,
                eos,
                mesh,
                riemann_solver,
                cfl_factor * dt);

        boundary_conditions_periodic(exec_space, cons_arrays, 1);

        cons_to_prim(exec_space, as_const(cons_arrays), prim_arrays, eos);

        ++it;

        if (output_freq > 0 && it % output_freq == 0) {
            int const padding = 10;
            std::stringstream ss;
            ss << "test_" << std::setfill('0') << std::setw(padding) << it << ".npy";
            std::fstream file(ss.str(), std::fstream::out);
            std::cout << "Saving " << ss.str() << ' ';
            save_npy(file, prim_arrays.p);
            std::cout << "done\n" << std::flush;
        }
    }
    exec_space.fence();
    auto const end = std::chrono::steady_clock::now();

    double const time_in_seconds = std::chrono::duration<double>(end - start).count();
    double const cells_updated = nt * static_cast<double>(nx * nx * nx);
    double const to_mega = 1E-6;
    std::cout << cells_updated / time_in_seconds * to_mega << '\n';
}
