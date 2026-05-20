#include <chrono>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <ostream> // IWYU pragma: keep (std::flush)
#include <string>

#include <Kokkos_Core.hpp>
#include <cons_to_prim.hpp>
#include <euler_arrays.hpp>
#include <godunov.hpp>
#include <hllc.hpp>
#include <init_implode.hpp>
#include <perfect_gas.hpp>
#include <periodic_boundary_conditions.hpp>
#include <prim_to_cons.hpp>
#include <sched.h>
#include <time_step.hpp>
#include <uniform_mesh.hpp>
#include <unistd.h>

using std::getenv;

namespace {
inline void print_affinity()
{
    cpu_set_t set;
    CPU_ZERO(&set);

    if (sched_getaffinity(0, sizeof(cpu_set_t), &set) == 0) {
        std::cout << "CPU affinity: ";

        bool first = true;
        for (int i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &set)) {
                if (!first) {
                    std::cout << ",";
                }
                std::cout << i;
                first = false;
            }
        }
        std::cout << "\n";
    } else {
        std::cout << "Failed to get CPU affinity\n";
    }
}
} // namespace

int main(int argc, char** argv)
{
    using index_t = int;
    using real_t = double;

    std::string mode = "strong"; // strong | weak
    std::string out_file = "results.csv";
    std::string kernel = "scalar"; // scalar | vector

    int threads = 1;

    if (char const* env_p = getenv("OMP_NUM_THREADS")) {
        threads = std::stoi(env_p);
    }

    int const nx_init = 256;
    int const nt_init = 200;

    int nx = nx_init;
    int nt = nt_init;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--mode") {
            mode = argv[++i];
        } else if (arg == "--nx") {
            nx = std::stoi(argv[++i]);
        } else if (arg == "--nt") {
            nt = std::stoi(argv[++i]);
        } else if (arg == "--out") {
            out_file = argv[++i];
        } else if (arg == "--kernel") {
            kernel = argv[++i];
        }
    }

    Kokkos::ScopeGuard const scope(argc, argv);
    Kokkos::DefaultExecutionSpace const exec_space;

    print_affinity();

    real_t const cfl_factor = 0.49;
    real_t const gamma = 1.4;

    auto run_scalar = [&](int nx_local) {
        std::size_t const nxg_z = nx_local + 2;
        index_t const nxg = nx_local + 2;

        real_t const dx = 1.0 / nx_local;

        PerfectGas<real_t> eos(gamma);
        UniformMesh3d<real_t> mesh(dx, dx, dx);
        hllc riemann_solver;

        auto prim_alloc = create_prim_arrays_1d<real_t>(exec_space, nxg_z * nxg_z * nxg_z);

        auto cons_alloc = create_cons_arrays_1d<real_t>(exec_space, nxg_z * nxg_z * nxg_z);

        EulerPrimArrays prim_arrays = to_mdspan<Kokkos::mdspan<
                real_t,
                Kokkos::dextents<index_t, 3>,
                Kokkos::layout_left>>(prim_alloc, nxg, nxg, nxg);

        EulerConsArrays cons_arrays = to_mdspan<Kokkos::mdspan<
                real_t,
                Kokkos::dextents<index_t, 3>,
                Kokkos::layout_left>>(cons_alloc, nxg, nxg, nxg);

        init_implode(exec_space, prim_arrays, mesh);
        prim_to_cons(exec_space, as_const(prim_arrays), cons_arrays, eos);

        exec_space.fence();

        auto const start = std::chrono::steady_clock::now();

        for (int it = 0; it < nt; ++it) {
            real_t dt = time_step(exec_space, as_const(prim_arrays), eos, mesh);

            godunov(exec_space,
                    as_const(prim_arrays),
                    cons_arrays,
                    eos,
                    mesh,
                    riemann_solver,
                    cfl_factor * dt);

            boundary_conditions_periodic(exec_space, cons_arrays, 1);
            cons_to_prim(exec_space, as_const(cons_arrays), prim_arrays, eos);
        }

        exec_space.fence();

        auto const end = std::chrono::steady_clock::now();
        double t = std::chrono::duration<double>(end - start).count();

        double cells = double(nt) * double(nx_local) * double(nx_local) * double(nx_local);

        double const to_mega = 1e-6;

        double mcell_s = (cells / t) * to_mega;

        std::ofstream f(out_file, std::ios::app);
        f << mode << "," << nx_local << "," << nt << "," << kernel << "," << threads << "," << t
          << "," << mcell_s << "\n";
    };

    auto run_vector = [&](int nx_local) {
        std::size_t const nxg_z = nx_local + 2;
        index_t const nxg = nx_local + 2;

        real_t const dx = 1.0 / nx_local;

        PerfectGas<real_t> eos(gamma);
        UniformMesh3d<real_t> mesh(dx, dx, dx);
        hllc riemann_solver;

        auto prim_alloc = create_prim_arrays_1d<real_t>(exec_space, nxg_z * nxg_z * nxg_z);

        auto cons_alloc = create_cons_arrays_1d<real_t>(exec_space, nxg_z * nxg_z * nxg_z);

        EulerPrimArrays prim_arrays = to_mdspan<Kokkos::mdspan<
                real_t,
                Kokkos::dextents<index_t, 3>,
                Kokkos::layout_left>>(prim_alloc, nxg, nxg, nxg);

        EulerConsArrays cons_arrays = to_mdspan<Kokkos::mdspan<
                real_t,
                Kokkos::dextents<index_t, 3>,
                Kokkos::layout_left>>(cons_alloc, nxg, nxg, nxg);

        init_implode(exec_space, prim_arrays, mesh);
        prim_to_cons_vec(exec_space, as_const(prim_arrays), cons_arrays, eos);

        exec_space.fence();

        auto const start = std::chrono::steady_clock::now();

        for (int it = 0; it < nt; ++it) {
            real_t dt = time_step_vec(exec_space, as_const(prim_arrays), eos, mesh);

            godunov_vec(
                    exec_space,
                    as_const(prim_arrays),
                    cons_arrays,
                    eos,
                    mesh,
                    riemann_solver,
                    cfl_factor * dt);

            boundary_conditions_periodic(exec_space, cons_arrays, 1);
            cons_to_prim_vec(exec_space, as_const(cons_arrays), prim_arrays, eos);
        }

        exec_space.fence();

        auto const end = std::chrono::steady_clock::now();
        double t = std::chrono::duration<double>(end - start).count();

        double cells = double(nt) * double(nx_local) * double(nx_local) * double(nx_local);

        double const to_mega = 1e-6;
        double mcell_s = (cells / t) * to_mega;

        std::ofstream f(out_file, std::ios::app);
        f << mode << "," << nx_local << "," << nt << "," << kernel << "," << threads << "," << t
          << "," << mcell_s << "\n";
    };

    if (kernel == "vector") {
        run_vector(nx);
    } else {
        run_scalar(nx);
    }

    return 0;
}
