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
#include <omp.h>
#include <perfect_gas.hpp>
#include <periodic_boundary_conditions.hpp>
#include <prim_to_cons.hpp>
#include <sched.h>
#include <time_step.hpp>
#include <uniform_mesh.hpp>
#include <unistd.h>

static inline void print_affinity()
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

int main(int argc, char** argv)
{
    using index_t = int;
    using real_t = double;

    std::string mode = "strong";
    std::string out_file = "results.csv";

    int nx_base = 256;
    int nt = 200;
    int warmup = 2;
    int repeats = 5;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--mode") {
            mode = argv[++i];
        } else if (arg == "--nx") {
            nx_base = std::stoi(argv[++i]);
        } else if (arg == "--nt") {
            nt = std::stoi(argv[++i]);
        } else if (arg == "--warmup") {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--repeats") {
            repeats = std::stoi(argv[++i]);
        } else if (arg == "--out") {
            out_file = argv[++i];
        }
    }

    Kokkos::ScopeGuard const scope(argc, argv);
    Kokkos::DefaultExecutionSpace const exec_space;

    print_affinity();

    real_t const cfl_factor = 0.49;
    real_t const gamma = 1.4;

    auto run = [&](int nx) {
        std::size_t const nxg_z = nx + 2;
        index_t const nxg = nx + 2;
        real_t const dx = 1.0 / nx;

        PerfectGas<real_t> const eos(gamma);
        UniformMesh3d<real_t> const mesh(dx, dx, dx);
        hllc const riemann_solver;

        EulerPrimArrays const prims_alloc
                = create_prim_arrays_1d<real_t>(exec_space, nxg_z * nxg_z * nxg_z);

        EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
                real_t,
                Kokkos::dextents<index_t, 3>,
                Kokkos::layout_left>>(prims_alloc, nxg, nxg, nxg);

        EulerConsArrays const cons_alloc
                = create_cons_arrays_1d<real_t>(exec_space, nxg_z * nxg_z * nxg_z);

        EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
                real_t,
                Kokkos::dextents<index_t, 3>,
                Kokkos::layout_left>>(cons_alloc, nxg, nxg, nxg);

        init_implode(exec_space, prim_arrays, mesh);
        prim_to_cons(exec_space, as_const(prim_arrays), cons_arrays, eos);
        exec_space.fence();

        double best_time = 1e100;

        for (int r = 0; r < repeats + warmup; ++r) {
            auto const start = std::chrono::steady_clock::now();

            for (int it = 0; it < nt; ++it) {
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
            }

            exec_space.fence();

            auto const end = std::chrono::steady_clock::now();
            double t = std::chrono::duration<double>(end - start).count();

            if (r >= warmup && t < best_time) {
                best_time = t;
            }
        }

        double cells = double(nt) * double(nx * nx * nx);
        double mcell_s = (cells / best_time) * 1e-6;

        std::ofstream f(out_file, std::ios::app);
        f << mode << "," << nx << "," << nt << "," << omp_get_max_threads() << "," << best_time
          << "," << mcell_s << "\n";
    };

    if (mode == "strong") {
        run(nx_base);
    } else {
        int threads = omp_get_max_threads();
        run(nx_base * threads);
    }

    return 0;
}
