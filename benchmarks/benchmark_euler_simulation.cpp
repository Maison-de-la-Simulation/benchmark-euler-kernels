#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <cons_to_prim.hpp>
#include <euler_arrays.hpp>
#include <godunov.hpp>
#include <hllc.hpp>
#include <init_implode.hpp>
#include <perfect_gas.hpp>
#include <prim_to_cons.hpp>
#include <time_step.hpp>
#include <uniform_mesh.hpp>

#include "benchmark_utils.hpp"
#include "index_type.hpp"
#include "real_type.hpp"

namespace {

void EulerSimulation(benchmark::State& state)
{
    auto const nx = int_cast<index_t>(state.range());
    real_t const cfl_factor = 0.49;
    real_t const gamma = 1.4;

    real_t const dx = 1. / static_cast<real_t>(nx);
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

    for ([[maybe_unused]] auto _ : state) {
        real_t const dt = time_step(exec_space, as_const(prim_arrays), eos, mesh);

        godunov(exec_space,
                as_const(prim_arrays),
                cons_arrays,
                eos,
                mesh,
                riemann_solver,
                cfl_factor * dt);

        cons_to_prim(exec_space, as_const(cons_arrays), prim_arrays, eos);
        exec_space.fence();
    }
    set_constant_cells_processed(state, size(cons_arrays));
}

} // namespace

BENCHMARK(EulerSimulation)->DenseRange(16, 320, 32);
