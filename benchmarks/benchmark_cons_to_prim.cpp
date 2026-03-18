#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <cons_to_prim.hpp>
#include <euler_arrays.hpp>
#include <perfect_gas.hpp>

#include "benchmark_utils.hpp"
#include "index_type.hpp"
#include "real_type.hpp"

namespace {

void ConsToPrim(benchmark::State& state)
{
    auto const n = int_cast<index_t>(state.range());
    PerfectGas<real_t> const eos(1.4);
    Kokkos::DefaultExecutionSpace const exec_space;
    EulerPrimArrays const prims_alloc = create_prim_arrays_1d<real_t>(exec_space, n * n * n);
    EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);
    EulerConsArrays const cons_alloc = create_cons_arrays_1d<real_t>(exec_space, n * n * n);
    EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);
    EulerCons<real_t> const cons {.d = 1, .e = 1 / 0.4, .mx0 = 0, .mx1 = 0, .mx2 = 0};
    init_from_state(exec_space, cons_arrays, cons);
    exec_space.fence();

    for ([[maybe_unused]] auto _ : state) {
        cons_to_prim(exec_space, as_const(cons_arrays), prim_arrays, eos);
        exec_space.fence();
        benchmark::ClobberMemory();
    }

    set_constant_cells_processed(state, size(prim_arrays));
    set_constant_bytes_processed(state, size_bytes(prim_arrays) + size_bytes(cons_arrays));
}

} // namespace

BENCHMARK(ConsToPrim)->DenseRange(8, 31, 8)->DenseRange(32, 320, 32);
