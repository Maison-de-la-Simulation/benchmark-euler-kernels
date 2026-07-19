#include <cstddef>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include <euler_arrays.hpp>
#include <godunov.hpp>
#include <hllc.hpp>
#include <perfect_gas.hpp>
#include <uniform_mesh.hpp>
#include <utils.hpp>

#include "benchmark_utils.hpp"
#include "index_type.hpp"
#include "real_type.hpp"

namespace {

void Godunov(benchmark::State& state)
{
    auto const n = int_cast<index_t>(state.range());
    std::size_t const ng_z = n + 2;
    index_t const ng = n + 2;

    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);
    real_t const dt = 1E-9;
    Kokkos::DefaultExecutionSpace const exec_space;
    EulerPrimArrays const prims_alloc
            = create_prim_arrays_1d<real_t>(exec_space, ng_z * ng_z * ng_z);
    EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, ng, ng, ng);
    EulerConsArrays const cons_alloc
            = create_cons_arrays_1d<real_t>(exec_space, ng_z * ng_z * ng_z);
    EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, ng, ng, ng);
    EulerPrim<real_t> const prim {.d = 1, .p = 1, .ux0 = 0, .ux1 = 0, .ux2 = 0};
    init_from_state(exec_space, prim_arrays, prim);
    init_from_state(exec_space, cons_arrays, to_cons(prim, eos.internal_energy(prim.d, prim.p)));
    exec_space.fence();

    for ([[maybe_unused]] auto _ : state) {
        godunov(exec_space, as_const(prim_arrays), cons_arrays, eos, mesh, hllc(), dt);
        exec_space.fence();
        benchmark::ClobberMemory();
    }

    set_constant_cells_processed(state, size(cons_arrays));
    set_constant_bytes_processed(state, size_bytes(prim_arrays) + (2 * size_bytes(cons_arrays)));
}

void GodunovOpti(benchmark::State& state)
{
    auto const n = int_cast<index_t>(state.range());
    std::size_t const ng_z = n + 2;
    index_t const ng = n + 2;

    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);
    real_t const dt = 1E-9;
    Kokkos::DefaultExecutionSpace const exec_space;
    EulerPrimArrays const prims_alloc
            = create_prim_arrays_1d<real_t>(exec_space, ng_z * ng_z * ng_z);
    EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, ng, ng, ng);
    EulerConsArrays const cons_alloc
            = create_cons_arrays_1d<real_t>(exec_space, ng_z * ng_z * ng_z);
    EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, ng, ng, ng);
    EulerPrim<real_t> const prim {.d = 1, .p = 1, .ux0 = 0, .ux1 = 0, .ux2 = 0};
    init_from_state(exec_space, prim_arrays, prim);
    init_from_state(exec_space, cons_arrays, to_cons(prim, eos.internal_energy(prim.d, prim.p)));
    exec_space.fence();

    for ([[maybe_unused]] auto _ : state) {
        godunov_opti(exec_space, as_const(prim_arrays), cons_arrays, eos, mesh, hllc(), dt);
        exec_space.fence();
        benchmark::ClobberMemory();
    }

    set_constant_cells_processed(state, size(cons_arrays));
    set_constant_bytes_processed(state, size_bytes(prim_arrays) + (2 * size_bytes(cons_arrays)));
}

void GodunovWorstRem(benchmark::State& state)
{
    auto const n = int_cast<index_t>(state.range() + 2);
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);
    real_t const dt = 1E-9;
    auto nn = static_cast<std::size_t>(n);
    std::size_t const n3 = nn * nn * nn;

    Kokkos::DefaultExecutionSpace const exec_space;
    EulerPrimArrays const prims_alloc = create_prim_arrays_1d<real_t>(exec_space, n3);
    EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);
    EulerConsArrays const cons_alloc = create_cons_arrays_1d<real_t>(exec_space, n3);
    EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);
    EulerPrim<real_t> const prim {.d = 1, .p = 1, .ux0 = 0, .ux1 = 0, .ux2 = 0};
    init_from_state(exec_space, prim_arrays, prim);
    init_from_state(exec_space, cons_arrays, to_cons(prim, eos.internal_energy(prim.d, prim.p)));
    exec_space.fence();

    for ([[maybe_unused]] auto _ : state) {
        godunov(exec_space, as_const(prim_arrays), cons_arrays, eos, mesh, hllc(), dt);
        exec_space.fence();
        benchmark::ClobberMemory();
    }

    set_constant_cells_processed(state, size(cons_arrays));
    set_constant_bytes_processed(state, size_bytes(prim_arrays) + (2 * size_bytes(cons_arrays)));
}

void GodunovVectorized(benchmark::State& state)
{
    auto const n = int_cast<index_t>(state.range() + 2);
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);
    real_t const dt = 1E-9;
    auto nn = static_cast<std::size_t>(n);
    std::size_t const n3 = nn * nn * nn;

    Kokkos::DefaultExecutionSpace const exec_space;
    EulerPrimArrays const prims_alloc = create_prim_arrays_1d<real_t>(exec_space, n3);
    EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);
    EulerConsArrays const cons_alloc = create_cons_arrays_1d<real_t>(exec_space, n3);
    EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);
    EulerPrim<real_t> const prim {.d = 1, .p = 1, .ux0 = 0, .ux1 = 0, .ux2 = 0};
    init_from_state(exec_space, prim_arrays, prim);
    init_from_state(exec_space, cons_arrays, to_cons(prim, eos.internal_energy(prim.d, prim.p)));
    exec_space.fence();

    for ([[maybe_unused]] auto _ : state) {
        godunov_vec(exec_space, as_const(prim_arrays), cons_arrays, eos, mesh, hllc(), dt);
        exec_space.fence();
        benchmark::ClobberMemory();
    }

    set_constant_cells_processed(state, size(cons_arrays));
    set_constant_bytes_processed(state, size_bytes(prim_arrays) + (2 * size_bytes(cons_arrays)));
}

void GodunovVectorized2(benchmark::State& state)
{
    auto const n = int_cast<index_t>(state.range() + 2);
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);
    real_t const dt = 1E-9;
    auto nn = static_cast<std::size_t>(n);
    std::size_t const n3 = nn * nn * nn;

    Kokkos::DefaultExecutionSpace const exec_space;
    EulerPrimArrays const prims_alloc = create_prim_arrays_1d<real_t>(exec_space, n3);
    EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);
    EulerConsArrays const cons_alloc = create_cons_arrays_1d<real_t>(exec_space, n3);
    EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);
    EulerPrim<real_t> const prim {.d = 1, .p = 1, .ux0 = 0, .ux1 = 0, .ux2 = 0};
    init_from_state(exec_space, prim_arrays, prim);
    init_from_state(exec_space, cons_arrays, to_cons(prim, eos.internal_energy(prim.d, prim.p)));
    exec_space.fence();

    for ([[maybe_unused]] auto _ : state) {
        godunov_vec2(exec_space, as_const(prim_arrays), cons_arrays, eos, mesh, hllc(), dt);
        exec_space.fence();
        benchmark::ClobberMemory();
    }

    set_constant_cells_processed(state, size(cons_arrays));
    set_constant_bytes_processed(state, size_bytes(prim_arrays) + (2 * size_bytes(cons_arrays)));
}

void GodunovVectorizedWorstRem(benchmark::State& state)
{
    auto const n = int_cast<index_t>(state.range() + 2);
    PerfectGas<real_t> const eos(1.4);
    UniformMesh3d<real_t> const mesh(1., 1., 1.);
    real_t const dt = 1E-9;

    auto nn = static_cast<std::size_t>(n);
    std::size_t const n3 = nn * nn * nn;

    Kokkos::DefaultExecutionSpace const exec_space;
    EulerPrimArrays const prims_alloc = create_prim_arrays_1d<real_t>(exec_space, n3);
    EulerPrimArrays const prim_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(prims_alloc, n, n, n);
    EulerConsArrays const cons_alloc = create_cons_arrays_1d<real_t>(exec_space, n3);
    EulerConsArrays const cons_arrays = to_mdspan<Kokkos::mdspan<
            real_t,
            Kokkos::dextents<index_t, 3>,
            Kokkos::layout_left>>(cons_alloc, n, n, n);
    EulerPrim<real_t> const prim {.d = 1, .p = 1, .ux0 = 0, .ux1 = 0, .ux2 = 0};
    init_from_state(exec_space, prim_arrays, prim);
    init_from_state(exec_space, cons_arrays, to_cons(prim, eos.internal_energy(prim.d, prim.p)));
    exec_space.fence();

    for ([[maybe_unused]] auto _ : state) {
        godunov_vec(exec_space, as_const(prim_arrays), cons_arrays, eos, mesh, hllc(), dt);
        exec_space.fence();
        benchmark::ClobberMemory();
    }

    set_constant_cells_processed(state, size(cons_arrays));
    set_constant_bytes_processed(state, size_bytes(prim_arrays) + (2 * size_bytes(cons_arrays)));
}

} // namespace

// BENCHMARK(Godunov)
//         ->UseRealTime()
//         ->DenseRange(8, 31, 8)
//         ->DenseRange(32, 127, 16)
//         ->DenseRange(128, 320, 32);

// BENCHMARK(GodunovOpti)
//         ->UseRealTime()
//         ->DenseRange(8, 31, 8)
//         ->DenseRange(32, 127, 16)
//         ->DenseRange(128, 320, 32);

BENCHMARK(GodunovVectorized)->Arg(128);

//         ->UseRealTime()
//         ->DenseRange(8, 31, 8)
//         ->DenseRange(32, 127, 16)
//         ->DenseRange(128, 320, 32);

BENCHMARK(GodunovVectorized2)->Arg(128);
// ->UseRealTime()
// ->DenseRange(8, 31, 8)
// ->DenseRange(32, 127, 16)
// ->DenseRange(128, 320, 32);

// BENCHMARK(GodunovWorstRem)->UseRealTime()->DenseRange(7, 31, 8)->DenseRange(31, 320, 32);
// BENCHMARK(GodunovVectorizedWorstRem)->UseRealTime()->DenseRange(7, 31, 8)->DenseRange(31, 320, 32);
