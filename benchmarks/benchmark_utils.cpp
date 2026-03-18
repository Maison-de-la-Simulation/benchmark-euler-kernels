#include <benchmark/benchmark.h>

#include "benchmark_utils.hpp"

void set_constant_bytes_processed(benchmark::State& state, std::size_t const bytes)
{
    state.counters["bytes_per_second"] = benchmark::
            Counter(static_cast<double>(bytes), benchmark::Counter::kIsIterationInvariantRate);
}

void set_constant_cells_processed(benchmark::State& state, std::size_t const cells)
{
    state.counters["cells_per_second"] = benchmark::
            Counter(static_cast<double>(cells), benchmark::Counter::kIsIterationInvariantRate);
}
