#include <stdexcept>
#include <utility>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

#define BENCHMARK_RT(func) BENCHMARK(func)->UseRealTime()


template <class R, class T>
R int_cast(T t)
{
    if (std::in_range<R>(t)) {
        return static_cast<R>(t);
    }
    throw std::runtime_error("Conversion cannot preserve value representation");
}


void set_constant_bytes_processed(benchmark::State& state, std::size_t bytes);

void set_constant_cells_processed(benchmark::State& state, std::size_t cells);
