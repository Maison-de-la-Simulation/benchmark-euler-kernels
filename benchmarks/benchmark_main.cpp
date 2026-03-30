#include <iostream>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv)
{
    ::Kokkos::ScopeGuard const scope(argc, argv);
    Kokkos::print_configuration(std::cout);
    ::benchmark::Initialize(&argc, argv);
    // ::benchmark::MaybeReenterWithoutASLR(argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
