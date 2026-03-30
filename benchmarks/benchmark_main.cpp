#include <iostream>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv)
{
    ::benchmark::MaybeReenterWithoutASLR(argc, argv);
    ::Kokkos::ScopeGuard const scope(argc, argv);
    ::benchmark::Initialize(&argc, argv);
    Kokkos::print_configuration(std::cout);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
