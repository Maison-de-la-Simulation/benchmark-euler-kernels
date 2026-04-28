#include <iostream>

#include <benchmark/benchmark.h>
#include <impl/Kokkos_InitializeFinalize.hpp>

#include <Kokkos_Core.hpp>
#include <omp.h>

int main(int argc, char** argv)
{
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int cpu = sched_getcpu();
        printf(" BEFORE KOKKKOS INIT: Benchmark startup: thread %d on cpu %d\n", tid, cpu);
    }
    ::benchmark::MaybeReenterWithoutASLR(argc, argv);
    // ::Kokkos::ScopeGuard const scope(argc, argv);
    Kokkos::InitializationSettings settings;
    settings.set_num_threads(10);
    Kokkos::initialize(settings);
    {
        // Re-spread threads after Kokkos init
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(tid * 2, &set);
            pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

            int cpu = sched_getcpu();
            printf(" AFTER KOKKOS INIT: Benchmark startup: thread %d on cpu %d\n", tid, cpu);
        }

        ::benchmark::Initialize(&argc, argv);
        Kokkos::print_configuration(std::cout);
        if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
            return 1;
        }
        ::benchmark::RunSpecifiedBenchmarks();
        ::benchmark::Shutdown();
        return 0;
    }
    Kokkos::finalize();
}
