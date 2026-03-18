find_package(benchmark REQUIRED)

add_executable(euler_benchmarks)
target_link_libraries(
    euler_benchmarks
    PRIVATE benchmark::benchmark euler_operators Kokkos::kokkos
)
target_sources(
    euler_benchmarks
    PRIVATE
        benchmark_cons_to_prim.cpp
        benchmark_euler_simulation.cpp
        benchmark_godunov.cpp
        benchmark_main.cpp
        benchmark_prim_to_cons.cpp
        benchmark_time_step.cpp
        benchmark_utils.cpp
)
