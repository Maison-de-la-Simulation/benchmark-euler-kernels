cmake_minimum_required(VERSION 3.25)

project(euler_benchmarks CXX)

find_package(Kokkos 5...<6 REQUIRED)

add_subdirectory(euler_utils)

add_subdirectory(benchmarks)

add_subdirectory(euler_operators)

add_subdirectory(simulations)

include(CTest)
add_subdirectory(tests)
