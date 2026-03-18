cmake_minimum_required(VERSION 3.25)

project(euler_benchmarks CXX)

find_package(Kokkos REQUIRED)

add_subdirectory(benchmarks)

add_subdirectory(euler_operators)

add_subdirectory(simulations)
