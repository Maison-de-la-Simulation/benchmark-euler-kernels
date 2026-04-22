#!/bin/bash

set -e

module purge
module load \
  gcc/13.4.0/gcc-15.1.0 \
  cmake/3.31.9/gcc-15.1.0 \
  cuda/12.8.1/none-none
export install_dir=$PWD/opt/a100
export Kokkos_ROOT=$install_dir/kokkos
export benchmark_ROOT=$install_dir/benchmark
export gtest_ROOT=$install_dir/gtest

git clone --branch v1.9.4 --depth 1 https://github.com/google/benchmark.git
cmake \
  -D BENCHMARK_ENABLE_TESTING=OFF \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-benchmark \
  -S benchmark
cmake --build build-benchmark
cmake --install build-benchmark --prefix "$benchmark_ROOT"
rm -rf build-benchmark benchmark

git clone https://github.com/kokkos/kokkos.git
cd kokkos || exit
git checkout 7f8988b4d
cd .. || exit

cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D Kokkos_ARCH_AMPERE80=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_DEPRECATED_CODE_4=OFF \
  -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
  -B build-kokkos \
  -S kokkos
cmake --build build-kokkos
cmake --install build-kokkos --prefix "$Kokkos_ROOT"
rm -rf build-kokkos kokkos

git clone --branch v1.17.0 --depth 1 https://github.com/google/googletest.git
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-gtest \
  -S googletest
cmake --build build-gtest
cmake --install build-gtest --prefix "$gtest_ROOT"
rm -rf build-gtest googletest

cmake -D CMAKE_BUILD_TYPE=Release -B build-a100
cmake --build build-a100
