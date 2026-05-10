#!/bin/bash

module purge

module load \
  gcc/13.4.0/gcc-15.1.0 \
  cmake/3.31.9/gcc-15.1.0 \
  cuda/12.8.1/none-none

export install_dir=$PWD/opt/v100
export Kokkos_ROOT=$install_dir/kokkos
export benchmark_ROOT=$install_dir/benchmark
export GTest_ROOT=$install_dir/googletest

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

git clone --branch 5.1.1 --depth 1 https://github.com/kokkos/kokkos.git
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D Kokkos_ARCH_VOLTA70=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_DEPRECATED_CODE_5=OFF \
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
  -B build-googletest \
  -S googletest
cmake --build build-googletest
cmake --install build-googletest --prefix "$GTest_ROOT"
rm -rf build-googletest googletest

cmake -D CMAKE_BUILD_TYPE=Release -B build-v100
cmake --build build-v100
