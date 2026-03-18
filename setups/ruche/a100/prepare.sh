#!/bin/bash

module purge

module load \
    gcc/11.2.0/gcc-4.8.5 \
    cmake/3.28.3/gcc-11.2.0 \
    cuda/12.2.1/gcc-11.2.0

export install_dir=$PWD/opt/a100
export Kokkos_ROOT=$install_dir/kokkos
export benchmark_ROOT=$install_dir/benchmark

git clone --branch v1.9.4 --depth 1 https://github.com/google/benchmark.git
cmake \
  -D BENCHMARK_ENABLE_TESTING=OFF \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-benchmark \
  -S benchmark
cmake --build build-benchmark
cmake --install build-benchmark --prefix $benchmark_ROOT
rm -rf build-benchmark benchmark

git clone --branch fix-simd-from-4.7.1 --depth 1 https://github.com/tpadioleau/kokkos.git
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
cmake --install build-kokkos --prefix $Kokkos_ROOT
rm -rf build-kokkos kokkos

cmake -D CMAKE_BUILD_TYPE=Release -B build-a100
cmake --build build-a100
