#!/bin/bash

module purge

module load \
    gcc-native/13.2 \
    cmake/3.27.9

export install_dir=$PWD/opt/genoa
export Kokkos_ROOT=$install_dir/kokkos
export benchmark_ROOT=$install_dir/benchmark

git clone --branch v1.9.4 --depth 1 https://github.com/google/benchmark.git
cmake \
  -D BENCHMARK_ENABLE_TESTING=OFF \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-benchmark \
  -S benchmark
cmake --build build-benchmark --parallel 8
cmake --install build-benchmark --prefix $benchmark_ROOT
rm -rf build-benchmark benchmark

git clone --branch fix-simd-from-4.7.1 --depth 1 https://github.com/tpadioleau/kokkos.git
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D Kokkos_ARCH_ZEN4=ON \
  -D Kokkos_ENABLE_DEPRECATED_CODE_4=OFF \
  -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
  -D Kokkos_ENABLE_OPENMP=ON \
  -B build-kokkos \
  -S kokkos
cmake --build build-kokkos --parallel 8
cmake --install build-kokkos --prefix $Kokkos_ROOT
rm -rf build-kokkos kokkos

cmake -D CMAKE_BUILD_TYPE=Release -B build-genoa
cmake --build build-genoa --parallel 8
