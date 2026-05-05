#!/bin/bash

module purge

module load \
  gcc-native/13.2 \
  cmake/3.27.9 \
  rocm/6.3.3

export CC=hipcc
export CXX=hipcc

export install_dir=$PWD/opt/mi300a
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
cmake --build build-benchmark --parallel 8
cmake --install build-benchmark --prefix "$benchmark_ROOT"
rm -rf build-benchmark benchmark

git clone https://github.com/kokkos/kokkos.git
cd kokkos || exit
git checkout 7f8988b4d
cd .. || exit

cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D Kokkos_ARCH_ZEN4=ON \
  -D Kokkos_ARCH_AMD_GFX942_APU=ON \
  -D Kokkos_ENABLE_DEPRECATED_CODE_4=OFF \
  -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
  -D Kokkos_ENABLE_HIP=ON \
  -D Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON \
  -B build-kokkos \
  -S kokkos
cmake --build build-kokkos --parallel 8
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

cmake -DGTest_ROOT="$gtest_ROOT" -D CMAKE_BUILD_TYPE=Release -B build-mi300
cmake --build build-mi300 --parallel 8
