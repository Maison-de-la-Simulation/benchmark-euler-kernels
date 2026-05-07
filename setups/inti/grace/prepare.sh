#!/bin/bash

module load nvhpc/26.3 \
  mpi/openmpi/5 \
  flavor/hdf5/parallel \
  cmake/3.31.4
# /ccc/products/openmpi-5.0.8/nvidia--26.3__cuda--13.0/default/

export install_dir=$PWD/opt/gh200
export Kokkos_ROOT=$install_dir/kokkos
export benchmark_ROOT=$install_dir/benchmark
export gtest_ROOT=$install_dir/gtest

# ========================
# benchmark
# ========================
cmake \
  -D BENCHMARK_ENABLE_TESTING=OFF \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-benchmark \
  -S benchmark
cmake --build build-benchmark --parallel 8
cmake --install build-benchmark --prefix "$benchmark_ROOT"
rm -rf build-benchmark

# ========================
# kokkos
# ========================
cmake \
  -D CMAKE_C_COMPILER=$(which gcc) \
  -D CMAKE_CXX_COMPILER=$(which c++) \
  -D CMAKE_BUILD_TYPE=Release \
  -D Kokkos_ARCH_ARMV9_GRACE=ON \
  -D Kokkos_ENABLE_OPENMP=ON \
  -D CMAKE_CXX_FLAGS="-mcpu=neoverse-v2+crypto+sve2-aes+sve2-sha3+sve2-sm4+norng" \
  -B build-kokkos \
  -S .
cmake --build build-kokkos --parallel 8
cmake --install build-kokkos --prefix "$Kokkos_ROOT"
rm -rf build-kokkos

# ========================
# gtest
# ========================
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-gtest \
  -S googletest
cmake --build build-gtest --parallel 8
cmake --install build-gtest --prefix "$gtest_ROOT"
rm -rf build-gtest

# ========================
# your project
# ========================
cmake \
  -DGTest_ROOT="$gtest_ROOT" \
  -DCMAKE_BUILD_TYPE=Release \
  -B build-gh200

cmake --build build-gh200 --parallel 8
