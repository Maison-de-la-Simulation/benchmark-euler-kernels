#!/bin/bash
#MSUB -r build_euler_grace
#MSUB -q gh200-bxi
#MSUB -n 1
#MSUB -c 8
#MSUB -T 1800
#MSUB -o ./slurm_out/build_grace%I.out
#MSUB -e ./slurm_out/build_grace%I.err
#MSUB -A INTI0046

set -x

cd "${BRIDGE_MSUB_PWD}" || exit

mkdir -p slurm_out

module purge
module load gcc/13.3.0
module load cmake/3.31.4

export install_dir="$PWD/opt/grace"
export Kokkos_ROOT="$install_dir/kokkos"
export benchmark_ROOT="$install_dir/benchmark"
export GTest_ROOT="$install_dir/googletest"

# build kokkos
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D Kokkos_ENABLE_OPENMP=ON \
  -D Kokkos_ARCH_ARMV9_GRACE=ON \
  -D Kokkos_ENABLE_DEPRECATED_CODE_5=OFF \
  -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
  -B build-kokkos \
  -S kokkos
cmake --build build-kokkos --parallel 8
cmake --install build-kokkos --prefix "$Kokkos_ROOT"
# rm -rf build-kokkos

# build google benchmark
cmake \
  -D BENCHMARK_ENABLE_TESTING=OFF \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-benchmark \
  -S benchmark
cmake --build build-benchmark --parallel 8
cmake --install build-benchmark --prefix "$benchmark_ROOT"
# rm -rf build-benchmark

# build google test
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-gtest \
  -S googletest
cmake --build build-gtest --parallel 8
cmake --install build-gtest --prefix "$GTest_ROOT"
# rm -rf build-gtest
BUILD_DIR=build-grace

cmake \
  -B "${BUILD_DIR}" \
  -S . \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D CMAKE_CXX_COMPILER=g++ \
  -D CMAKE_CXX_FLAGS="-mcpu=neoverse-v2 -msve-vector-bits=128" \
  -D Kokkos_DIR="$install_dir/kokkos/lib64/cmake/Kokkos" \
  -D benchmark_DIR="$benchmark_ROOT/lib64/cmake/benchmark" \
  -D GTest_ROOT="${GTest_ROOT}"

cmake --build "${BUILD_DIR}" --parallel 8
