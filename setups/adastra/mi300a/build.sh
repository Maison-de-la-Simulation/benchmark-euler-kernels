#!/bin/bash
set -xe

module purge
module load cpe/24.07
module load craype-accel-amd-gfx942 craype-x86-genoa
module load PrgEnv-cray
module load cmake/3.27.9
module load rocm/6.3.3

export CC=hipcc
export CXX=hipcc

############################################################
# 2. GPU TARGET
############################################################
export AMDGPU_TARGETS=gfx942

############################################################
# 3. INSTALL PREFIX
############################################################
export install_dir=$PWD/opt/mi300a

export Kokkos_ROOT=$install_dir/kokkos
export GTest_ROOT=$install_dir/googletest
export benchmark_ROOT=$install_dir/benchmark

mkdir -p "$install_dir"

############################################################
# 4. CLEAN BUILD AREA
############################################################
# rm -rf build-kokkos build-gtest build-benchmark build-mi300

############################################################
# 5. BUILD KOKKOS (CRITICAL FIRST STEP)
############################################################
git clone --branch 5.1.1 --depth 1 https://github.com/kokkos/kokkos.git

cmake -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D Kokkos_ARCH_AMD_GFX942_APU=ON \
  -D Kokkos_ENABLE_DEPRECATED_CODE_5=OFF \
  -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
  -D Kokkos_ENABLE_HIP=ON \
  -D Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON \
  -B build-kokkos \
  -S kokkos

cmake --build build-kokkos --parallel 8
cmake --install build-kokkos --prefix "$Kokkos_ROOT"

rm -rf kokkos build-kokkos

############################################################
# 6. BUILD GOOGLE TEST
############################################################
# git clone --branch v1.17.0 --depth 1 https://github.com/google/googletest.git

cmake -S googletest -B build-gtest \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-gtest \
  -S googletest

cmake --build build-gtest --parallel 8
cmake --install build-gtest --prefix "$GTest_ROOT"

rm -rf googletest build-gtest

############################################################
# 7. BUILD GOOGLE BENCHMARK
############################################################
git clone --branch v1.9.5 --depth 1 https://github.com/google/benchmark.git

cmake -S benchmark -B build-benchmark \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER=CC \
  -D CMAKE_CXX_STANDARD=20 \
  -D BENCHMARK_ENABLE_TESTING=OFF \
  -D BENCHMARK_DOWNLOAD_DEPENDENCIES=OFF \
  -B build-benchmark

cmake --build build-benchmark --parallel 8
cmake --install build-benchmark --prefix "$benchmark_ROOT"

rm -rf benchmark build-benchmark

############################################################
# 8. BUILD YOUR PROJECT
############################################################
cmake -S . -B build-mi300 \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-mi300 --parallel 8
