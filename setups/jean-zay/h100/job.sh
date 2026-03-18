#!/bin/bash

#SBATCH --job-name=spack              # Job name
#SBATCH --account=ksw@h100
#SBATCH --constraint=h100
#SBATCH --ntasks=1                   # Number of MPI processes (= total number of GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --gres=gpu:1                 # nombre de GPU par nœud (max 8 avec gpu_p2)
#SBATCH --cpus-per-task=24           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=02:00:00              # temps d'execution maximum demande (HH:MM:SS)
#SBATCH --qos=qos_gpu_h100-dev

set -ex

module purge

module load \
    arch/h100 \
    gcc/12.2.0 \
    cmake/3.31.4 \
    cuda/12.8.0

export install_dir=$PWD/opt/h100
export Kokkos_ROOT=$install_dir/kokkos
export benchmark_ROOT=$install_dir/benchmark

cmake \
  -D BENCHMARK_ENABLE_TESTING=OFF \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -B build-benchmark \
  -S benchmark
cmake --build build-benchmark --parallel 24
cmake --install build-benchmark --prefix $benchmark_ROOT
rm -rf build-benchmark benchmark

cmake \
  -D BUILD_TESTING=OFF \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_STANDARD=20 \
  -D Kokkos_ARCH_HOPPER90=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_DEPRECATED_CODE_4=OFF \
  -D Kokkos_ENABLE_DEPRECATION_WARNINGS=OFF \
  -B build-kokkos \
  -S kokkos
cmake --build build-kokkos --parallel 24
cmake --install build-kokkos --prefix $Kokkos_ROOT
rm -rf build-kokkos kokkos

cmake -D CMAKE_BUILD_TYPE=Release -B build-h100
cmake --build build-h100 --parallel 24

srun ./build-h100/euler_benchmarks --kokkos-print-configuration
