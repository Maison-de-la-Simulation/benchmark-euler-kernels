#!/bin/bash

#SBATCH --account=cad16293
#SBATCH --job-name=euler-benchmarks-mi250
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --constraint=MI250
#SBATCH --threads-per-core=1

export CC=cc
export CXX=CC

module purge
module load cpe/24.07
module load PrgEnv-amd
module load craype-accel-amd-gfx90a

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/adastra/mi250/
BENCHMARK_FILTER=${1:-""}

export OMP_NUM_THREADS=1
export HSA_XNACK=1
export CXX=hipcc

SAVE_FILTER=$(echo "$BENCHMARK_FILTER" | sed 's/[()|^\/]/_/g')
./build-mi250/benchmarks/euler_benchmarks \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  --benchmark_out_format=json \
  --benchmark_out=./results/adastra/mi250/"${SLURM_JOB_ID}_mi250_${SAVE_FILTER}.json"
