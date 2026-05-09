#!/bin/bash

#SBATCH --account=cad16293
#SBATCH --job-name=euler-benchmarks-mi300a
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --constraint=MI300
#SBATCH --threads-per-core=1

module purge
module load cpe/24.07
module load PrgEnv-amd
module load craype-accel-amd-gfx942

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/bm_json/adastra/
BENCHMARK_FILTER=${1:-""}

export OMP_NUM_THREADS=1
export HSA_XNACK=1
export CXX=hipcc

SAVE_FILTER=$(echo "$BENCHMARK_FILTER" | sed 's/[()|^\/]/_/g')
./build-mi300/benchmarks/euler_benchmarks \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  --benchmark_out_format=json \
  --benchmark_out=./results/adastra/mi300/bm_json/"[${SLURM_JOB_ID}]_mi300_${SAVE_FILTER}.json"

# --benchmark_dry_run \
