#!/bin/bash
#SBATCH --job-name=bench_skx
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:04:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

module purge
module load \
  gcc/13.4.0/gcc-15.1.0 \
  cmake/3.31.9/gcc-15.1.0 \
  cuda/12.8.1/none-none

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/ruche/

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=true

BENCHMARK_FILTER=${1:-""}

# include SLURM_JOB_ID in the JSON output filename
./build-a100/benchmarks/euler_benchmarks \
  --benchmark_out_format=json \
  --benchmark_out=./results/ruche/"[${SLURM_JOB_ID}]_a100-${BENCHMARK_FILTER}.json" # --benchmark_filter="${BENCHMARK_FILTER}" \

##SBATCH --exclusive
##SBATCH --hint=nomultithread
