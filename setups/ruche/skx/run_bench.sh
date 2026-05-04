#!/bin/bash
#SBATCH --job-name=bench_skx
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --partition=cpu_short
#SBATCH --exclusive
#SBATCH --hint=nomultithread

module purge
module load \
  gcc/13.4.0/gcc-15.1.0 \
  cmake/3.31.9/gcc-15.1.0

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/ruche/skx

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=true

BENCHMARK_FILTER=${1:-""}

# include SLURM_JOB_ID in the JSON output filename
./build-skx/benchmarks/euler_benchmarks \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  --benchmark_out_format=json \
  --benchmark_out=./results/ruche/skx/"[${SLURM_JOB_ID}]_skx-${BENCHMARK_FILTER}.json"
