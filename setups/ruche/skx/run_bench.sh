#!/bin/bash
#SBATCH --job-name=bench_skx
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:20:00
#SBATCH --partition=cpu_short
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --nodes=1

module purge
module load gcc/13.4.0/gcc-15.1.0 cmake/3.31.9/gcc-15.1.0
# numactl/2.0.19/gcc-15.1.0
set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/ruche/skx
BENCHMARK_FILTER=${1:-""}

# # Single-threaded baseline
# echo "========== RUNNING WITH 1 THREAD (baseline) =========="
# export OMP_NUM_THREADS=1
# unset OMP_PROC_BIND
# unset OMP_PLACES
# ./build-skx/benchmarks/euler_benchmarks \
#   --benchmark_filter="${BENCHMARK_FILTER}" \
#   --benchmark_out_format=json \
#   --benchmark_out=./results/ruche/skx/mt/"[${SLURM_JOB_ID}]_T1_baseline_${BENCHMARK_FILTER}.json"

# 20 threads on one socket
echo "========== RUNNING WITH 20 THREADS (socket 0) =========="
export OMP_NUM_THREADS=20
export OMP_PROC_BIND=close
# export OMP_PLACES=cores
export OMP_PLACES=numa_domains

./build-skx/benchmarks/euler_benchmarks \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  --benchmark_out_format=json \
  --benchmark_out=./results/ruche/skx/mt/"[${SLURM_JOB_ID}]_T20-debug${BENCHMARK_FILTER}.json"
