#!/bin/bash
#MSUB -r euler_bench_grace
#MSUB -q gh200-bxi
#MSUB -n 1
#MSUB -c 1
#MSUB -x
#MSUB -T 3600
#MSUB -o ./slurm_out/bench_grace_%I.out
#MSUB -e ./slurm_out/bench_grace_%I.err
#MSUB -A INTI0046

set -x

cd "${BRIDGE_MSUB_PWD}" || exit

mkdir -p slurm_out results/inti

module purge
module load gcc/13.3.0

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores

BENCHMARK_FILTER="Godunov"

./build-grace/benchmarks/euler_benchmarks \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  --benchmark_out_format=json \
  --benchmark_out="./results/inti/[${MOAB_JOBID}]_grace-${BENCHMARK_FILTER}.json"
