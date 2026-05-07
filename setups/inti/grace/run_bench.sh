#!/bin/bash
#MSUB

module purge

module load \
  gcc-13.2.0 \
  cmake/3.29.6

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/inti/grace/
BENCHMARK_FILTER=${1:-""}

export OMP_NUM_THREADS=1
# export OMP_PROC_BIND=CLOSE
# export OMP_PLACES=THREADS

# export OMP_DISPLAY_AFFINITY=TRUE
# export OMP_AFFINITY_FORMAT="thread %0.3n -> cpu %A"
# numactl -H

# srun bash -c 'echo $SLURM_CPUS_PER_TASK; grep Cpus_allowed_list /proc/self/status'

./build-genoa/benchmarks/euler_benchmarks \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  --benchmark_out_format=json \
  --benchmark_out=./results/bm_json/adastra/"[${SLURM_JOB_ID}]_BASE_${BENCHMARK_FILTER}.json"
# --benchmark_out=./results/adastra/genoa/bm_json/mt/"[${SLURM_JOB_ID}]_T${OMP_NUM_THREADS}-${OMP_PROC_BIND}-${OMP_PLACES}_${BENCHMARK_FILTER}.json"
