#!/bin/bash

#SBATCH --account=cad16293
#SBATCH --job-name=euler-benchmarks-genoa-strong
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --constraint=GENOA
#SBATCH --threads-per-core=1

set -x

module purge

module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-cray

cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/adastra/genoa/bm_scaling

BENCHMARK_FILTER="Strong"
# BENCHMARK_FILTER=${1:-""}

export OMP_PLACES=threads
export OMP_PROC_BIND=close

export OMP_DISPLAY_AFFINITY=TRUE
export OMP_AFFINITY_FORMAT="thread %0.3n -> cpu %A"

THREAD_COUNTS=(
  1
  2
  4
  8
  16
  32
  64
  96
  128
  192
  256
  384
)

for T in "${THREAD_COUNTS[@]}"; do
  echo "=================================================="
  echo "Running ${BENCHMARK_FILTER} with ${T} threads"
  echo "=================================================="

  export OMP_NUM_THREADS=${T}

  OUTFILE="./results/adastra/genoa/bm_scaling/[${SLURM_JOB_ID}]_T${T}_${BENCHMARK_FILTER}.json"

  setarch $(uname -m) -R \
    ./build-genoa/benchmarks/euler_benchmarks \
    --benchmark_filter="${BENCHMARK_FILTER}" \
    --benchmark_out_format=json \
    --benchmark_out="${OUTFILE}"
done
