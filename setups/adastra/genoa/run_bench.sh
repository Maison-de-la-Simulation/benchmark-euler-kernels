#!/bin/bash

#SBATCH --account=cad16293
#SBATCH --job-name=euler-benchmarks-genoa
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --constraint=GENOA
#SBATCH --threads-per-core=1

module purge

module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-cray

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/bm_json/adastra/
BENCHMARK_FILTER=${1:-""}

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=CLOSE

./build-genoa/benchmarks/euler_benchmarks \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  --benchmark_out_format=json \
  --benchmark_out=./results/bm_json/adastra/"[${SLURM_JOB_ID}]_BASE_${BENCHMARK_FILTER}.json"
