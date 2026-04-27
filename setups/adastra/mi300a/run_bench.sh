#!/bin/bash
#SBATCH --account=cad16293
#SBATCH --job-name=bench_mi300
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

# A CrayPE environment version
module load cpe/24.07
# An architecture
module load craype-accel-amd-gfx942 craype-x86-mi300
# A compiler to target the architecture
module load PrgEnv-cray
# Some architecture related libraries and tools
module load amd-mixed

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/adastra/mi300/bm_json/
BENCHMARK_FILTER=${1:-""}

export OMP_NUM_THREADS=1

./build-mi300/benchmarks/euler_benchmarks \
  --benchmark_dry_run \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  --benchmark_out_format=json \
  --benchmark_out=./results/adastra/mi300/bm_json/"[${SLURM_JOB_ID}]_${BENCHMARK_FILTER}.json"
