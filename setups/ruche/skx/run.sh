#!/bin/bash
#SBATCH --job-name=benchmark_skx
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:10:00
#SBATCH --partition=cpu_short

module purge
module load \
  gcc/13.4.0/gcc-15.1.0 \
  cmake/3.31.9/gcc-15.1.0

set -x
cd ${SLURM_SUBMIT_DIR}

# $1 = optional base name
RESULT_NAME=${1:-"unnamed"}

echo "RESULT_NAME =" "$RESULT_NAME"

mkdir -p slurm_out results/ruche/skx

./build-skx/benchmarks/euler_benchmarks \
  --benchmark_out_format=json \
  --benchmark_out="./results/ruche/skx/[${SLURM_JOB_ID}]_${RESULT_NAME}_bm_skx.json"
