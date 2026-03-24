#!/bin/bash
#SBATCH --job-name=benchmark_a100
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=gpua100
#SBATCH --gres=gpu:1

module purge
module load \
  gcc/13.4.0/gcc-15.1.0 \
  cmake/3.31.9/gcc-15.1.0 \
  cuda/12.8.1/none-none

set -x
cd ${SLURM_SUBMIT_DIR}

# $1 = optional base name
RESULT_NAME=${1:-"unnamed"}
# $2 = optional benchmark filter regex, default to all benchmarks
BENCHMARK_FILTER=${2:-".*"}

echo "BENCHMARK_FILTER =" "$BENCHMARK_FILTER"
echo "RESULT_NAME =" "$RESULT_NAME"

mkdir -p slurm_out results/ruche/a100

./build-a100/benchmarks/euler_benchmarks \
  --benchmark_filter="$BENCHMARK_FILTER" \
  --benchmark_out_format=json \
  --benchmark_out="./results/ruche/a100/[${SLURM_JOB_ID}]_${RESULT_NAME}_bm_a100.json"
