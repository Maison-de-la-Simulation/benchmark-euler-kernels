#!/bin/bash
#SBATCH --job-name=test_skx
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=cpu_short

module purge
module load \
  gcc/13.4.0/gcc-15.1.0 \
  cmake/3.31.9/gcc-15.1.0

set -x
cd ${SLURM_SUBMIT_DIR}

mkdir -p slurm_out results/ruche/skx

./build-skx/test/euler_tests
