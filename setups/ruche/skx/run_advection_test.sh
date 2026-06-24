#!/bin/bash

#SBATCH --job-name=advection_skx
#SBATCH --output=./slurm_out/advection_test/%x.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:45:00
#SBATCH --partition=cpu_short
#SBATCH --hint=nomultithread

module purge
module load \
  gcc/13.4.0/gcc-15.1.0 \
  cmake/3.31.9/gcc-15.1.0

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

export OMP_PROC_BIND=close

mkdir -p slurm_out/advection_test results/ruche/skx

./build-skx/tests/cosine_advection_test --gtest_filter="EulerVectorized.*"
