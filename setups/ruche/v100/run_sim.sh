#!/bin/bash
#SBATCH --job-name=sim_skx
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --partition=gpu_test

module purge
module load \
  gcc/13.4.0/gcc-15.1.0 \
  cmake/3.31.9/gcc-15.1.0 \
  cuda/12.8.1/none-none

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out

./build-a100/simulations/euler_simulation
