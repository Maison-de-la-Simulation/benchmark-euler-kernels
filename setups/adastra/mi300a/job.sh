#!/bin/bash

#SBATCH --account=gen2224
#SBATCH --job-name=benchmarks-euler
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --constraint=MI300
#SBATCH --time=05:59:00

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

module purge

module load \
       gcc-native/13.2 \
       rocm/6.3.3

export HSA_XNACK=1

srun ./build-mi300a/euler_benchmarks --kokkos-print-configuration
