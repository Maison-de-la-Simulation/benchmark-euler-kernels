#!/bin/bash

#SBATCH --account=gen2224
#SBATCH --job-name=benchmarks-euler
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --constraint=GENOA
#SBATCH --time=05:59:00
#SBATCH --exclusive

# To compute in the submission directory
cd ${SLURM_SUBMIT_DIR}

module purge

module load \
       gcc-native/13.2

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_DISPLAY_AFFINITY=true

export OMP_NUM_THREADS=8
srun ./build-genoa/euler_benchmarks --kokkos-print-configuration
