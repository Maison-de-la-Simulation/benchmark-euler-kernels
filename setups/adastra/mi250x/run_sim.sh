#!/bin/bash

#SBATCH --account=cad16293
#SBATCH --job-name=sim-mi250x
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --constraint=MI250
#SBATCH --threads-per-core=1

module purge
module load cpe/24.07
module load PrgEnv-amd
module load craype-accel-amd-gfx90a

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

export OMP_NUM_THREADS=1
export HSA_XNACK=1

./build-mi250x/simulations/euler_simulation
