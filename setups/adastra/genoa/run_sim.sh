#!/bin/bash

#SBATCH --account=cad16293
#SBATCH --job-name=euler-simulations-genoa
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
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

mkdir -p slurm_out

export OMP_NUM_THREADS=32
export OMP_PROC_BIND=CLOSE

./build-genoa/simulations/euler_simulation
