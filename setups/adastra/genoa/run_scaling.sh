#!/bin/bash
#SBATCH --account=cad16293
#SBATCH --job-name=strong_scaling_genoa
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:08:00
#SBATCH --exclusive
#SBATCH --constraint=GENOA

module purge
module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-cray

set -x
cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/scaling/

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

export OMP_DISPLAY_AFFINITY=TRUE

OUT=results/scaling/strong_genoa16-32.csv

echo "mode,nx,nt,threads,time_s,mcells_s" >$OUT

for t in 1 2 4 8 16 32; do
  export OMP_NUM_THREADS=$t

  echo "Starting $t"
  srun ./build-genoa/simulations/euler_scaling \
    --mode strong \
    --nx 256 \
    --nt 100 \
    --warmup 0 \
    --repeats 1 \
    --out $OUT
  echo "Finished $t"
done
