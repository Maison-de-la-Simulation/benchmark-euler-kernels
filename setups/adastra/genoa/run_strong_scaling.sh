#!/bin/bash
#SBATCH --account=cad16293
#SBATCH --job-name=scaling_genoa
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --threads-per-core=1
#SBATCH --hint=nomultithread
#SBATCH --constraint=GENOA

module purge
module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-cray

set -ex

cd "${SLURM_SUBMIT_DIR}" || exit

mkdir -p slurm_out results/scaling/genoa/

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DISPLAY_AFFINITY=TRUE

#-------- SETUP --------
SCALING=weak  # strong | weak
KERNEL=vector # scalar | vector

BASE_NX=256
BASE_NX_WEAK=32
NT=100

THREADS_LIST="1 2 4 8 16 32 64 128 144 160 176 184 192"

OUT="results/scaling/genoa/${SLURM_JOB_ID}_${SCALING}-genoa_${KERNEL}_FINAL.csv"

echo "mode,nx,nt,kernel,threads,time_s,mcells_s" >"$OUT"

# -------- RUN --------
for t in $THREADS_LIST; do
  export OMP_NUM_THREADS=$t

  if [[ "$SCALING" == "strong" ]]; then
    NX=$BASE_NX
  elif [[ "$SCALING" == "weak" ]]; then
    NX=$((BASE_NX_WEAK * t))
  else
    echo "Unknown SCALING=$SCALING"
    exit 1
  fi

  echo "Running ${SCALING^^} scaling | kernel=${KERNEL} | threads=${t} | nx=${NX}"

  srun ./build-genoa/simulations/euler_scaling \
    --mode "$SCALING" \
    --nx "$NX" \
    --nt "$NT" \
    --out "$OUT" \
    --kernel "$KERNEL"

  echo "Done | ${SCALING^^} | t=${t} | nx=${NX}"
done
