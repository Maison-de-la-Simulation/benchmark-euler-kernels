#!/bin/bash
#SBATCH --account=cad16293
#SBATCH --job-name=bench_genoa
#SBATCH --output=./slurm_out/%x.o%j
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
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

mkdir -p slurm_out results/adastra/genoa/mt/
BENCHMARK_FILTER=${1:-""}

export OMP_NUM_THREADS=1
# export OMP_PROC_BIND=CLOSE
# export OMP_PLACES=THREADS

# export OMP_DISPLAY_AFFINITY=TRUE
# export OMP_AFFINITY_FORMAT="thread %0.3n -> cpu %A"
# numactl -H

# srun bash -c 'echo $SLURM_CPUS_PER_TASK; grep Cpus_allowed_list /proc/self/status'

./build-genoa/benchmarks/euler_benchmarks \
  --benchmark_filter="${BENCHMARK_FILTER}" \
  --benchmark_out_format=json \
  --benchmark_out=./results/adastra/genoa/bm_json/"[${SLURM_JOB_ID}]_BASE_${BENCHMARK_FILTER}.json"
# --benchmark_out=./results/adastra/genoa/bm_json/mt/"[${SLURM_JOB_ID}]_T${OMP_NUM_THREADS}-${OMP_PROC_BIND}-${OMP_PLACES}_${BENCHMARK_FILTER}.json"
