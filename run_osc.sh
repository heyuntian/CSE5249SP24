#!/bin/sh
#SBATCH --account=PAS2030
#SBATCH -p XXX
#SBATCH --nodes=1 --ntasks-per-node=48
#SBATCH --time 2-00:00:00
#SBATCH --job-name=grid-search

source ~/.bashrc
export PYTHONNOUSERSITE=true
source activate XXX
echo $CONDA_PREFIX
echo $SLURM_JOB_ID

GPUS=$(srun hostname | tr '\n' ' ')
GPUS=${GPUS//".cluster"/""}
echo $GPUS

#module load cuda/11.8
#nvidia-smi
#which nvidia-smi

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/he.1773/miniconda3-23.9.0/envs/mg-nov/lib
DATA=citeseer
EPOCH=1000

for CR_LEVEL in 1 2 3
do
  for LAMBDA in 0.1 0.3 0.5 0.7 0.9 0.99
  do
    for LR in 0.02 0.01 0.005 0.001
    do
      for SEED in 0 1 2 3 4
      do
        python main.py --jobid ${SLURM_JOB_ID} --data ${DATA} --coarsen-level ${CR_LEVEL} --lambda-fl ${LAMBDA} --learning-rate ${LR} --epoch ${EPOCH} --seed ${SEED}
      done
    done
  done
done