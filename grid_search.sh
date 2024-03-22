#!/bin/sh
#SBATCH -N 1
#SBATCH -p a100
#SBATCH --time 6:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=he.1773@osu.edu
#SBATCH -o results/log

source /home/he.1773/.bashrc
source activate confMILE
echo $CONDA_PREFIX
echo $SLURM_JOB_ID

GPUS=$(srun hostname | tr '\n' ' ')
GPUS=${GPUS//".cluster"/""}
echo $GPUS

module load cuda/11.8
nvidia-smi
which nvidia-smi

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/he.1773/miniconda3-23.9.0/envs/mg-nov/lib
for EPOCH in 200 500 1000
do
  for LAMBDA in 0.1 0.3 0.5 0.7 0.9 0.99
  do
    for LR in 0.02 0.01 0.005 0.001
    do
      python main.py --jobid ${SLURM_JOB_ID} --lambda-fl ${LAMBDA} --learning-rate ${LR} --epoch ${EPOCH}
    done
  done
done