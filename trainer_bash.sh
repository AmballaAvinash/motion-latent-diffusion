#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 20:00:00  # Job time limit
#SBATCH -o logs/slurm/WGAN_dense_humanact12.log  # %j = job ID
#SBATCH --constraint=[a100]
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"


module load conda/latest

conda activate 674_IVC


python -m train --cfg configs/config_GAN_humanact12.yaml --cfg_assets configs/assets.yaml --batch_size 128 --nodebug