#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --constraint=[a100]
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"


module load miniconda/22.11.1-1

conda activate 674_IVC

python -m train --cfg configs/config_vae_kitml.yaml --cfg_assets configs/assets.yaml --batch_size 64 --nodebug