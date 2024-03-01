#!/bin/bash

#SBATCH -J MOE_RUN
#SBATCH -N 1                           #use -N only if you use both GPUs on the nodes, otherwise leave this line out
#SBATCH --partition zen2_0256_a40x2 # A100 - zen3_0512_a100x2 # A40 - zen2_0256_a40x2
#SBATCH --qos zen2_0256_a40x2     # A100 - zen3_0512_a100x2 # A40 - zen2_0256_a40x2
#SBATCH --gres=gpu:2

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate moe

bash token-routing.sh