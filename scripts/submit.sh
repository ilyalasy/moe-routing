#!/bin/bash

#SBATCH -J MOE_RUN
#SBATCH -N 1                           #use -N only if you use both GPUs on the nodes, otherwise leave this line out
#SBATCH --partition zen2_0256_a40x2 # A100 - zen3_0512_a100x2 # A40 - zen2_0256_a40x2
#SBATCH --qos zen2_0256_a40x2     # A100 - zen3_0512_a100x2 # A40 - zen2_0256_a40x2
#SBATCH --gres=gpu:2

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate moe

# Change this vars if needed
export HF_DATASETS_CACHE="$DATA/huggingface/datasets"
export HF_HOME="$DATA/huggingface/hub"
accelerate launch --config_file accelerate_config.yaml token-routing.py --output $HOME/repos/output/experts.pt --subset_size 0.1 \
    --model "8b-1T" --batch_size 8 --num_workers 32