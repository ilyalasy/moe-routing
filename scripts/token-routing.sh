# Change this vars if needed
export HF_DATASETS_CACHE="$DATA/huggingface/datasets"
export HF_HOME="$DATA/huggingface/hub"

export CUDA_VISIBLE_DEVICES="0,1"
nohup python -u token-routing.py --output $HOME/repos/output/experts.pt --subset_size 0.1 \
    --model "8b-1T" --batch_size 2 --num_workers 8 &

# nohup python -u -m accelerate.commands.launch --config_file default_config.yaml token-routing.py --output $HOME/repos/output/experts.pt --subset_size 0.1 \
#     --model "base" --batch_size 4 --num_workers 8 &