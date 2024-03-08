# Change this vars if needed
export HF_DATASETS_CACHE="$DATA/huggingface/datasets"
export HF_HOME="$DATA/huggingface/hub"

export CUDA_VISIBLE_DEVICES="0,2,3"
export PYTHONPATH=$PWD

python $PWD/inference/run.py --output $HOME/repos/output --subset_size 0.01 \
    --seq_len 256 --model "8b" --batch_size 1 --num_workers 8