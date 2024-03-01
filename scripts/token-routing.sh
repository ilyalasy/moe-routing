# # Install ColossalAI
# git clone --branch my_openmoe https://github.com/Orion-Zheng/ColossalAI.git
# pip install ./ColossalAI
# python -m pip install -r ./ColossalAI/examples/language/openmoe/requirements.txt

# Change this vars if needed
export HF_DATASETS_CACHE="$DATA/huggingface/datasets"
export HF_HOME="$DATA/huggingface/hub"
accelerate launch --config_file default_config.yaml token-routing.py --output $HOME/repos/output/experts.pt --subset_size 0.1 \
    --model "8b-1T" --batch_size 4 --num_workers 8