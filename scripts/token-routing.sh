# # Install ColossalAI
# git clone --branch my_openmoe https://github.com/Orion-Zheng/ColossalAI.git
# pip install ./ColossalAI
# python -m pip install -r ./ColossalAI/examples/language/openmoe/requirements.txt

# Change this vars if needed
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
export HF_HOME="$HOME/.cache/huggingface/hub"
python token-routing.py