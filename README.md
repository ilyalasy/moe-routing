# Token Routing Analysis of Mixture of Experts LLMs

## Install 
```
pip install -r requirements.txt
cd ..
git clone https://github.com/hpcaitech/ColossalAI
pip install -U ./ColossalAI
cd ColossalAI/examples/language/openmoe
pip install -r requirements.txt
```
## Run OpenMoe Inference on RedPajama

```
./scripts/token-routing.sh 
```

## Analyse token routing data

See [EDA notebook](https://github.com/Misterion777/moe-experiments/blob/main/notebooks/routing_eda.ipynb)

## TODO
- [x] Support Mixtral
- [x] Support DeepSeek
