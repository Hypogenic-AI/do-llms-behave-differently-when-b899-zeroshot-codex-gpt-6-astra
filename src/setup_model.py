import os,json
os.environ['HF_HOME']=os.path.abspath('models/hf')
from huggingface_hub import snapshot_download
p=snapshot_download('Qwen/Qwen2.5-7B-Instruct',revision='a09a35458c702b33eeacc393d103063234e8bc28',allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
json.dump({'model':'Qwen/Qwen2.5-7B-Instruct','path':p,'revision':p.split('/')[-1]},open('results/model_manifest.json','w'),indent=2)
print('Model downloaded',flush=True)
