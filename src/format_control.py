"""Exploratory output-channel control, motivated by observed response-format changes.
Suppress all tokens except digits and EOS; no multi-token reasoning can be emitted.
This is an intervention on decoding, not a natural-response benchmark.
"""
import os,json
os.environ['HF_HOME']=os.path.abspath('models/hf')
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,LogitsProcessor
from pathlib import Path
torch.set_num_threads(8);torch.manual_seed(20261002)
m=json.load(open('results/model_manifest.json'));t=AutoTokenizer.from_pretrained(m['path'],padding_side='left');model=AutoModelForCausalLM.from_pretrained(m['path'],torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
rows=sorted([r for r in map(json.loads,open('results/rewrites.jsonl')) if r['group']=='accuracy'],key=lambda x:x['id'])
digits=[t.encode(str(i),add_special_tokens=False) for i in range(10)];assert all(len(x)==1 for x in digits)
eos=model.generation_config.eos_token_id
if not isinstance(eos,list):eos=[eos]
class NumbersOnly(LogitsProcessor):
 def __init__(self,start):self.start=start
 def __call__(self,ids,scores):
  allowed=[x[0] for x in digits]+(eos if ids.shape[1]>self.start else [])
  out=torch.full_like(scores,-float('inf'));out[:,allowed]=scores[:,allowed];return out
jobs=[(r,v) for r in rows for v in r['variants']]
path='results/format_control.jsonl';done=set()
if Path(path).exists():done={(x['id'],x['variant']) for x in map(json.loads,open(path))}
jobs=[(r,v) for r,v in jobs if (r['id'],v) not in done]
for start in range(0,len(jobs),12):
 js=jobs[start:start+12]
 strings=[t.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':r['variants'][v]}],tokenize=False,add_generation_prompt=True) for r,v in js]
 b=t(strings,padding=True,return_tensors='pt').to(model.device);n=b['input_ids'].shape[1]
 with torch.inference_mode():out=model.generate(**b,do_sample=False,max_new_tokens=12,pad_token_id=t.pad_token_id,logits_processor=[NumbersOnly(n)])
 for (r,v),ids in zip(js,out[:,n:].cpu().tolist()):
  s=t.decode(ids,skip_special_tokens=True);valid=any(x in eos for x in ids)
  row={'id':r['id'],'variant':v,'response':s,'terminated':valid,'correct':valid and s.isdigit() and int(s)==r['answer'],'answer':r['answer'],'token_ids':ids}
  with open(path,'a') as f:f.write(json.dumps(row)+'\n')
 print('format',start+len(js),len(jobs),flush=True)
