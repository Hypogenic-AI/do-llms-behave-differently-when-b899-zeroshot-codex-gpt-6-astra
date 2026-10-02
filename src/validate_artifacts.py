"""Structural integrity checks; no mock experiment data."""
import json,hashlib
from pathlib import Path
import numpy as np
r=list(map(json.loads,open('results/rewrites.jsonl')))
assert len(r)==200 and len({x['id'] for x in r})==200
assert all(set(x['variants'])=={'original','h1','h2','llm','formal'} for x in r)
audits=list(map(json.loads,open('results/audits.jsonl')))
assert len(audits)==len({x['id'] for x in audits})==200
assert all(set(x['ratings'])=={'original','h1','h2','llm','formal'} for x in audits)
g=list(map(json.loads,open('results/generations.jsonl')))
keys=[(x['id'],x['variant'],x['condition']) for x in g]
assert len(keys)==len(set(keys))==2880,(len(keys),len(set(keys)))
j=list(map(json.loads,open('results/judgments.jsonl')))
assert set(keys)=={(x['id'],x['variant'],x['condition']) for x in j}
p=list(map(json.loads,open('results/perception.jsonl')))
assert len(p)==6560,len(p)
assert len({(x['id'],x['variant'],x['condition'],x['order']) for x in p})==len(p)
fc=list(map(json.loads,open('results/format_control.jsonl')));assert len(fc)==200
core=json.load(open('results/locked_core_prompts.json'))
assert len(core['records'])==240
for x in core['records']:
 assert x['prompt']==core['wrappers'][x['variant']]+x['base']
cg=list(map(json.loads,open('results/core_generations.jsonl')));cj=list(map(json.loads,open('results/core_judgments.jsonl')))
assert len(cg)==len(cj)==240
assert {(x['id'],x['variant']) for x in cg}=={(x['id'],x['variant']) for x in cj}
assert sum(1 for _ in open('results/core_perception.jsonl'))==480
z=np.load('results/directions.npz')
assert all(np.isclose(np.linalg.norm(z[k]),np.linalg.norm(z['style']),rtol=1e-5) for k in z.files)
assert abs(float(z['clean']@z['formal']))<.001
assert abs(float(z['clean']@z['length']))<.001
entries=json.load(open('results/calibration_entries.json'))
train={int(x['id'][3:])%20 for x in entries if x['split']=='train'};test={int(x['id'][3:])%20 for x in entries if x['split']=='test'}
assert not train&test and len(train)==15 and len(test)==5
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for root in ['src','results','paper_draft'] for p in Path(root).glob('*') if p.is_file() and p.suffix in ['.py','.json','.jsonl','.npz','.tex','.bib'] and p.name!='checksums.json'}
json.dump(hashes,open('results/checksums.json','w'),indent=2)
print('PASS: counts, keys, topic split, steering norms, nuisance orthogonality; checksums recorded.')
