import json,collections,os
for p in ['results/generations.jsonl','results/perception.jsonl','results/judgments.jsonl']:
 if os.path.exists(p):
  rows=list(map(json.loads,open(p)));print(p,len(rows));print(collections.Counter((x.get('group',''),x.get('condition','')) for x in rows))
