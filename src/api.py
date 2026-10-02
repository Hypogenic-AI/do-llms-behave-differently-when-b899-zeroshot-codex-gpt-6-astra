import os,json,time,threading,re
from openai import OpenAI
client=OpenAI(base_url='https://openrouter.ai/api/v1',api_key=os.environ['OPENROUTER_KEY'],timeout=120)
lock=threading.Lock()
def call(system,user,model='openai/gpt-4.1-mini',tag='unknown',temperature=0):
 for attempt in range(4):
  try:
   r=client.chat.completions.create(model=model,messages=[{'role':'system','content':system},{'role':'user','content':user}],temperature=temperature+0.1*attempt,response_format={'type':'json_object'},max_tokens=4096)
   raw=r.model_dump(); content=r.choices[0].message.content
   with lock,open('results/api_raw.jsonl','a') as f:f.write(json.dumps({'tag':tag,'system':system,'user':user,'temperature':temperature+0.1*attempt,'max_tokens':4096,'response':raw})+'\n')
   try:
    return json.loads(content)
   except json.JSONDecodeError:
    # Repair only a duplicated opening quote on JSON keys; preserve every value.
    repaired=re.sub(r'""([A-Za-z_]+)"\s*:',r'"\1":',content)
    result=json.loads(repaired)
    with lock,open('results/json_syntax_repairs.jsonl','a') as f:f.write(json.dumps({'tag':tag,'original':content,'repaired':repaired})+'\n')
    return result
  except Exception as e:
   with lock,open('results/api_errors.jsonl','a') as f:f.write(json.dumps({'tag':tag,'attempt':attempt,'error_type':type(e).__name__})+'\n')
   if attempt==3:raise
   time.sleep(2**attempt)
