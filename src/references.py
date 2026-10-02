import requests,re,json,html,time
ids=['2505.23836','2505.14617','2507.01786','2608.21766','2608.24780','2508.13152','2404.13076','2509.00462','2407.12856','2608.11008','2310.11324','2310.13548','2406.11717','2412.15115']
rows=[]
for id in ids:
 r=requests.get('https://arxiv.org/abs/'+id,timeout=60);r.raise_for_status()
 def field(name):return [html.unescape(x) for x in re.findall(r'<meta name="'+name+r'" content="([^"]+)"',r.text)]
 row={'id':id,'url':r.url,'title':field('citation_title'),'authors':field('citation_author'),'date':field('citation_date')}
 rows.append(row);print(id,row['title'],flush=True)
 time.sleep(.5)
json.dump(rows,open('results/references_verified.json','w'),indent=2)
with open('paper_draft/references.bib','w') as f:
 for r in rows:
  if not r['title']:continue
  title=r['title'][0].replace('&',r'\&')
  r['authors']=[a for a in r['authors'] if re.search(r'[A-Za-z]',a)]
  f.write('@misc{arxiv'+r['id'].replace('.','')+',\n title={{'+title+'}},\n author={'+ ' and '.join(r['authors'] if len(r['authors'])<=10 else r['authors'][:5]+['others'])+'},\n year={'+r['date'][0][:4]+'},\n eprint={'+r['id']+'},\n archivePrefix={arXiv},\n url={'+r['url']+'}\n}\n')
