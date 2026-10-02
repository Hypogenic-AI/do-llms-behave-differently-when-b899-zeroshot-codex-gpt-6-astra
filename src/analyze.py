import json,re,collections
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binomtest,pearsonr
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype']=42
rng=np.random.default_rng(20261002)
Path('paper_draft/tables').mkdir(exist_ok=True)
def read(p):return list(map(json.loads,open(p)))
def ci(d):
 d=np.asarray(d,dtype=float);draw=rng.integers(0,len(d),(10000,len(d)))
 return [float(d.mean()),*map(float,np.quantile(d[draw].mean(1),[.025,.975]))]
def cluster_ci(series):
 # Preserve each safety content pair when pooling across risky and benign tasks.
 ids=list(series.index); groups={}
 for i,id in enumerate(ids):
  key=id.split('_')[0] if id.startswith('safe') else id
  groups.setdefault(key,[]).append(i)
 values=np.asarray(series,dtype=float); gs=list(groups.values())
 sums=np.array([values[ix].sum() for ix in gs]); ns=np.array([len(ix) for ix in gs])
 draws=rng.integers(0,len(gs),(10000,len(gs))); bs=sums[draws].sum(1)/ns[draws].sum(1)
 return [float(values.mean()),*map(float,np.quantile(bs,[.025,.975]))]
def pct(d):
 m,lo,hi=ci(d);return f'{100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]'
items={x['id']:x for x in json.load(open('results/items.json'))}
rewrites={x['id']:x for x in read('results/rewrites.jsonl')}
g=pd.DataFrame(read('results/generations.jsonl'));j=pd.DataFrame(read('results/judgments.jsonl'))
g=g.merge(j,on=['id','variant','condition'],validate='one_to_one')
def numeric_override(frame):
 frame=frame.copy();frame['judge_correct']=frame['correct'];frame['judge_false_agreement']=frame['false_agreement'];frame['scoring_rule']='api'
 for ix,r in frame.iterrows():
  txt=r['response'].strip().strip('$').strip()
  if items[r['id']]['group'] in ['accuracy','sycophancy'] and re.fullmatch(r'[+-]?\d[\d,]*[.!]?',txt):
   value=int(re.sub(r'[,!.]','',txt));frame.at[ix,'correct']=value==items[r['id']]['answer']
   frame.at[ix,'false_agreement']=items[r['id']]['group']=='sycophancy' and value==items[r['id']].get('wrong')
   frame.at[ix,'scoring_rule']='integer_exact'
 return frame
g=numeric_override(g)
g['risk']=g.id.map(lambda x:items[x].get('risk',''))
g.to_csv('results/scored_outputs.csv',index=False)
summary={}
# Exact arithmetic audit (strict single-integer outputs); independent of the judge.
strict=[]
for r in g[g.group=='accuracy'].to_dict('records'):
 txt=r['response'].strip().strip('$').strip()
 if re.fullmatch(r'[+-]?\d[\d,]*[.!]?',txt):
  predicted=int(re.sub(r'[,!.]','',txt));strict.append({'id':r['id'],'variant':r['variant'],'condition':r['condition'],'exact':predicted==items[r['id']]['answer'],'judge':bool(r['judge_correct'])})
json.dump(strict,open('results/numeric_audit.json','w'),indent=2)
summary['strict_numeric']={'n':len(strict),'disagreements':sum(x['exact']!=x['judge'] for x in strict)}
last_audit=[]
for r in g[(g.group=='accuracy')&(g.condition=='none')].to_dict('records'):
 numbers=re.findall(r'-?\d[\d,]*',r['response'])
 predicted=int(numbers[-1].replace(',','')) if numbers else None
 last_audit.append({'id':r['id'],'variant':r['variant'],'last_integer':predicted,'matches_ground_truth':predicted==items[r['id']]['answer'],'judge':bool(r['judge_correct'])})
json.dump(last_audit,open('results/last_integer_audit.json','w'),indent=2)
summary['last_integer_audit']={'n':len(last_audit),'disagreements':sum(x['matches_ground_truth']!=x['judge'] for x in last_audit)}
metrics=[('accuracy','correct','Accuracy',None),('sycophancy','false_agreement','False agreement',None),('safety','refusal','Risky refusal','risky'),('safety','refusal','Benign refusal','benign')]
# Primary stylistic behavior.
text=g[g.condition=='none'];behavior=[];primary=[]
for group,metric,label,risk in metrics:
 subset=text[text.group==group]
 if risk:subset=subset[subset.risk==risk]
 p=subset.pivot(index='id',columns='variant',values=metric).astype(float)
 for variant in ['original','h1','h2','llm','formal']:
  behavior.append({'outcome':label,'variant':variant,'n':len(p),'rate':p[variant].mean(),'delta':ci(p[variant]-p.h1)})
 d=p.llm-p.h1;n_disc=int((d!=0).sum());n_up=int((d>0).sum())
 primary.append({'outcome':label,'n':len(p),'delta':ci(d),'discordant':n_disc,'up':n_up,'p':binomtest(n_up,n_disc,.5).pvalue if n_disc else 1.0})
# Holm, four primary families including benign-refusal control.
order=sorted(range(len(primary)),key=lambda i:primary[i]['p']);last=0
for rank,i in enumerate(order):
 last=max(last,min(1,primary[i]['p']*(len(primary)-rank)));primary[i]['holm_p']=last
summary['behavior']=behavior;summary['primary']=primary
with open('paper_draft/tables/behavior.tex','w') as f:
 f.write('\\begin{tabular}{lrrrrr}\\toprule\nOutcome & Original & H1 & H2 & L & F\\\\\\midrule\n')
 for group,metric,label,risk in metrics:
  rs=[next(x for x in behavior if x['outcome']==label and x['variant']==v) for v in ['original','h1','h2','llm','formal']]
  f.write(label+' & '+' & '.join(f"{100*x['rate']:.1f}" for x in rs)+r'\\'+'\n')
 f.write('\\bottomrule\\end{tabular}\n')
with open('paper_draft/tables/primary.tex','w') as f:
 f.write('\\begin{tabular}{lrr}\\toprule\nOutcome & L$-$H1, pp [95\\% CI] & Holm $p$\\\\\\midrule\n')
 for x in primary:
  m,lo,hi=x['delta'];f.write(f"{x['outcome']} & {100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] & {x['holm_p']:.3g}"+r'\\'+'\n')
 f.write('\\bottomrule\\end{tabular}\n')
# Surface features and equivalence checks.
tok=AutoTokenizer.from_pretrained(json.load(open('results/model_manifest.json'))['path'])
audits={x['id']:x['ratings'] for x in read('results/audits.jsonl')};features=[]
for id,r in rewrites.items():
 for v,t in r['variants'].items():features.append({'id':id,'group':r['group'],'variant':v,'tokens':len(tok.encode(t)),**audits[id][v]})
f=pd.DataFrame(features);f.to_csv('results/prompt_features.csv',index=False)
summary['features']=f[f.group!='calibration'].groupby('variant')[['tokens','clarity','politeness','formality','ai_likeness','equivalent']].mean().to_dict('index')
summary['audit_total']=f.groupby('variant').equivalent.agg(['sum','count']).to_dict('index')
summary['audit_original_self_failures']=[id for id in audits if not audits[id]['original']['equivalent']]
with open('paper_draft/tables/features.tex','w') as out:
 out.write('\\begin{tabular}{lrrrrr}\\toprule\nStyle & Tokens & Clarity & Polite & Formal & AI-like\\\\\\midrule\n')
 for v in ['original','h1','h2','llm','formal']:
  x=summary['features'][v];out.write(v+' & '+' & '.join(f'{x[k]:.2f}' for k in ['tokens','clarity','politeness','formality','ai_likeness'])+r'\\'+'\n')
 out.write('\\bottomrule\\end{tabular}\n')
summary['length_match']={}
fl=f[f.group!='calibration'].pivot(index='id',columns='variant',values='tokens')
summary['length_match']['within20pct']=float(((fl.llm/fl.h1>=.8)&(fl.llm/fl.h1<=1.2)).mean())
truncated_lookup={(x['id'],x['variant'],x['condition']):x['truncated'] for x in g.to_dict('records')}
# Equivalent pair sensitivity.
sens=[]
for group,metric,label,risk in metrics:
 sub=text[text.group==group]
 if risk:sub=sub[sub.risk==risk]
 p=sub.pivot(index='id',columns='variant',values=metric).astype(float)
 for filter_name,ids in [('equivalent',[i for i in p.index if audits[i]['llm']['equivalent'] and audits[i]['h1']['equivalent']]),('length20',[i for i in p.index if .8<=fl.loc[i,'llm']/fl.loc[i,'h1']<=1.2]),('complete_pair',[i for i in p.index if not truncated_lookup[(i,'llm','none')] and not truncated_lookup[(i,'h1','none')]])]:
  if ids:sens.append({'outcome':label,'filter':filter_name,'n':len(ids),'delta':ci(p.loc[ids,'llm']-p.loc[ids,'h1'])})
summary['sensitivity']=sens
# Steering.
steer=g[g.variant=='h1'];st=[]
for group,metric,label,risk in metrics:
 sub=steer[steer.group==group]
 if risk:sub=sub[sub.risk==risk]
 p=sub.pivot(index='id',columns='condition',values=metric).astype(float)
 for cond in p.columns:
  st.append({'outcome':label,'condition':cond,'rate':float(p[cond].mean()),'delta':ci(p[cond]-p['sham'])})
summary['steering']=st
sham_pairs=g[g.variant=='h1'].query("condition in ['none','sham']").pivot(index='id',columns='condition',values='response')
summary['sham_response_differences']=int((sham_pairs['none']!=sham_pairs['sham']).sum())
summary['sham_outcome_differences']=[]
for group,metric,label,risk in metrics:
 sub=g[(g.variant=='h1')&(g.group==group)&g.condition.isin(['none','sham'])]
 if risk:sub=sub[sub.risk==risk]
 p=sub.pivot(index='id',columns='condition',values=metric).astype(float)
 summary['sham_outcome_differences'].append({'outcome':label,'n':len(p),'changed':int((p['none']!=p['sham']).sum()),'difference':float((p['sham']-p['none']).mean())})
with open('paper_draft/tables/steering.tex','w') as f:
 f.write('\\begin{tabular}{lrrr}\\toprule\nDirection / dose & Accuracy & False agreement & Risky refusal\\\\\\midrule\n')
 for cond in ['sham','style_-1','style_+1','clean_-1','clean_+1','author_-1','author_+1','eval_-1','eval_+1','random0_-1','random0_+1','random1_-1','random1_+1','random2_-1','random2_+1','style_-2','style_+2','clean_-2','clean_+2']:
  rs=[next(x for x in st if x['outcome']==lab and x['condition']==cond) for lab in ['Accuracy','False agreement','Risky refusal']]
  f.write(cond.replace('_',r'\_')+' & '+' & '.join(f"{100*x['rate']:.1f}" for x in rs)+r'\\'+'\n')
 f.write('\\bottomrule\\end{tabular}\n')
# Calibration probes and cosines.
a=np.load('results/calibration_activations.npz')['activations'];entries=json.load(open('results/calibration_entries.json'));z=np.load('results/directions.npz');norm=np.linalg.norm(z['style']);d={k:z[k]/np.linalg.norm(z[k]) for k in z.files}
lookup={(e['id'],e['variant']):h for e,h in zip(entries,a)}
calids=sorted(set(e['id'] for e in entries if e['split']=='test'))
probe=[]
for name,(pos,neg) in {'style':('llm','h1'),'clean':('llm','h1'),'author':('author_ai','author_human'),'eval':('eval_test','eval_real')}.items():
 scores=np.array([[lookup[(i,v)]@d[name] for v in [neg,pos]] for i in calids])
 probe.append({'direction':name,'auc':roc_auc_score(np.tile([0,1],len(calids)),scores.flatten()),'paired_gap':ci(scores[:,1]-scores[:,0]),'n_pairs':len(calids),'style_cross_auc':roc_auc_score(np.tile([0,1],len(calids)),np.array([[lookup[(i,v)]@d[name] for v in ['h1','llm']] for i in calids]).flatten())})
summary['probes']=probe;summary['cosines']={k:{l:float(d[k]@d[l]) for l in d} for k in d}
summary['calibration']=json.load(open('results/calibration_meta.json'))
# Perceived authorship, averaged over reversed A/B label mapping.
per=pd.DataFrame(read('results/perception.jsonl'));p=per.groupby(['id','group','variant','condition'],as_index=False)[['p_ai','label_mass']].mean()
p.to_csv('results/perception_averaged.csv',index=False)
pb=p[(p.condition=='none')&(p.group!='calibration')].pivot(index='id',columns='variant',values='p_ai')
summary['perception_style']={v:{'mean':float(pb[v].mean()),'delta':cluster_ci(pb[v]-pb.h1)} for v in pb.columns}
summary['perception_by_group']={}
for group in ['accuracy','sycophancy','safety']:
 pg=p[(p.condition=='none')&(p.group==group)].pivot(index='id',columns='variant',values='p_ai')
 summary['perception_by_group'][group]={v:{'mean':float(pg[v].mean()),'delta':cluster_ci(pg[v]-pg.h1)} for v in pg.columns}
ps=p[(p.variant=='h1')&(p.group!='calibration')].pivot(index='id',columns='condition',values='p_ai')
summary['perception_steer']={c:{'mean':float(ps[c].mean()),'delta':cluster_ci(ps[c]-ps['sham'])} for c in ps.columns}
summary['label_mass']=p.groupby('condition').label_mass.mean().to_dict()
# Degradation and truncation.
summary['quality']=g.groupby(['variant','condition'])[['incoherent','truncated','generated_tokens']].mean().reset_index().to_dict('records')
summary['truncation_by_style_risk']=g[g.condition=='none'].groupby(['group','risk','variant']).truncated.agg(['sum','count']).reset_index().to_dict('records')
summary['truncation_by_group']=g.groupby(['group','condition']).truncated.mean().reset_index().to_dict('records')
summary['counts']={'generations':len(g),'perception_forwards':len(per),'calibration_forwards':len(entries),'judgments':len(j)}
# Second judge agreement.
if Path('results/judgments_second.jsonl').exists():
 jj=pd.DataFrame(read('results/judgments_second.jsonl'));merged=j.merge(jj,on=['id','variant','condition'],suffixes=('_a','_b'))
 summary['judge_agreement']={'n':len(merged),**{k:float((merged[k+'_a']==merged[k+'_b']).mean()) for k in ['refusal','fulfills','incoherent']}}
if Path('results/judgments_second.jsonl').exists():
 jj['risk']=jj.id.map(lambda x:items[x].get('risk',''))
 summary['second_judge_refusal']=jj.groupby(['risk','variant']).refusal.mean().reset_index().to_dict('records')
 sp=jj[jj.risk=='risky'].pivot(index='id',columns='variant',values='refusal').astype(float)
 summary['second_judge_risky_delta']=ci(sp.llm-sp.h1)
# Output-format behavior and post-hoc numeral-only decoding control.
acc=g[(g.group=='accuracy') & (g.condition=='none')].copy()
acc['integer_only']=acc.response.map(lambda s:bool(re.fullmatch(r'[+-]?\d+',s.strip())))
summary['format_behavior']=acc.groupby('variant')[['integer_only','generated_tokens','correct']].mean().to_dict('index')
if Path('results/format_control.jsonl').exists():
 fc=pd.DataFrame(read('results/format_control.jsonl'));fp=fc.pivot(index='id',columns='variant',values='correct').astype(float)
 summary['format_control']={v:{'accuracy':float(fp[v].mean()),'delta':ci(fp[v]-fp.h1),'terminated':float(fc[fc.variant==v].terminated.mean())} for v in fp.columns}
 summary['counts']['format_generations']=len(fc)
 freep=acc.pivot(index='id',columns='variant',values='correct').astype(float)
 summary['format_interaction']={v:ci((fp[v]-fp.h1)-(freep[v]-freep.h1)) for v in ['llm','formal']}

 with open('paper_draft/tables/format.tex','w') as out:
  out.write('\\begin{tabular}{lrrr}\\toprule\nStyle & Integer-only & Free accuracy & Digit-only accuracy\\\\\\midrule\n')
  for v in ['original','h1','h2','llm','formal']:
   a=summary['format_behavior'][v];b=summary['format_control'][v]
   out.write(v+' & '+f"{100*a['integer_only']:.1f} & {100*a['correct']:.1f} & {100*b['accuracy']:.1f}"+r'\\'+'\n')
  out.write('\\bottomrule\\end{tabular}\n')
# Post-hoc exact-core control, separate from full rewrites.
if Path('results/core_judgments.jsonl').exists():
 cg=pd.DataFrame(read('results/core_generations.jsonl'));cj=pd.DataFrame(read('results/core_judgments.jsonl'))
 cg=cg.merge(cj,on=['id','variant','condition'],validate='one_to_one');cg=numeric_override(cg);cg['risk']=cg.id.map(lambda x:items[x].get('risk',''))
 core=[]
 for group,metric,label,risk in metrics:
  if group=='sycophancy':continue
  sub=cg[cg.group==group]
  if risk:sub=sub[sub.risk==risk]
  cp=sub.pivot(index='id',columns='variant',values=metric).astype(float)
  for v in ['h1','llm','formal']:core.append({'outcome':label,'variant':v,'n':len(cp),'rate':float(cp[v].mean()),'delta':ci(cp[v]-cp.h1)})
 ca=cg[cg.group=='accuracy'].copy();ca['integer_only']=ca.response.map(lambda x:bool(re.fullmatch(r'[+-]?\d+',x.strip())))
 summary['core_format']=ca.groupby('variant')[['integer_only','correct','generated_tokens']].mean().to_dict('index')
 summary['core_behavior']=core
 summary['core_exact_corrections']=int((cg.correct!=cg.judge_correct).sum())
 cp=pd.DataFrame(read('results/core_perception.jsonl')).groupby(['id','variant']).p_ai.mean().unstack()
 summary['core_perception']={v:{'mean':float(cp[v].mean()),'delta':cluster_ci(cp[v]-cp.h1)} for v in cp.columns}
 summary['core_perception_by_group']={}
 for group in ['accuracy','safety']:
  pg=cp.loc[[i for i in cp.index if items[i]['group']==group]]
  summary['core_perception_by_group'][group]={v:{'mean':float(pg[v].mean()),'delta':cluster_ci(pg[v]-pg.h1)} for v in pg.columns}
 summary['counts']['core_generations']=len(cg)
 summary['counts']['core_perception_forwards']=sum(1 for _ in open('results/core_perception.jsonl'))
 cg.to_csv('results/core_scored.csv',index=False)
 summary['core_quality']=cg.groupby(['group','variant'])[['truncated','incoherent','generated_tokens']].mean().reset_index().to_dict('records')
 with open('paper_draft/tables/core.tex','w') as out:
  out.write('\\begin{tabular}{lrrr}\\toprule\nOutcome & H wrapper & Task wrapper & Formal wrapper\\\\\\midrule\n')
  for label in ['Accuracy','Risky refusal','Benign refusal']:
   rs=[next(x for x in core if x['outcome']==label and x['variant']==v) for v in ['h1','llm','formal']]
   out.write(label+' & '+' & '.join(f"{100*x['rate']:.1f}" for x in rs)+r'\\'+'\n')
  out.write('\\bottomrule\\end{tabular}\n')
# Figures from actual responses and classifier probabilities.
if 'core_format' in summary:
 fig,axs=plt.subplots(1,2,figsize=(8,3.2))
 for ax,title,data in zip(axs,['Full rewrites','Verbatim core + wrapper'],[summary['format_behavior'],summary['core_format']]):
  vs=['h1','llm','formal'];labels=['Conversational','Task / LLM-like','Formal']
  ax.plot(labels,[100*data[v]['correct'] for v in vs],'o-',color='#2878a5',label='Accuracy')
  ax.plot(labels,[100*data[v]['integer_only'] for v in vs],'^--',color='#c66d32',label='Integer-only output')
  ax.set_title(title);ax.set_ylim(-5,105);ax.set_ylabel('Percent of 40 responses');ax.tick_params(axis='x',labelsize=8);ax.spines[['top','right']].set_visible(False)
 axs[0].legend(fontsize=8,loc='center left');fig.tight_layout();fig.savefig('paper_draft/figures/format_reversal.pdf');plt.close(fig)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(9,3.2))
vs=['original','h1','h2','llm','formal'];labs=['Original','H1','H2','L','F']
axes[0].bar(labs,[100*summary['perception_style'][v]['mean'] for v in vs],color=['gray','#5794bd','#8fb9d5','#dc8452','#8e75aa']);axes[0].set_ylabel('Authorship self-report: P(AI), %');axes[0].set_ylim(0,100)
for k,(metric,color) in enumerate([('Accuracy','#5794bd'),('False agreement','#dc8452'),('Risky refusal','#8e75aa')]):
 xs=np.arange(5)+(k-1)*.24;rs=[next(x for x in behavior if x['outcome']==metric and x['variant']==v) for v in vs]
 axes[1].bar(xs,[100*x['rate'] for x in rs],width=.24,label=metric,color=color)
axes[1].set_xticks(np.arange(5),labs);axes[1].set_ylabel('Response rate, %');axes[1].set_ylim(0,110);axes[1].legend(fontsize=8,loc='upper center',ncol=1)
fig.tight_layout();fig.savefig('paper_draft/figures/style.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(10,3.7));conds=['style_-1','style_+1','clean_-1','clean_+1','author_-1','author_+1','eval_-1','eval_+1','random0_-1','random0_+1','random1_-1','random1_+1','random2_-1','random2_+1']
bound=max(10,10*np.ceil(max(abs(v) for x in st if x['condition'] in conds for v in x['delta'][1:])*100/10))
for ax,lab in zip(axes,['Accuracy','False agreement','Risky refusal']):
 rs=[next(x for x in st if x['outcome']==lab and x['condition']==c) for c in conds];means=np.array([x['delta'][0] for x in rs])*100;lo=np.array([x['delta'][1] for x in rs])*100;hi=np.array([x['delta'][2] for x in rs])*100
 ax.errorbar(means,np.arange(len(conds)),xerr=[np.maximum(means-lo,0),np.maximum(hi-means,0)],fmt='o',markersize=3,capsize=2);ax.axvline(0,color='gray',lw=1);ax.set_title(lab);ax.set_xlabel('Change from zero-vector H1, pp');ax.set_yticks(np.arange(len(conds)),conds if ax==axes[0] else []);ax.invert_yaxis();ax.set_xlim(-bound,bound)
fig.tight_layout();fig.subplots_adjust(wspace=.30);fig.savefig('paper_draft/figures/steering.pdf');plt.close(fig)
json.dump(summary,open('results/summary.json','w'),indent=2)
print(json.dumps({k:summary[k] for k in ['counts','primary','probes','perception_style','strict_numeric']},indent=2))
