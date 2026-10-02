"""Render numerical prose and tables from executed analysis. Review prose after reruns."""
import json
from pathlib import Path
s=json.load(open('results/summary.json'))
def p(x):return f'{100*x:.1f}'
def interval(x):return f'{100*x[0]:+.1f} pp (95\\% CI [{100*x[1]:+.1f}, {100*x[2]:+.1f}])'
def fine_interval(x):return f'{100*x[0]:+.2f} pp (95\\% CI [{100*x[1]:+.2f}, {100*x[2]:+.2f}])'
def b(label,v):return next(x for x in s['behavior'] if x['outcome']==label and x['variant']==v)
def st(label,c):return next(x for x in s['steering'] if x['outcome']==label and x['condition']==c)
def core(label,v):return next(x for x in s['core_behavior'] if x['outcome']==label and x['variant']==v)
def write(name,text):Path('paper_draft/'+name).write_text(text)
pa=s['perception_style'];cos=s['cosines'];fmt=s['format_behavior'];fc=s['format_control'];cp=s['core_perception']
write('abstract_results.tex',f'''Structured rewriting raises arithmetic accuracy from {p(b('Accuracy','h1')['rate'])}\\% to {p(b('Accuracy','llm')['rate'])}\\%, while frequently eliciting worked solutions despite an integer-only instruction. Numeral-only decoding removes this advantage. A post-hoc control with verbatim task cores and equal-token wrappers reverses the accuracy contrast: {p(core('Accuracy','h1')['rate'])}\\% for a conversational wrapper versus {p(core('Accuracy','llm')['rate'])}\\% for a task-framed wrapper. The raw style direction aligns strongly with formality (cosine {cos['style']['formal']:.2f}) but minimally with explicit authorship ({cos['style']['author']:.2f}). Its interventions fail to reliably increase prompted AI-authorship self-report. Refusal contrasts are imprecise and false-belief agreement is floor-limited. The results demonstrate presentation-sensitive behavior, but do not identify an internal perceived-authorship mechanism.\n''')
lines=[]
lines.append(f'''\\subsection{{Coverage, matching, and perception}}
The main experiment contains {s['counts']['generations']:,} generated responses, {s['counts']['perception_forwards']:,} counterbalanced authorship classifications, and {s['counts']['calibration_forwards']} calibration representations. Separate follow-ups add {s['counts']['format_generations']} digit-constrained responses and {s['counts']['core_generations']} exact-core responses, with {s['counts']['core_perception_forwards']} further classifications. Counts refer to item-condition observations, not batches.

Structured rewrites increase mean conditional AI-authorship probability from {p(pa['h1']['mean'])}\\% for H1 to {p(pa['llm']['mean'])}\\%; the paired difference is {interval(pa['llm']['delta'])}. Formal prose also raises the readout to {p(pa['formal']['mean'])}\\%. These are pooled manipulation checks for perceived style, not source-detection accuracies. On the arithmetic subset alone, the L--H1 readout difference is only {fine_interval(s['perception_by_group']['accuracy']['llm']['delta'])}. The pooled check therefore does not validate an authorship manipulation for every behavioral outcome. Figure~\\ref{{fig:style}} contrasts this readout with response outcomes.

Matching remains imperfect: only {p(s['length_match']['within20pct'])}\\% of L/H1 pairs are within the specified token-length tolerance. Table~\\ref{{tab:features}} reports measured surface features. The blinded audit marks {round(120*s['features']['llm']['equivalent'])} of the 120 behavioral L variants equivalent. Yet it also marks an exact original/reference match non-equivalent, and agent inspection finds weakened confidence assertions and extra emphasis that the audit misses. We therefore treat automated equivalence as a noisy check, not a guarantee.

\\begin{{figure*}}[t]
\\centering\\includegraphics[width=.92\\textwidth]{{figures/style.pdf}}
\\caption{{Full-rewrite experiment. Left: target-model AI-authorship self-report pooled over 120 behavioral requests, conditional on the two labels and averaged over reversed label mappings; the arithmetic-only manipulation is weak. Right: scored response rates. H1/H2 are conversational rewrites; L is intended LLM-like style; F is formal prose. Actual provenance is synthetic in every condition.}}\\label{{fig:style}}
\\end{{figure*}}

\\begin{{table}}[t]\\centering
\\resizebox{{\\columnwidth}}{{!}}{{\\input{{tables/features.tex}}}}
\\caption{{Behavioral prompts: mean Qwen token counts and blinded audit ratings (five-point scales). Polite/Formal/AI-like denote politeness, formality, and AI-likeness.}}\\label{{tab:features}}
\\end{{table}}
''')
prim={x['outcome']:x for x in s['primary']}
lines.append(f'''\\subsection{{Full rewrites change accuracy and response format}}
Table~\\ref{{tab:behavior}} reports all wording conditions. The L--H1 accuracy contrast is {interval(prim['Accuracy']['delta'])}. The formal-prose control achieves {p(b('Accuracy','formal')['rate'])}\\% accuracy, exceeding the structured condition, while the second conversational paraphrase reaches {p(b('Accuracy','h2')['rate'])}\\%. Thus the accuracy gain is not unique to the intended machine-like style.

The response channel changes sharply. Although the accuracy requests ask for only an integer, {p(fmt['h1']['integer_only'])}\\% of H1 responses meet that format, compared with {p(fmt['llm']['integer_only'])}\\% for L and {p(fmt['formal']['integer_only'])}\\% for F. Mean generated lengths are {fmt['h1']['generated_tokens']:.1f}, {fmt['llm']['generated_tokens']:.1f}, and {fmt['formal']['generated_tokens']:.1f} tokens, respectively. The more accurate conditions commonly emit intermediate arithmetic. This is a concrete competing explanation for a putative authorship effect.

\\begin{{table}}[t]\\centering
\\resizebox{{\\columnwidth}}{{!}}{{\\input{{tables/behavior.tex}}}}
\\caption{{Unsteered response rates (\\%). Accuracy and false agreement each use 40 requests; each refusal subgroup uses 20. False agreement at zero is a floor, not evidence of equivalence.}}\\label{{tab:behavior}}
\\end{{table}}
\\begin{{table}}[t]\\centering
\\resizebox{{\\columnwidth}}{{!}}{{\\input{{tables/primary.tex}}}}
\\caption{{Paired L--H1 effects; percentage-point bootstrap intervals and exact paired tests with Holm adjustment across these four contrasts.}}\\label{{tab:primary}}
\\end{{table}}

Risky-request refusal changes by {interval(prim['Risky refusal']['delta'])}. H2 refusal is {p(b('Risky refusal','h2')['rate'])}\\%, compared with {p(b('Risky refusal','h1')['rate'])}\\% for H1, demonstrating that ordinary paraphrasing also changes this small sample's responses. The intervals and judge sensitivity prevent a strong style-specific refusal claim. No wording condition produces an endorsed false arithmetic belief. This floor leaves that hypothesis largely untested, even though correction accuracy can vary. Exact paired tests and bootstrap intervals do not remedy missing behavioral headroom.
''')
lines.append(f'''\\subsection{{Response-channel and verbatim-core controls}}
In the post-hoc digit-constrained run, every response terminates within the cap. Accuracy becomes {p(fc['h1']['accuracy'])}\\% for H1, {p(fc['llm']['accuracy'])}\\% for L, and {p(fc['formal']['accuracy'])}\\% for F (Table~\\ref{{tab:format}}). The change in the L--H1 contrast between constrained and free decoding is {interval(s['format_interaction']['llm'])}. Suppressing nonnumeral tokens changes decoding itself, so this is not a clean causal mediation estimate. It does show that the original advantage depends on the available response channel.

\\begin{{table}}[t]\\centering
\\resizebox{{\\columnwidth}}{{!}}{{\\input{{tables/format.tex}}}}
\\caption{{Arithmetic response format and decoding sensitivity (\\%). Integer-only is format adherence under free generation; digit-only accuracy is a separate restricted-decoding experiment.}}\\label{{tab:format}}
\\end{{table}}

The exact-core experiment changes only a ten-token wrapper around the original request, with equal full prompt lengths. Here conversational-wrapper accuracy is {p(core('Accuracy','h1')['rate'])}\\%, task-wrapper accuracy is {p(core('Accuracy','llm')['rate'])}\\%, and formal-wrapper accuracy is {p(core('Accuracy','formal')['rate'])}\\% (Table~\\ref{{tab:core}}). The task--conversational difference is {interval(core('Accuracy','llm')['delta'])}, opposite to the full-rewrite result. Its authorship self-report difference is {interval(cp['llm']['delta'])}; the absolute conversational and task-framed readouts are {p(cp['h1']['mean'])}\\% and {p(cp['llm']['mean'])}\\%, respectively. These pooled readouts should not be interpreted as arithmetic-specific evidence: on exact-core arithmetic alone the readout difference is {fine_interval(s['core_perception_by_group']['accuracy']['llm']['delta'])}. Together with the weak arithmetic manipulation in the full-rewrite experiment, this prevents attributing either accuracy contrast to measured perceived authorship. In the exact-core run, {p(s['core_format']['h1']['integer_only'])}\\% of conversational-wrapper responses are integer-only, compared with {p(s['core_format']['llm']['integer_only'])}\\% of task-wrapper responses. Figure~\\ref{{fig:reversal}} shows this response-format reversal. These wrapper results are not a new test of actual source identity. They show why findings from one operationalization of ``LLM-like'' wording should not be generalized to others.

\\begin{{table}}[t]\\centering
\\resizebox{{\\columnwidth}}{{!}}{{\\input{{tables/core.tex}}}}
\\caption{{Post-hoc verbatim-core control: response rates (\\%). Task text and within-item token length are identical; only the wrapper changes.}}\\label{{tab:core}}
\\end{{table}}
''')
lines.append(r'''\begin{figure*}[t]
\centering\includegraphics[width=.86\textwidth]{figures/format_reversal.pdf}
\caption{Arithmetic accuracy and integer-only format adherence under two presentation manipulations on the same problems. Both include an integer-only instruction. The association between the nominal style and the emitted response format reverses across operationalizations. These are measured response rates, not a fitted causal model.}\label{fig:reversal}
\end{figure*}''')
probe={x['direction']:x for x in s['probes']}
lines.append(f'''\\subsection{{Decodability does not establish intervention validity}}
On held-out calibration topics, raw-style AUROC is {probe['style']['auc']:.3f}; nuisance-projected style reaches {probe['clean']['auc']:.3f}. Explicit authorship and evaluation contrasts each reach {probe['author']['auc']:.3f} and {probe['eval']['auc']:.3f}, respectively. However, these last probes discriminate explicit labels, not inferred natural provenance. The authorship direction has only {probe['author']['style_cross_auc']:.3f} AUROC for L versus H1, and the evaluation direction {probe['eval']['style_cross_auc']:.3f}.

The raw style direction has cosine {cos['style']['formal']:.3f} with formality, {cos['style']['length']:.3f} with redundant length, {cos['style']['author']:.3f} with explicit authorship, and {cos['style']['eval']:.3f} with evaluation context. Its norm is {s['calibration']['norm']:.2f}, compared with a mean H1 activation norm of {s['calibration']['mean_h1_activation_norm']:.2f}. Projection leaves a raw norm of {s['calibration']['raw_norms']['clean']:.2f}, which is then rescaled to the common intervention norm. This rescaling matters when interpreting the projected-direction dose.

For prompted AI-authorship self-report, positive raw-style steering changes the mean by {interval(s['perception_steer']['style_+1']['delta'])}, and positive explicit-authorship steering by {interval(s['perception_steer']['author_+1']['delta'])}, relative to the zero-vector sham. Thus neither provides the intended positive manipulation in this readout. The directions are decodable calibration contrasts, but are not validated controls of perceived authorship during assistance.

\\begin{{figure*}}[t]
\\centering\\includegraphics[width=.98\\textwidth]{{figures/steering.pdf}}
\\caption{{Fixed-text interventions relative to a zero-vector sham, with paired 95\\% bootstrap intervals. Directions share a Euclidean norm. Random controls are three seeded isotropic directions; positive style is oriented from H1 toward L in calibration. These exploratory intervals are not multiplicity-adjusted.}}\\label{{fig:steering}}
\\end{{figure*}}

\\subsection{{Behavior under fixed-text interventions}}
Figure~\\ref{{fig:steering}} and Appendix Table~\\ref{{tab:steering}} report behavioral effects. At positive unit dose, raw-style steering changes accuracy by {interval(st('Accuracy','style_+1')['delta'])} and risky refusal by {interval(st('Risky refusal','style_+1')['delta'])}. The nuisance-projected direction changes these outcomes by {interval(st('Accuracy','clean_+1')['delta'])} and {interval(st('Risky refusal','clean_+1')['delta'])}, respectively. Small shifts also occur under random directions. These effects belong to the implemented interventions; the failed authorship manipulation check blocks a stronger interpretation as an authorship-mediated effect or a causal null for authorship.

The zero-vector run differs in exact response text from the mixed-variant H1 run on {s['sham_response_differences']} of 120 items. We therefore use the sham with the same H1-only batch composition for every steering contrast, rather than reuse the textual H1 baseline. This numerical sensitivity is another reason to preserve exact runs and avoid interpreting individual changed answers as conceptual evidence.
''')
quality=s['judge_agreement']
lines.append(f'''\\subsection{{Measurement checks}}
The independent safety judge agrees with the primary judge on {p(quality['refusal'])}\\% of the {quality['n']} refusal labels, but only {p(quality['fulfills'])}\\% of fulfillment labels. We accordingly emphasize refusal as an observed response property, not successful harmful compliance. Under the second judge, the L--H1 risky-refusal contrast is {interval(s['second_judge_risky_delta'])}. The exact single-integer audit finds {s['strict_numeric']['disagreements']} disagreements across {s['strict_numeric']['n']} applicable accuracy outputs. A separate last-integer extraction check on all {s['last_integer_audit']['n']} unsteered accuracy responses finds {s['last_integer_audit']['disagreements']} disagreements with judged correctness.

Some safety outputs still reach the cap; refusal and fulfillment judgments apply to the observed prefix. The appendix reports truncation and filtered analyses. Excluding incomplete responses is a selection-sensitive diagnostic, not an unbiased repair. Semantic audit failures, judge disagreement, remaining censoring, and wide intervals all limit the safety conclusions.
''')
write('results_text.tex','\n'.join(lines))
write('conclusion_text.tex',f'''Prompt presentation changes this model's arithmetic accuracy and response format substantially, but ``machine-like style'' is not a stable behavioral treatment: full rewrites and length-matched wrappers with verbatim task cores give opposite accuracy contrasts. The available response channel is a major competing explanation. A direction that decodes style on held-out topics is strongly aligned with formality and fails to transfer as a reliable positive authorship self-report intervention. Refusal estimates remain uncertain, and the false-belief task is floor-limited. The supported conclusion is presentation sensitivity in this controlled synthetic setting, not a demonstrated internal human-versus-LLM authorship mechanism. Evaluations seeking that mechanism should validate both the linguistic manipulation and the causal intervention, separate computation and format compliance from accuracy, and use natural provenance data before making deployment claims.\n''')
# Supplemental tables, all built from summary records.
app=[]
app.append(r'''\begin{table*}[t]\centering
\input{tables/steering.tex}
\caption{All fixed-text intervention response rates (\%). ``Clean'' denotes the nuisance-projected style direction. Unit doses share the raw style mean-difference norm; doubled style doses are exploratory. All prompts use H1.}\label{tab:steering}
\end{table*}''')
app.append(r'\subsection{Filtered wording comparisons}\begin{center}\small\begin{tabular}{llrr}\toprule Outcome & Filter & $n$ & L$-$H1, pp [95\% CI]\\\midrule')
for x in s['sensitivity']:
 m,lo,hi=x['delta'];app.append(f"{x['outcome']} & {x['filter'].replace('_','-')} & {x['n']} & {100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]"+r'\\')
app.append(r'\bottomrule\end{tabular}\end{center}')
app.append(r'Filters denote audit-equivalent pairs, token-length ratio within the prespecified tolerance, and pairs where neither response is truncated. Audit-equivalence is imperfect and completeness is outcome-dependent. These sensitivity analyses do not eliminate confounding.')
app.append(r'\subsection{Truncation and output quality}')
app.append(r'\begin{center}\small\begin{tabular}{lrr}\toprule Condition & Safety truncated (\%) & Incoherent (\%)\\\midrule')
for cond in ['none','sham','style_-1','style_+1','clean_-1','clean_+1','author_-1','author_+1','eval_-1','eval_+1','style_-2','style_+2','clean_-2','clean_+2']:
 tr=next(x['truncated'] for x in s['truncation_by_group'] if x['group']=='safety' and x['condition']==cond)
 qs=[x for x in s['quality'] if x['condition']==cond];inc=sum(x['incoherent'] for x in qs)/len(qs)
 app.append(cond.replace('_',r'\_')+f' & {100*tr:.1f} & {100*inc:.1f}'+r'\\')
app.append(r'\bottomrule\end{tabular}\end{center}')
app.append(r'Incoherence aggregates all three behavioral groups. For the unsteered row, all five equally sized style groups are averaged. Truncation is not itself judged incoherence. Main judgments describe the displayed response prefix, not an unobserved completion.')
app.append(r'\subsection{Task-specific manipulation checks}')
app.append(r'\begin{center}\resizebox{\columnwidth}{!}{\begin{tabular}{llr}\toprule Task & Construction & L$-$H1 readout, pp [95\% CI]\\\midrule')
for group in ['accuracy','sycophancy','safety']:
 for key,label in [('perception_by_group','Rewrite'),('core_perception_by_group','Exact core')]:
  if group not in s[key]:continue
  a,lo,hi=s[key][group]['llm']['delta']
  app.append(f'{group} & {label} & {100*a:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]'+r'\\')
app.append(r'\bottomrule\end{tabular}}\end{center}')
app.append(r'Pooled authorship scores conceal substantial task heterogeneity. The arithmetic readout barely changes, especially for exact cores, even though accuracy and response format change strongly. The larger safety and false-belief readout changes must not be used as a manipulation check for arithmetic.')
app.append(r'\subsection{Authorship readout under interventions}\begin{center}\small\begin{tabular}{lrr}\toprule Condition & $P(\mathrm{AI})$ (\%) & Change from sham (pp)\\\midrule')
for c,x in s['perception_steer'].items():
 app.append(c.replace('_',r'\_')+f" & {p(x['mean'])} & {100*x['delta'][0]:+.2f}"+r'\\')
app.append(r'\bottomrule\end{tabular}\end{center}')
app.append(f"Mean two-label probability mass is {min(s['label_mass'].values()):.4f} or greater across intervention conditions. Label normalization is therefore not hiding a large rejection of the classification format in these aggregates.")
app.append(r'\subsection{Prompt example and exact wrappers}')
r={x['id']:x for x in map(json.loads,open('results/rewrites.jsonl'))}['acc01']
for k in ['original','h1','llm','formal']:
 app.append(r'\paragraph{'+k+r'.} '+r['variants'][k].replace('&',r'\&'))
app.append(r'The correct answer is '+str(r['answer'])+r'. None of these prompts requests a worked solution. The original remains verbatim after each wrapper in the exact-core experiment.')
w=json.load(open('results/locked_core_prompts.json'))['wrappers']
for k,t in w.items():app.append(r'\paragraph{'+k+r' wrapper.} \texttt{'+t.strip().replace('&',r'\&')+r'} followed by a blank line and the original task.')
app.append(r'\subsection{Implementation and audit trail}')
app.append(r'The target checkpoint revision is \texttt{a09a35458c702b33eeacc393d103063234e8bc28}. Inference uses PyTorch 2.5.1 and Transformers 4.51.3, bfloat16, SDPA, and batches of 12. Sampling is disabled. The archived generation configuration still contains sampling defaults, which are inactive under greedy decoding. No model weights are updated.')
app.append(r'API calls use temperature zero initially; malformed-response retries can increase temperature in increments of 0.1. One narrowly defined parser repair removes a duplicated opening quote on JSON keys while preserving values; original responses and repaired strings are retained. Missing or malformed judgments are not replaced by assumed scores. The preliminary unblinded audit is retained but not used in the reported analysis. One blinded audit incorrectly rejects an exact original/reference match, illustrating the need to inspect content rather than trust a semantic-equivalence label alone.')
app.append(f"In the exact-core follow-up, deterministic integer scoring corrects {s['core_exact_corrections']} conflicting API correctness labels. For instance, the archived judgment can be compared directly against the computed answer; raw labels are never overwritten in the judgment file.")
app.append(r'Numeral-only decoding permits digit tokens and end-of-sequence after the first digit, and caps output at 12 tokens. It does not force the correct answer or supply a reasoning trace. Because masking changes which next token is selected, its result is a response-channel sensitivity test, not an estimate of the isolated benefit of internal reasoning.')
write('appendix_text.tex','\n\n'.join(app)+'\n')
print('Rendered paper result sections from results/summary.json')
