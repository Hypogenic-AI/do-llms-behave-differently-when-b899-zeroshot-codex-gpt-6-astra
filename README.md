# Prompt style, perceived authorship, and model behavior

This workspace contains an executed, single-model study of whether conversational and structured prompt styles change Qwen2.5-7B-Instruct's responses, and whether fixed-text activation interventions support an authorship-based explanation.

**Scope:** every item and rewrite is synthetic. “Human-like” means an intended conversational style, not verified human authorship. This is an exploratory mechanistic pilot, not a deployment-wide safety evaluation.

## Study

- 120 behavioral requests: 40 arithmetic questions, 40 false-belief arithmetic questions, and 20 risky/benign request pairs.
- Five wording conditions: original, two conversational paraphrases, structured LLM-like wording, and formal prose.
- 80 separate benign calibration requests, split by topic into 60 training and 20 test items.
- Fixed-text steering at decoder block 14: style, nuisance-projected style, explicit authorship, explicit evaluation context, three norm-matched random directions, and a zero-vector sham. Both signs, plus doubled style/projection doses.
- Counterbalanced target-model authorship classification; independent semantic audits and response judging; a second safety judge; exact numerical checks.
- An exploratory numeral-only decoding control tests response-channel sensitivity in arithmetic.
- An exact-core follow-up holds task text verbatim and wrapper token lengths equal on 40 accuracy and 40 safety requests.

The complete protocol and amendments are in [results/protocol.md](results/protocol.md). The short-cap generation pilot, failed initial software run, original unblinded audit, malformed API responses, and retry logs are retained separately. No simulated measurements are used.

## Findings

The completed study contains **3,320 generated responses**, **7,040 authorship classifications**, and **800 calibration representations**. See [the compiled paper](paper_draft/main.pdf) and [the numerical summary](results/summary.json).

- **Full rewrites:** arithmetic accuracy is 70.0% for the intended LLM-like style versus 22.5% for conversational H1, a paired gain of 47.5 percentage points (95% bootstrap CI: 32.5–62.5). The more accurate styles often violate the integer-only instruction by producing worked solutions.
- **The effect reverses under a stricter content control:** with verbatim task cores and equal-token wrappers, conversational accuracy is 95.0% versus 25.0% for task-framed wording. The arithmetic-specific authorship readout barely changes, so the pooled authorship check cannot explain this reversal. Numeral-only decoding also removes the original advantage.
- **No identified authorship mechanism:** the style direction aligns strongly with formality (cosine 0.871), minimally with explicit authorship (0.032), and fails the intended positive authorship-readout steering check. This is not evidence that authorship can never matter.
- **Safety results are inconclusive:** the primary risky-refusal difference is −10 percentage points (95% CI: −35 to +15); false-belief agreement is zero in all wording conditions, leaving that test floor-limited. Automated safety judges agree on 90.5% of refusal labels but only 68.0% of fulfillment labels.

All estimates describe this synthetic, single-model study. The workspace retains failures and protocol amendments, and separates primary comparisons from exploratory follow-ups. Single-integer correctness is checked against arithmetic ground truth; conflicting raw judge labels remain archived.

## Files

- `paper_draft/main.tex` and `paper_draft/main.pdf`: paper source and compiled paper.
- `src/`: item construction, API calls, inference, intervention, judging, analysis, and validation.
- `results/items.json`, `rewrites.jsonl`, `audits.jsonl`: requests, exact variants, and blinded audits.
- `results/generations.jsonl`: main response records with a 512-token cap (including token IDs and truncation flags).
- `results/calibration_activations.npz`, `directions.npz`, and calibration metadata: actual hidden states and intervention vectors.
- `results/perception.jsonl`: both A/B label orders for authorship classification.
- `results/judgments*.jsonl`: blind response judgments and second-judge check.
- `results/api_raw.jsonl`: API requests and full provider responses; credentials are never saved.
- `results/format_control.jsonl`: actual restricted-decoding outputs.
- `results/locked_core_prompts.json`, `core_generations.jsonl`, `core_perception.jsonl`, and `core_judgments.jsonl`: verbatim-core follow-up records.
- `results/scored_outputs.csv`, `summary.json`, and paper tables/figures: analysis outputs.
- `results/model_manifest.json`, `environment.json`, and `checksums.json`: provenance.

Calibration split membership in `calibration_entries.json` and `calibration_meta.json` is authoritative. Early raw item/rewrite records retain an unused preliminary split field; the executed split is by topic, with no train/test topic overlap.

Model weights and caches live under ignored `models/`; the isolated environment is `.venv/`.

## Reproduce

Run from the repository root on Linux with an NVIDIA GPU with sufficient free memory (the executed run used one 48 GB RTX A6000), Python 3.12, `uv`, and a LaTeX installation with `latexmk` and BibTeX. For fresh API generation/scoring set `OPENROUTER_KEY` in the environment. Qwen is ungated; no model-access token is required.

```bash
uv venv .venv
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python src/setup_model.py
.venv/bin/python src/analyze.py
.venv/bin/python src/build_report.py
.venv/bin/python src/validate_artifacts.py
cd paper_draft
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This re-analyzes archived measurements and rebuilds the paper. Downloading the model also supplies the tokenizer used for length analysis. To rerun inference and API stages, see `src/reproduce.sh`. Scripts resume existing records by key. For an entirely fresh experiment, copy `src/`, the requirements files, and paper sources into a new directory and begin with an empty `results/`; do not mix new calibration vectors with old inference records. Fresh API outputs need not match archived outputs even at temperature zero. Greedy GPU inference can also differ slightly by software or hardware.

`src/run_all.sh` executes calibration, wording responses, steering, and authorship readout. `src/format_control.py` executes the post-hoc decoding control. `src/judge.py` scores available response records. `src/analyze.py` creates statistical summaries, tables, and figures; `src/build_report.py` renders the numerical paper sections. The prose describes this archived run and should be reviewed if rerunning with changed data. `src/validate_artifacts.py` checks record counts, uniqueness, topic separation, direction norms, and orthogonality.

## Interpretation limits

Only one target model and one layer are studied. The sycophancy task is narrow, the safety sample is small, and automated judges are imperfect. Content matching is approximate. Source-attribution and evaluation vectors are learned from explicit labels, not validated natural provenance. Authorship classification uses a different system task from response generation. A steering effect establishes an effect of that vector intervention, not that perceived authorship mediates the effect. Nonsignificant comparisons do not establish equivalence.
