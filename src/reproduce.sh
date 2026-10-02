#!/usr/bin/env bash
# Run from the repository root. Existing raw records are resumed, never fabricated.
set -euo pipefail
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv
fi
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python src/setup_model.py
.venv/bin/python src/build_data.py
.venv/bin/python src/rewrite.py
.venv/bin/python src/audit.py
src/run_all.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/format_control.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/locked_core.py
.venv/bin/python src/judge.py --input results/core_generations.jsonl --output results/core_judgments.jsonl
.venv/bin/python src/judge.py
.venv/bin/python src/judge.py --model openai/gpt-4.1-mini --subset baseline_safety --output results/judgments_second.jsonl
.venv/bin/python src/analyze.py
.venv/bin/python src/build_report.py
.venv/bin/python src/validate_artifacts.py
(cd paper_draft && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
