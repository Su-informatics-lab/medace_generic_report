#!/bin/bash -l
#SBATCH -J medace_hf_extract
#SBATCH -A r01985
#SBATCH -p hopper
#SBATCH --qos=hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
INPUT_PATH="${INPUT_PATH:-$REPO_DIR/reports}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs}"
FAIL_DIR="${FAIL_DIR:-$OUTPUT_DIR/failed}"
PATTERN="${PATTERN:-*.pdf}"
LIMIT="${LIMIT:-0}"

MODEL_ALIAS="${MODEL_ALIAS:-llama3.1-8b}"
NUM_GPUS="${NUM_GPUS:-1}"
DTYPE="${DTYPE:-auto}"
INPUT_FORMAT="${INPUT_FORMAT:-auto}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
CONDA_ENV="${CONDA_ENV:-}"
SCRATCH_ROOT="${SCRATCH:-$HOME/scratch}"
HF_HOME="${HF_HOME:-$SCRATCH_ROOT/hf-cache}"

export REPO_DIR INPUT_PATH OUTPUT_DIR FAIL_DIR PATTERN LIMIT
export MODEL_ALIAS NUM_GPUS DTYPE INPUT_FORMAT TEMPERATURE TOP_P MAX_NEW_TOKENS MAX_RETRIES SKIP_EXISTING
export HF_HOME
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR" "$FAIL_DIR" "$HF_HOME"

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  source "$VENV_PATH/bin/activate"
elif [[ -n "$CONDA_ENV" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

python -u - <<'PY'
import json
import os
import sys
import traceback
from pathlib import Path

from main import extract, load_model, load_report, resolve_model
from schema import get_extraction_model

input_path = Path(os.environ["INPUT_PATH"])
output_dir = Path(os.environ["OUTPUT_DIR"])
fail_dir = Path(os.environ["FAIL_DIR"])
pattern = os.environ["PATTERN"]
limit = int(os.environ["LIMIT"])
skip_existing = os.environ["SKIP_EXISTING"] == "1"
input_format = None if os.environ["INPUT_FORMAT"] == "auto" else os.environ["INPUT_FORMAT"]

if input_path.is_dir():
    files = sorted(p for p in input_path.glob(pattern) if p.is_file())
else:
    files = [input_path]

if limit > 0:
    files = files[:limit]

if not files:
    raise SystemExit(f"no files found under {input_path} with pattern {pattern}")

model_alias = os.environ["MODEL_ALIAS"]
model_id = resolve_model(model_alias)
num_gpus = int(os.environ["NUM_GPUS"])
dtype = os.environ["DTYPE"]
temperature = float(os.environ["TEMPERATURE"])
top_p = float(os.environ["TOP_P"])
max_new_tokens = int(os.environ["MAX_NEW_TOKENS"])
max_retries = int(os.environ["MAX_RETRIES"])

print(f"[info] loading model once: {model_alias} -> {model_id}", flush=True)
model, tokenizer = load_model(model_id, num_gpus=num_gpus, dtype=dtype)
model_cls = get_extraction_model()

ok = 0
failed = 0

for path in files:
    out_path = output_dir / f"{path.stem}.json"
    err_path = fail_dir / f"{path.stem}.error.txt"

    if skip_existing and out_path.exists():
        print(f"[skip] {path.name}", flush=True)
        continue

    try:
        report_text = load_report(path, input_format)
        result = extract(
            report_text,
            model,
            tokenizer,
            model_cls=model_cls,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            max_retries=max_retries,
        )
        out_path.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if err_path.exists():
            err_path.unlink()
        ok += 1
        print(f"[ok] {path.name} -> {out_path.name}", flush=True)
    except Exception as e:
        failed += 1
        err_path.write_text(
            f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        print(f"[fail] {path.name}: {e}", file=sys.stderr, flush=True)

print(f"[done] ok={ok} failed={failed} output_dir={output_dir}", flush=True)
PY
