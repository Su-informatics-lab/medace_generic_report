#!/bin/bash -l
#SBATCH --job-name=medace_resume
#SBATCH --account=r01985
#SBATCH --partition=hopper
#SBATCH --qos=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ── Resume from previous run ──────────────────────────────────────────────
# Point all output paths to the EXISTING quartz_8819693 directory so
# --skip-existing skips the ~1000 JSONs already written.
REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
PREV_RUN="$REPO_DIR/outputs/quartz_8819693"

INPUT_PATH="${INPUT_PATH:-/N/project/_A-Aa-a0-9/note/report}"
OUTPUT_DIR="$PREV_RUN/json"
FAIL_DIR="$PREV_RUN/failed"
STATUS_TSV="$PREV_RUN/status.tsv"
AGGREGATE_CSV="$PREV_RUN/extractions.csv"
PATTERN="${PATTERN:-*.txt}"
LIMIT="${LIMIT:-0}"

MODEL="${MODEL:-oss-20b}"
NUM_GPUS="${NUM_GPUS:-2}"
DTYPE="${DTYPE:-auto}"
INPUT_FORMAT="${INPUT_FORMAT:-text}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SKIP_EXISTING=1

VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
HF_HOME="${HF_HOME:-/N/project/textattn/hf_cache}"

mkdir -p "$OUTPUT_DIR" "$FAIL_DIR" "$HF_HOME" "$REPO_DIR/logs"
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME

if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(<"$HF_HOME/token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

PYTHON_BIN="$VENV_PATH/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[fatal] missing virtualenv python at $PYTHON_BIN" >&2
  exit 2
fi

echo "[resume] Resuming into $PREV_RUN"
echo "[resume] Existing JSONs: $(ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)"
echo "[resume] Existing failures: $(ls "$FAIL_DIR"/* 2>/dev/null | wc -l)"

"$PYTHON_BIN" batch_extract_hf.py \
  --input-path "$INPUT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --fail-dir "$FAIL_DIR" \
  --status-tsv "$STATUS_TSV" \
  --aggregate-csv "$AGGREGATE_CSV" \
  --pattern "$PATTERN" \
  --limit "$LIMIT" \
  --model "$MODEL" \
  --num-gpus "$NUM_GPUS" \
  --dtype "$DTYPE" \
  --input-format "$INPUT_FORMAT" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-retries "$MAX_RETRIES" \
  --skip-existing

echo "[resume] Done at $(date)"
echo "[resume] Final JSONs: $(ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)"
