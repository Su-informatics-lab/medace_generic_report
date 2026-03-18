#!/bin/bash -l
#SBATCH --job-name=medace_hf_oss120b
#SBATCH --account=r01985
#SBATCH --partition=hopper
#SBATCH --qos=hopper
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$SLURM_SUBMIT_DIR}"
INPUT_PATH="${INPUT_PATH:-/N/project/_A-Aa-a0-9/note/report}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/quartz_${SLURM_JOB_ID}/json}"
FAIL_DIR="${FAIL_DIR:-$REPO_DIR/outputs/quartz_${SLURM_JOB_ID}/failed}"
STATUS_TSV="${STATUS_TSV:-$REPO_DIR/outputs/quartz_${SLURM_JOB_ID}/status.tsv}"
AGGREGATE_CSV="${AGGREGATE_CSV:-$REPO_DIR/outputs/quartz_${SLURM_JOB_ID}/extractions.csv}"
PATTERN="${PATTERN:-*.txt}"
LIMIT="${LIMIT:-0}"

MODEL="${MODEL:-oss-20b}"
NUM_GPUS="${NUM_GPUS:-1}"
DTYPE="${DTYPE:-auto}"
INPUT_FORMAT="${INPUT_FORMAT:-text}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

VENV_PATH="${VENV_PATH:-$REPO_DIR/.venv}"
HF_HOME="${HF_HOME:-/N/project/textattn/hf_cache}"

mkdir -p "$(dirname "$STATUS_TSV")" "$(dirname "$AGGREGATE_CSV")" "$OUTPUT_DIR" "$FAIL_DIR" "$HF_HOME" "$REPO_DIR/logs"
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
  $( [[ "$SKIP_EXISTING" == "1" ]] && echo --skip-existing )
