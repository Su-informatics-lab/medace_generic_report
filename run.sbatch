#!/bin/bash -l
#SBATCH --job-name=medace_quartz_test
#SBATCH --account=r01985
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------------
# user-tunable settings
# -------------------------------
CODE_DIR="${CODE_DIR:-/N/project/_A-Aa-a0-9/hw56/medace_generic_report}"
INPUT_DIR="${INPUT_DIR:-/N/project/_A-Aa-a0-9/note/report}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
LIMIT="${LIMIT:-5}"                 # 0 = all files; default 5 for a smoke test
PORT="${PORT:-8020}"
READ_TIMEOUT="${READ_TIMEOUT:-1800}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
IMG_FILE="${IMG_FILE:-/N/project/_A-Aa-a0-9/hw56/medace_generic_report/containers/vllm-openai-latest.sif}"
HF_HOME="${HF_HOME:-/N/project/textattn/hf_cache}"
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/N/project/textattn/apptainer_cache}"
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/N/project/textattn/apptainer_tmp}"
API="http://127.0.0.1:${PORT}"
RUN_DIR="${RUN_DIR:-$CODE_DIR/outputs/quartz_${SLURM_JOB_ID}}"
JSON_DIR="$RUN_DIR/json"
CSV_OUT="$RUN_DIR/extractions.csv"
LOG_DIR="$CODE_DIR/logs"
STATUS_TSV="$RUN_DIR/status.tsv"
YAML_FILE="$CODE_DIR/GPT-OSS_Hopper.yaml"
VLLM_LOG="$LOG_DIR/vllm_${SLURM_JOB_ID}.log"
VLLM_ERR="$LOG_DIR/vllm_${SLURM_JOB_ID}.err"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$JSON_DIR" "$HF_HOME" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
cd "$CODE_DIR"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" >&2
}

log "job started on host $(hostname)"
log "code dir: $CODE_DIR"
log "input dir: $INPUT_DIR"
log "run dir: $RUN_DIR"
log "image   : $IMG_FILE"
log "hf home : $HF_HOME"

export PYTHONUNBUFFERED=1
export HF_HOME
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export APPTAINER_CACHEDIR APPTAINER_TMPDIR
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# pick up HF token from file if present
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(<"$HF_HOME/token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
  export APPTAINERENV_HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
  log "HF token loaded from environment/file"
else
  log "warn: HF token not found in environment or $HF_HOME/token"
  log "warn: model download may fail if the repo requires authentication"
fi

# only pass what the clean container needs
export APPTAINERENV_HF_HOME=/root/.cache/huggingface
export APPTAINERENV_VLLM_NO_USAGE_STATS=1

# -------------------------------
# modules: avoid Quartz Python modules; use the repo venv directly
# -------------------------------
module purge || true
module load apptainer/1.3.6

# -------------------------------
# use repo venv python explicitly
# -------------------------------
PYTHON_BIN="$CODE_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  log "fatal: missing virtualenv python at $PYTHON_BIN"
  exit 2
fi

command -v apptainer >/dev/null || { log "fatal: apptainer not found after module load"; exit 4; }
command -v curl >/dev/null || { log "fatal: curl not found"; exit 5; }
[[ -f "$CODE_DIR/main.py" ]] || { log "fatal: missing $CODE_DIR/main.py"; exit 6; }
[[ -d "$INPUT_DIR" ]] || { log "fatal: missing input dir $INPUT_DIR"; exit 9; }
[[ -f "$IMG_FILE" ]] || { log "fatal: Apptainer image not found: $IMG_FILE"; exit 7; }

log "python: $("$PYTHON_BIN" --version 2>&1)"
log "python path: $PYTHON_BIN"
log "apptainer: $(apptainer --version 2>&1)"
"$PYTHON_BIN" -c "import pydantic, schema; print('python env ok')" >&2
log "gpu inventory follows"
nvidia-smi >&2 || true

# -------------------------------
# write the GPT-OSS Hopper config
# -------------------------------
cat > "$YAML_FILE" <<'YAML'
no-enable-prefix-caching: true
max-cudagraph-capture-size: 2048
max-num-batched-tokens: 8192
stream-interval: 20
YAML
log "wrote Hopper config to $YAML_FILE"

# -------------------------------
# start vLLM in the background
# -------------------------------
log "starting vLLM server"

apptainer exec --cleanenv --nv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --bind "$CODE_DIR":/work \
  "$IMG_FILE" /bin/sh -lc "
    set -eu
    cd /work
    vllm serve '$MODEL' \
      --config /work/$(basename "$YAML_FILE") \
      --served-model-name '$MODEL' \
      --host 0.0.0.0 \
      --port '$PORT' \
      --download-dir /root/.cache/huggingface \
      --tensor-parallel-size 1
  " >"$VLLM_LOG" 2>"$VLLM_ERR" &
SERVER_PID=$!
log "vLLM pid: $SERVER_PID"
log "vLLM stdout: $VLLM_LOG"
log "vLLM stderr: $VLLM_ERR"

cleanup() {
  log "stopping vLLM server"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# wait for readiness
for ((i=1; i<=300; i++)); do
  if curl -fsS "$API/v1/models" >/dev/null 2>&1; then
    log "vLLM ready at $API"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "fatal: vLLM exited before becoming ready"
    log "last vLLM stderr lines follow"
    tail -n 80 "$VLLM_ERR" >&2 || true
    exit 8
  fi
  if (( i % 10 == 0 )); then
    log "waiting for vLLM readiness... (${i}/300)"
  fi
  if (( i == 300 )); then
    log "fatal: vLLM did not become ready in time"
    log "last vLLM stderr lines follow"
    tail -n 80 "$VLLM_ERR" >&2 || true
    exit 8
  fi
  sleep 2
done

# warmup
curl -fsS "$API/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
  >/dev/null || true
log "warmup request sent"

# -------------------------------
# build report list
# -------------------------------
mapfile -t REPORTS < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name 'GENETIC_REPORT_*.txt' -o -name '*.txt' \) | sort)

if [[ ${#REPORTS[@]} -eq 0 ]]; then
  log "fatal: no report .txt files found under $INPUT_DIR"
  exit 10
fi

if [[ "$LIMIT" =~ ^[0-9]+$ ]] && (( LIMIT > 0 )) && (( ${#REPORTS[@]} > LIMIT )); then
  REPORTS=("${REPORTS[@]:0:LIMIT}")
fi

echo -e "source_file\tjson_path\tstatus\tmessage" > "$STATUS_TSV"
log "processing ${#REPORTS[@]} report(s)"

# -------------------------------
# run extraction sequentially
# -------------------------------
for idx in "${!REPORTS[@]}"; do
  report="${REPORTS[$idx]}"
  base="$(basename "$report")"
  stem="${base%.*}"
  json_out="$JSON_DIR/${stem}.extracted.json"
  client_log="$LOG_DIR/${stem}_${SLURM_JOB_ID}.client.log"
  client_err="$LOG_DIR/${stem}_${SLURM_JOB_ID}.client.err"

  log "[$((idx+1))/${#REPORTS[@]}] extracting $base"
  if "$PYTHON_BIN" main.py \
      --input "$report" \
      --output "$json_out" \
      --model "$MODEL" \
      --backend vllm \
      --base-url "$API" \
      --input-format text \
      --read-timeout "$READ_TIMEOUT" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --max-retries "$MAX_RETRIES" \
      >"$client_log" 2>"$client_err"; then
    printf '%s\t%s\tok\t\n' "$report" "$json_out" >> "$STATUS_TSV"
    log "[$((idx+1))/${#REPORTS[@]}] success: $base"
  else
    printf '%s\t%s\tfail\tsee %s\n' "$report" "$json_out" "$client_err" >> "$STATUS_TSV"
    log "[$((idx+1))/${#REPORTS[@]}] failed: $base; last client stderr lines follow"
    tail -n 50 "$client_err" >&2 || true
  fi
done

# -------------------------------
# aggregate JSON outputs into CSV
# -------------------------------
log "aggregating JSON outputs into CSV"
JSON_DIR="$JSON_DIR" CSV_OUT="$CSV_OUT" "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

json_dir = Path(os.environ["JSON_DIR"])
csv_out = Path(os.environ["CSV_OUT"])


def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                out.update(flatten(value, new_key))
            elif isinstance(value, list):
                out[new_key] = json.dumps(value, ensure_ascii=False)
            else:
                out[new_key] = value
    else:
        out[prefix or "value"] = obj
    return out

rows = []
all_fields = ["source_file", "json_path"]
for path in sorted(json_dir.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "source_file": path.stem.replace(".extracted", "") + ".txt",
        "json_path": str(path),
    }
    row.update(flatten(data))
    rows.append(row)
    for field in row:
        if field not in all_fields:
            all_fields.append(field)

with csv_out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"[csv] wrote {csv_out} with {len(rows)} row(s)")
PY

log "done: JSON outputs at $JSON_DIR"
log "done: CSV output at $CSV_OUT"
log "done: status TSV at $STATUS_TSV"
