#!/bin/bash -l
#SBATCH --job-name=medace_quartz_test
#SBATCH --account=r01985
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------------
# User-tunable settings
# -------------------------------
CODE_DIR="${CODE_DIR:-/N/project/_A-Aa-a0-9/hw56/medace_generic_report}"
INPUT_DIR="${INPUT_DIR:-/N/project/_A-Aa-a0-9/note/report}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
LIMIT="${LIMIT:-5}"               # 0 = all files; default 5 for a quick smoke test
PORT="${PORT:-8020}"
GPU_UTIL="${GPU_UTIL:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-1024}"
READ_TIMEOUT="${READ_TIMEOUT:-1800}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12000}"
MAX_RETRIES="${MAX_RETRIES:-3}"
IMG_FILE="${IMG_FILE:-/N/project/_A-Aa-a0-9/hw56/medace_generic_report/containers/vllm-openai-deploy.sif}"
HF_HOME="${HF_HOME:-/N/project/textattn/hf_cache}"
API="http://127.0.0.1:${PORT}"
RUN_DIR="${RUN_DIR:-$CODE_DIR/outputs/quartz_${SLURM_JOB_ID}}"
JSON_DIR="$RUN_DIR/json"
CSV_OUT="$RUN_DIR/extractions.csv"
LOG_DIR="$CODE_DIR/logs"
STATUS_TSV="$RUN_DIR/status.tsv"

mkdir -p "$LOG_DIR" "$RUN_DIR" "$JSON_DIR" "$HF_HOME"
cd "$CODE_DIR"

export PYTHONUNBUFFERED=1
export HF_HOME
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# pick up HF token from file if present
if [[ -z "${HF_TOKEN:-}" && -f "$HF_HOME/token" ]]; then
  export HF_TOKEN="$(<"$HF_HOME/token")"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
else
  echo "[warn] HF token not found in environment or $HF_HOME/token"
  echo "[warn] model download may fail if the repo requires authentication"
fi

# -------------------------------
# Explicit Quartz modules
# -------------------------------
module purge || true
module load python/3.12.4
module load apptainer/1.3.6

# -------------------------------
# Activate venv
# -------------------------------
if [[ ! -f "$CODE_DIR/.venv/bin/activate" ]]; then
  echo "[fatal] missing virtualenv at $CODE_DIR/.venv"
  exit 2
fi
source "$CODE_DIR/.venv/bin/activate"

command -v python >/dev/null || { echo "[fatal] python not found after venv activation"; exit 3; }
command -v apptainer >/dev/null || { echo "[fatal] apptainer not found after module load"; exit 4; }
command -v curl >/dev/null || { echo "[fatal] curl not found"; exit 5; }
[[ -f "$CODE_DIR/main.py" ]] || { echo "[fatal] missing $CODE_DIR/main.py"; exit 6; }

python --version
which python
apptainer --version
nvidia-smi || true

if [[ ! -f "$IMG_FILE" ]]; then
  echo "[fatal] Apptainer image not found: $IMG_FILE"
  echo "        Build it once on the login node with:"
  echo "        apptainer pull $IMG_FILE docker://vllm/vllm-openai:deploy"
  exit 7
fi

# -------------------------------
# Start vLLM in the background
# -------------------------------
echo "[info] starting vLLM server"

apptainer_cmd=(
  apptainer exec --nv
  --bind "$HF_HOME":/root/.cache/huggingface
  "$IMG_FILE" bash -lc
  "set -euo pipefail; \
   export HF_HOME=/root/.cache/huggingface; \
   vllm serve '$MODEL' \
     --served-model-name '$MODEL' \
     --host 0.0.0.0 \
     --port '$PORT' \
     --download-dir /root/.cache/huggingface \
     --tensor-parallel-size 1 \
     --gpu-memory-utilization '$GPU_UTIL' \
     --max-model-len '$MAX_MODEL_LEN' \
     --max-num-seqs '$MAX_NUM_SEQS' \
     --max-num-batched-tokens '$MAX_BATCHED_TOKENS' \
     --enable-prefix-caching \
     --enforce-eager"
)

"${apptainer_cmd[@]}" >"$LOG_DIR/vllm_${SLURM_JOB_ID}.log" 2>"$LOG_DIR/vllm_${SLURM_JOB_ID}.err" &
SERVER_PID=$!

cleanup() {
  echo "[info] stopping vLLM server"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for readiness
for ((i=1; i<=300; i++)); do
  if curl -fsS "$API/v1/models" >/dev/null 2>&1; then
    echo "[info] vLLM ready at $API"
    break
  fi
  if (( i == 300 )); then
    echo "[fatal] vLLM did not become ready in time"
    exit 8
  fi
  sleep 2
done

# Warmup
curl -fsS "$API/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"ping"}],"max_tokens":1}' \
  >/dev/null || true

# -------------------------------
# Build report list
# -------------------------------
mapfile -t REPORTS < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -name 'GENETIC_REPORT_*.txt' -o -name '*.txt' \) | sort)

if [[ ${#REPORTS[@]} -eq 0 ]]; then
  echo "[fatal] no report .txt files found under $INPUT_DIR"
  exit 9
fi

if [[ "$LIMIT" =~ ^[0-9]+$ ]] && (( LIMIT > 0 )) && (( ${#REPORTS[@]} > LIMIT )); then
  REPORTS=("${REPORTS[@]:0:LIMIT}")
fi

echo -e "source_file\tjson_path\tstatus\tmessage" > "$STATUS_TSV"
echo "[info] processing ${#REPORTS[@]} report(s)"

# -------------------------------
# Run extraction sequentially
# -------------------------------
for report in "${REPORTS[@]}"; do
  base="$(basename "$report")"
  stem="${base%.*}"
  json_out="$JSON_DIR/${stem}.extracted.json"
  client_log="$LOG_DIR/${stem}_${SLURM_JOB_ID}.client.log"
  client_err="$LOG_DIR/${stem}_${SLURM_JOB_ID}.client.err"

  echo "[extract] $base"
  if python main.py \
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
  else
    printf '%s\t%s\tfail\tsee %s\n' "$report" "$json_out" "$client_err" >> "$STATUS_TSV"
    echo "[warn] extraction failed for $base"
  fi
done

# -------------------------------
# Aggregate JSON outputs into CSV
# -------------------------------
JSON_DIR="$JSON_DIR" CSV_OUT="$CSV_OUT" python - <<'PY'
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

echo "[done] JSON outputs: $JSON_DIR"
echo "[done] CSV output : $CSV_OUT"
echo "[done] status TSV : $STATUS_TSV"
