#!/bin/bash -l
#SBATCH --job-name=pull_vllm_image
#SBATCH --account=r01985
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module purge
module load apptainer/1.3.6

CODE_DIR="/N/project/_A-Aa-a0-9/hw56/medace_generic_report"
IMG_FILE="$CODE_DIR/containers/vllm-openai-latest.sif"
APPTAINER_CACHEDIR="/N/project/textattn/apptainer_cache"
APPTAINER_TMPDIR="/N/project/textattn/apptainer_tmp"

mkdir -p "$CODE_DIR/containers" "$CODE_DIR/logs" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

export APPTAINER_CACHEDIR
export APPTAINER_TMPDIR

cd "$CODE_DIR"

rm -f "$IMG_FILE"
apptainer pull "$IMG_FILE" docker://vllm/vllm-openai:latest

ls -lh "$IMG_FILE"
