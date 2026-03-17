# MedACE Pediatric Genetic Report Extraction

Schema-driven extraction for pediatric and NICU genetic testing reports, with REDCap-compatible import/export and a Quartz-oriented vLLM workflow.

## Repository layout

```text
.
├── README.md
├── JOURNEY.md
├── main.py
├── pull_container.sbatch
├── redcap_columns_from_dictionary.json
├── run.sbatch
└── schema.py
```

## Scope

This repository does four things:

- extracts structured genetics data from report text or PDFs
- preserves report-level linkage among test metadata, findings, HPO terms, and diagnoses
- imports from and exports back to the current flat REDCap layout
- runs batch extraction on IU Quartz with a local vLLM server inside Apptainer

## Core files

- `main.py` — main CLI for extraction and REDCap import/export
- `schema.py` — single source of truth for the extraction schema and validation
- `run.sbatch` — main Quartz batch entry point; starts vLLM and runs extraction
- `pull_container.sbatch` — one-time Slurm job to build the Apptainer image
- `redcap_columns_from_dictionary.json` — current derived REDCap column order
- `JOURNEY.md` — working status, blockers, and next steps

## Recommended Quartz setup

### 1) Create the virtual environment

Create the environment once inside the repo.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "pydantic>=2,<3"
```

Optional extras:

```bash
# PDF input
python -m pip install "pymupdf>=1.24" "pdfplumber>=0.11"

# data dictionary parsing
python -m pip install "openpyxl>=3.1"
```

Batch jobs should use the repo-local `.venv` directly. Do not rely on Quartz Python modules inside `run.sbatch`.

### 2) Configure project-level caches

Keep model and container caches on project storage, not in home.

```bash
export HF_HOME=/N/project/textattn/hf_cache
mkdir -p "$HF_HOME"
printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
chmod 600 "$HF_HOME/token"
```

Recommended Apptainer cache locations:

```bash
export APPTAINER_CACHEDIR=/N/project/textattn/apptainer_cache
export APPTAINER_TMPDIR=/N/project/textattn/apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
```

`run.sbatch` reads the Hugging Face token from `$HF_HOME/token` automatically.

### 3) Build the Apptainer image once

Do this through Slurm, not on the login node.

```bash
sbatch pull_container.sbatch
```

Expected image:

```text
containers/vllm-openai-latest.sif
```

A successful sanity check looks like:

```bash
module purge
module load apptainer/1.3.6
apptainer exec --cleanenv containers/vllm-openai-latest.sif /bin/sh -lc 'python3 -c "import vllm; print(vllm.__version__)"'
```

## Running extraction on Quartz

`run.sbatch` is the main entry point. It should remain the single production script.

It is responsible for:

1. loading Apptainer
2. using the repo-local `.venv`
3. writing the GPT-OSS Hopper config
4. starting a local vLLM server inside Apptainer
5. waiting for readiness
6. running `main.py` over the report files
7. writing JSON, CSV, and status outputs

### Smoke test

```bash
mkdir -p logs outputs
sbatch --export=ALL,LIMIT=1 run.sbatch
```

### Small batch

```bash
sbatch --export=ALL,LIMIT=5 run.sbatch
```

### All reports

```bash
sbatch --export=ALL,LIMIT=0 run.sbatch
```

## Important Quartz notes

- On Quartz, `gpu` and `gpu-interactive` are V100 partitions.
- On Quartz, `hopper` is the H100 partition.
- `gpt-oss-120b` should run on `hopper`, not on the V100 partitions.
- At the moment, Hopper access is allocation-dependent. If the current Slurm account does not have Hopper QOS, jobs will fail with `Invalid qos specification`.
- Check the live partition layout with:

```bash
sinfo -o "%P %G %N" | egrep 'hopper|gpu|interactive'
```

- Check account/QOS associations with:

```bash
sacctmgr -nP show assoc where user=$USER format=cluster,account,user,partition,qos%50,defaultqos%30
```

## Runtime overrides

Useful overrides at submission time:

- `INPUT_DIR` — folder containing report `.txt` files
- `LIMIT` — number of files to process; `0` means all files
- `MODEL` — defaults to `openai/gpt-oss-120b`
- `HF_HOME` — Hugging Face cache root
- `IMG_FILE` — Apptainer image path
- `RUN_DIR` — output directory for the current job

Example:

```bash
sbatch --export=ALL,INPUT_DIR=/N/project/_A-Aa-a0-9/note/report,LIMIT=10 run.sbatch
```

## Outputs

### Logs

```text
logs/medace_quartz_test_<jobid>.log
logs/medace_quartz_test_<jobid>.err
logs/vllm_<jobid>.log
logs/vllm_<jobid>.err
logs/<report>_<jobid>.client.log
logs/<report>_<jobid>.client.err
```

The Slurm `.err` file is the main progress log.

### Extraction artifacts

```text
outputs/quartz_<jobid>/json/*.extracted.json
outputs/quartz_<jobid>/extractions.csv
outputs/quartz_<jobid>/status.tsv
```

## Direct CLI use

For debugging or non-Slurm use:

```bash
source .venv/bin/activate
python main.py --help
```

Common tasks include:

- report extraction
- REDCap import to linked JSON
- linked JSON export back to REDCap-compatible rows
- printing canonical REDCap column order from a data dictionary

## Troubleshooting

- If `apptainer pull` is killed on the login node, use `pull_container.sbatch`.
- If a Hopper job fails with `Invalid qos specification`, the Slurm account does not currently have Hopper permission.
- If a batch job prints noisy Lmod dependency messages at startup, do not chase them first; confirm whether the repo `.venv` and Apptainer launch are actually failing.
- If the vLLM server exits before readiness, inspect `logs/vllm_<jobid>.err` first.

## Status tracking

The current project state, blockers, and next actions live in `JOURNEY.md`.

## License
MIT
