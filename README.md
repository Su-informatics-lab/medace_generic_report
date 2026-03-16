# MedACE Pediatric Genetic Report Extraction

Schema-driven extraction for pediatric and NICU genetic testing reports, with REDCap-compatible import and export utilities and a Quartz-ready vLLM workflow.

## Repository layout

```text
.
├── README.md
├── main.py
├── pull_container.sbatch
├── redcap_columns_from_dictionary.json
├── run.sbatch
└── schema.py
```

## What this repository does

- extracts structured genetics data from report text or PDFs
- keeps report-level linkage among test metadata, findings, HPO terms, and diagnoses
- supports import from and export back to the current REDCap flat layout
- runs on IU Quartz with a local vLLM server inside Apptainer

## Core files

- `main.py`: main CLI for extraction and REDCap import/export
- `schema.py`: single source of truth for the extraction schema and validation
- `run.sbatch`: one-step Quartz job that activates the environment, starts vLLM, waits for readiness, and runs extraction
- `pull_container.sbatch`: one-time job to build the Apptainer image from the official `vllm/vllm-openai` container
- `redcap_columns_from_dictionary.json`: current derived REDCap column order

## Quartz setup

### 1) create the Python environment

```bash
module purge
module load python/3.12.4
module load apptainer/1.3.6

python -m venv .venv
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

### 2) configure the Hugging Face cache

Store the model cache on project storage, not in home.

```bash
export HF_HOME=/N/project/textattn/hf_cache
mkdir -p "$HF_HOME"
printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
chmod 600 "$HF_HOME/token"
```

`run.sbatch` will read the token from `$HF_HOME/token` automatically.

### 3) build the Apptainer image once

```bash
sbatch pull_container.sbatch
```

Expected output image:

```text
containers/vllm-openai-latest.sif
```

## Running extraction on Quartz

`run.sbatch` is the main entry point. It does all of the following in one job:

1. loads Quartz modules
2. activates `.venv`
3. writes the GPT-OSS Hopper config
4. starts a local vLLM server inside Apptainer
5. waits for server readiness
6. runs `main.py` over the report files
7. writes JSON, CSV, and status outputs

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

## Important runtime parameters

You can override these at submission time with `--export=ALL,...`:

- `INPUT_DIR`: folder containing report `.txt` files
- `LIMIT`: number of files to process; `0` means all files
- `MODEL`: defaults to `openai/gpt-oss-120b`
- `HF_HOME`: Hugging Face cache root
- `IMG_FILE`: Apptainer image path

Example:

```bash
sbatch --export=ALL,INPUT_DIR=/N/project/_A-Aa-a0-9/note/report,LIMIT=10 run.sbatch
```

## Outputs

### Slurm and server logs

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

For non-Slurm or debugging workflows, inspect the CLI directly:

```bash
python main.py --help
```

Common tasks include:

- report extraction
- REDCap import to linked JSON
- linked JSON export back to REDCap-compatible rows
- printing canonical REDCap column order from a data dictionary

## Notes

- `run.sbatch` is the preferred production path on Quartz.
- The current batch workflow assumes report text files are available in a directory and processes them sequentially.
- The schema is designed to preserve report-to-diagnosis linkage when available and to stay backward compatible with the current REDCap layout.
