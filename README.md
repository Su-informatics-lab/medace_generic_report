# MedACE Pediatric Genetic Test Report Extractor

Schema-driven structured extraction for pediatric/NICU genetic test reports.

The pipeline now supports **two completely separate inference backends**:

- **Local HuggingFace inference** for direct model loading inside the Python process
- **External vLLM endpoint inference** for OpenAI-compatible servers such as the apptainer-backed `gpt-oss-20b` / `gpt-oss-120b` jobs you launch on IU HPC

The schema in `schema.py` remains the **single source of truth**.

---

## What it does

Given a genetic testing report (`.pdf` or `.txt`), the extractor:

1. Reads the report text
2. Builds extraction instructions directly from the Pydantic schema
3. Sends the prompt to either a local HF model or a vLLM endpoint
4. Parses the JSON response
5. Validates it against the schema with Pydantic
6. Re-asks with validation errors if needed
7. Writes a clean JSON file

---

## Files

```text
medace.py   # primary CLI; backend-switchable HF <-> vLLM
main.py     # backward-compatible wrapper that calls medace.py
schema.py   # extraction schema; edit here to add/remove fields
README.md   # usage notes
```

---

## Schema coverage

The expanded schema is aligned to the highlighted REDCap variables and keeps repeatable blocks as lists instead of hardcoded suffix fields.

Top-level groups:

- `patient`
- `nicu_flags`
- `test_info`
- `patient_phenotypes`
- `findings`
- `interpretation`
- `diagnoses`

### Important modeling choice

REDCap repeats fields like `finding1 ... finding20` and diagnosis blocks like `dxname`, `dxname_2`, `dxname_3`.

In this extractor those become:

- `findings: List[GeneticFinding]`
- `diagnoses: List[GeneticDiagnosis]`

That keeps the schema much cleaner while still mapping cleanly back to REDCap later.

---

## Supported backends

### 1) Local HuggingFace backend

Good for smaller models or quick local testing.

Supported aliases:

| Alias | Resolved model ID |
|---|---|
| `oss-120b` | `openai/gpt-oss-120b` |
| `oss-20b` | `openai/gpt-oss-20b` |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` |

You can also pass any full model ID directly.

### 2) vLLM endpoint backend

Good for IU HPC jobs where the model server is started separately via sbatch/apptainer.

The CLI expects an **OpenAI-compatible** endpoint, such as:

- `http://127.0.0.1:8020`
- `http://127.0.0.1:8020/v1`

Both are accepted.

---

## Installation

### Minimal install for endpoint mode only

If you only use a remote/local vLLM endpoint, you do **not** need `torch` or `transformers` in this environment.

```bash
pip install pydantic
pip install pymupdf   # recommended for PDFs
# or
pip install pdfplumber
```

### Install for local HF mode

```bash
pip install torch transformers accelerate pydantic
pip install pymupdf   # recommended
# or
pip install pdfplumber
```

---

## Quick start

### Local HF inference

```bash
python medace.py \
  -i report.pdf \
  --backend hf \
  -m oss-20b \
  --num-gpus 2
```

### vLLM endpoint inference

```bash
python medace.py \
  -i report.pdf \
  --backend vllm \
  --base-url http://127.0.0.1:8020 \
  -m openai/gpt-oss-120b
```

### Auto backend selection

If `--base-url` is supplied, `--backend auto` selects the endpoint backend automatically.

```bash
python medace.py \
  -i report.pdf \
  --base-url http://127.0.0.1:8020 \
  -m openai/gpt-oss-20b
```

### Backward compatibility

`main.py` still works and now simply delegates to `medace.py`:

```bash
python main.py -i report.pdf --base-url http://127.0.0.1:8020 -m openai/gpt-oss-20b
```

---

## CLI reference

```bash
python medace.py \
  -i report.pdf \
  -o output.json \
  -m oss-20b \
  --backend auto \
  --base-url http://127.0.0.1:8020 \
  --api-key EMPTY \
  --read-timeout 1800 \
  --num-gpus 2 \
  --dtype bfloat16 \
  --input-format pdf \
  --temperature 0.0 \
  --top-p 1.0 \
  --max-new-tokens 32768 \
  --max-retries 3 \
  --verbose
```

### Key arguments

- `--backend {auto,hf,vllm}`: choose inference backend
- `--base-url`: OpenAI-compatible vLLM endpoint base URL
- `--read-timeout`: useful for long endpoint generations on HPC
- `--num-gpus`: HF backend only; budgets across visible GPUs
- `--print-prompt`: prints the schema-derived prompt and exits

---

## IU HPC / vLLM workflow

Your sbatch launcher starts the server first, then this CLI should point at that server.

Typical pattern:

1. Start `gpt-oss-20b` or `gpt-oss-120b` with the sbatch/apptainer server job
2. Wait until the endpoint responds at `/v1/models`
3. Run:

```bash
python medace.py \
  -i /path/to/report.pdf \
  --backend vllm \
  --base-url http://127.0.0.1:8020 \
  -m openai/gpt-oss-120b
```

The served model name must match what the vLLM server exposes.

---

## Output format

The output is a JSON object conforming to `schema.py`.

Example skeleton:

```json
{
  "patient": {
    "patient_name_last": null,
    "patient_name_first": null,
    "date_of_birth": null,
    "mrn": null,
    "sex": null,
    "field_confidence": null
  },
  "nicu_flags": {
    "nicu_genetic_testing": true,
    "nicu_genetic_diagnosis": false,
    "field_confidence": {
      "nicu_genetic_testing": {
        "confidence": "high",
        "interval": "0.95-0.99"
      }
    }
  },
  "test_info": {
    "test_type": "Genome Sequencing",
    "test_name": "Rapid Genome Sequencing",
    "result_available": true,
    "test_declined": false,
    "lab": "IU Diagnostic Genomics",
    "order_date": "2024-08-22",
    "result_date": "2024-08-29",
    "timeframe": "NICU Stay",
    "analysis_type": "Proband Only",
    "secondary_findings": "Opt-OUT",
    "findings_reported": true,
    "field_confidence": {
      "test_type": {
        "confidence": "high",
        "interval": "0.95-0.99"
      }
    }
  },
  "patient_phenotypes": {
    "phenotype_test_date": "2024-08-22",
    "clinical_indication_text": null,
    "hpo_terms": [],
    "field_confidence": null
  },
  "findings": [],
  "interpretation": {
    "result": "Nondiagnostic",
    "reference_genome": "GRCh38/hg38",
    "interpretation_summary": null,
    "field_confidence": null
  },
  "diagnoses": [],
  "extraction_confidence": "high",
  "extraction_notes": null
}
```

---

## Key schema decisions

### Repeatable findings and diagnoses

Instead of creating dozens of hardcoded fields like:

- `genelocus`
- `genelocus_2`
- `genelocus_3`
- ...

the extractor now uses structured lists.

That means:

- easier prompting
- easier validation
- easier comparison logic later
- much less schema churn when the project evolves

### Diagnosis logic

The schema is designed so that:

- VUS-only findings can still appear in `findings`
- but they do **not** automatically become entries in `diagnoses`
- carrier status and ROH stay in `findings`, not `diagnoses`

### Phenotype handling

`patient_phenotypes` is kept separate so you can preserve the report-native phenotype text now and decide later whether to crosswalk those terms to HPO/ICD workflows.

---

## Debugging

### Print the prompt without running a model

```bash
python medace.py -i report.pdf -m oss-20b --print-prompt
```

### If JSON parsing fails repeatedly

The pipeline automatically re-asks with the parse error. If that still fails:

- lower `temperature`
- reduce `max_new_tokens`
- inspect the prompt with `--print-prompt`
- try the endpoint backend with a stronger model

### If validation fails repeatedly

That usually means one of two things:

- the schema is too strict for the report family
- the field description needs to be more explicit

Edit `schema.py`, not the prompt text.

---

## Extending the schema

Everything important lives in `schema.py`.

### Add a new scalar field

```python
class TestInformation(ConfidenceTrackedModel):
    accession_number: Optional[str] = Field(
        None,
        description="Accession number as printed on the report"
    )
```

### Add a new repeated block

```python
class AdditionalSample(ConfidenceTrackedModel):
    sample_type: Optional[str] = Field(None, description="Additional sample type")

class GeneticTestExtraction(StrictModel):
    additional_samples: Optional[List[AdditionalSample]] = Field(
        None,
        description="Additional samples linked to the report"
    )
```

No prompt rewrite is needed.

---

## Recommendation for this project

For IU HPC production runs with `gpt-oss-20b` / `gpt-oss-120b`, prefer:

- `--backend vllm`
- `--base-url http://127.0.0.1:<PORT>`
- `--temperature 0.0`

For local debugging on a single machine, HF mode is still available.
