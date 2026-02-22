# Pediatric Genetic Test Report Extractor

Structured data extraction from pediatric genetic test reports using local HuggingFace models with schema-driven prompting and self-correcting validation. Designed for the Indiana University NICU Genomics REDCap project (PID 31291).

## What It Does

Given a genetic test report (PDF or plain text), this tool:

1. Parses the report (auto-detects PDF vs text)
2. Generates a prompt from the Pydantic schema (so you never write prompts by hand)
3. Sends it to a locally-loaded LLM on your GPUs
4. Validates the output against the schema with Pydantic
5. If validation fails, feeds the errors back to the LLM and re-asks (up to N retries)
6. Writes a clean JSON file

The key idea: **`schema.py` is the single source of truth.** You define what to extract there — field names, types, constraints, enums — and the pipeline auto-generates the LLM prompt, validates the output, and handles retries. No prompt editing needed.

## Project Structure

```
├── main.py       # CLI entry point: load model, run extraction, write JSON
├── schema.py     # Pydantic schema: defines all fields, types, validators
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU(s) with sufficient VRAM
- HuggingFace model access (may require `huggingface-cli login` for gated models)

### Install Dependencies

```bash
pip install torch transformers accelerate pydantic

# for PDF support (install at least one):
pip install pymupdf       # recommended, fast C backend
# or
pip install pdfplumber    # pure-python alternative
```

## Supported Models

| Alias | HuggingFace Model ID | Notes |
|---|---|---|
| `oss-120b` | `openai/gpt-oss-120b` | MoE, strongest extraction quality |
| `oss-20b` | `openai/gpt-oss-20b` | MoE, good balance of speed and quality |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | Dense, fastest inference |

You can also pass any full HuggingFace model ID directly (e.g., `-m mistralai/Mistral-7B-Instruct-v0.3`).

## Quick Start

### Basic usage (plain text report)

```bash
python main.py -i report.txt -m oss-120b --num-gpus 4
```

### PDF report

```bash
python main.py -i report.pdf -m oss-20b --num-gpus 2
```

### Specify output path

```bash
python main.py -i report.pdf -m llama3.1-8b -o results/patient_001.json
```

### All options

```bash
python main.py \
  -i report.pdf \              # input file (required)
  -m oss-120b \                # model alias or full HF ID (required)
  --num-gpus 4 \               # number of GPUs (default: 1)
  --dtype bfloat16 \           # model dtype: auto, bfloat16, float16 (default: auto)
  --input-format pdf \         # force input format: text or pdf (default: auto-detect)
  --temperature 0.3 \          # sampling temperature (default: 0.3)
  --top-p 0.9 \                # nucleus sampling top-p (default: 0.9)
  --max-new-tokens 32768 \     # max tokens to generate (default: 32768)
  --max-retries 3 \            # validation retry attempts (default: 3)
  --verbose \                  # debug logging
  -o output.json               # output path (default: <input>.extracted.json)
```

## Output Format

The output is a JSON file matching the schema defined in `schema.py`. Example:

```json
{
  "patient": {
    "patient_name_last": "Smith",
    "patient_name_first": "Jane",
    "date_of_birth": "2024-01-15",
    "mrn": "12345678",
    "sex": "Female",
    "field_confidence": {
      "patient_name_last": {"confidence": "high", "interval": "0.95-0.99"},
      "patient_name_first": {"confidence": "high", "interval": "0.95-0.99"},
      "date_of_birth": {"confidence": "high", "interval": "0.95-0.99"},
      "mrn": {"confidence": "high", "interval": "0.90-0.99"},
      "sex": {"confidence": "high", "interval": "0.95-0.99"}
    }
  },
  "test_info": {
    "test_type": "Genome Sequencing",
    "test_type_other": null,
    "test_name": "Rapid Trio Genome Sequencing",
    "reanalysis": false,
    "lab": "GeneDx",
    "lab_other": null,
    "order_date": "2024-03-01",
    "result_date": "2024-03-20",
    "analysis_type": "Trio",
    "secondary_findings": "Opt-IN",
    "findings_reported": true,
    "field_confidence": {
      "test_type": {"confidence": "high", "interval": "0.95-0.99"},
      "lab": {"confidence": "high", "interval": "0.95-0.99"},
      "result_date": {"confidence": "high", "interval": "0.95-0.99"}
    }
  },
  "findings": [
    {
      "gene_locus": "SCN1A",
      "variant": "c.1234A>G",
      "protein": "p.Thr412Ala",
      "transcript": "NM_001165963.4",
      "dosage": null,
      "roh": false,
      "classification": "Pathogenic",
      "zygosity": "Heterozygous",
      "inheritance": "Autosomal Dominant",
      "segregation": "De Novo",
      "finding_class": "Primary",
      "additional_info": null,
      "field_confidence": {
        "gene_locus": {"confidence": "high", "interval": "0.95-0.99"},
        "variant": {"confidence": "high", "interval": "0.95-0.99"},
        "classification": {"confidence": "high", "interval": "0.95-0.99"},
        "segregation": {"confidence": "moderate", "interval": "0.75-0.90", "comment": "Stated as de novo based on trio analysis"}
      }
    }
  ],
  "interpretation": {
    "result": "Diagnostic",
    "reference_genome": "GRCh38/hg38",
    "interpretation_summary": "A pathogenic variant in SCN1A was identified, consistent with Dravet syndrome.",
    "field_confidence": {
      "result": {"confidence": "high", "interval": "0.95-0.99"},
      "reference_genome": {"confidence": "high", "interval": "0.90-0.99"}
    }
  },
  "extraction_confidence": "high",
  "extraction_notes": null
}
```

Fields the LLM cannot determine from the report will be `null`.

## Customizing the Schema

All extraction fields live in `schema.py`. The pipeline reads the schema at runtime, so changes take effect immediately — no prompt editing required.

### Adding a new field

Add it to the relevant sub-model with a `Field(description=...)`:

```python
class PatientDemographics(BaseModel):
    # ... existing fields ...
    study_id: Optional[str] = Field(None,
        description="Study ID if printed on the report")
```

The `description` string is what the LLM sees in its instructions. Make it specific.

### Supported field types

`schema.py` includes all field types used by the REDCap codebook. Here's a summary:

| Type | Example | Validation |
|---|---|---|
| **String** | `Field(None, description="...")` | Optional min/max length, regex pattern |
| **Integer** | `Field(None, ge=0, le=120, ...)` | Range via `ge`/`le` |
| **Float** | `Field(None, ge=0.0, le=100.0, ...)` | Range via `ge`/`le` |
| **Boolean** | `Field(None, description="...")` | True/False |
| **Enum** | `Field(None, description="...")` | Restricted to enum values |
| **Date string** | `Field(None, description="...YYYY-MM-DD...")` | `@field_validator` |
| **Time string** | `Field(None, description="...HH:MM...")` | `@field_validator` |
| **Dict** | `Optional[Dict[str, str]]` | Key-value pairs |
| **List** | `Optional[List[str]]` | List of items |
| **Cross-field** | `@model_validator(mode="after")` | Compare multiple fields |

### Adding a new enum

Define it, then use it as a field type:

```python
class TestTimeframe(str, Enum):
    nicu_stay = "NICU Stay"
    pre_nicu = "Pre-NICU Admission/OSH"
    prenatal = "Prenatal"
    post_nicu = "Post-NICU Discharge"

class TestInformation(BaseModel):
    timeframe: Optional[TestTimeframe] = Field(None,
        description="When the test was ordered relative to NICU stay")
```

The format instructions will automatically list the allowed values.

### Adding a new sub-model

Define a new `BaseModel` subclass, then add it as a field on `GeneticTestExtraction`:

```python
class GeneticDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnosis_name: Optional[str] = Field(None, description="Name of genetic diagnosis")
    associated_gene: Optional[str] = Field(None, description="Associated gene or locus")

class GeneticTestExtraction(BaseModel):
    # ... existing fields ...
    diagnosis: GeneticDiagnosis = Field(
        default_factory=GeneticDiagnosis, description="Genetic diagnosis if stated on report")
```

## How the Re-Ask Loop Works

```
report → [build prompt from schema] → LLM → [parse JSON] → [Pydantic validate]
                                        ↑                           |
                                        |     validation error      |
                                        +←——— [feed errors back] ←——+
```

If the LLM output fails JSON parsing or Pydantic validation, the errors are appended as a follow-up user message so the model can self-correct. This runs up to `--max-retries` times (default: 3). The Pydantic error messages are specific enough (e.g., "age must be >= 0 and <= 120, got 200") that the model usually fixes them in one retry.

## Troubleshooting

**Out of memory**: Lower `--num-gpus` or use a smaller model. The `oss-*` models are MoE architectures and are more memory-efficient than their parameter count suggests.

**Model download fails**: Run `huggingface-cli login` and ensure you have access to the model. Some models (e.g., Llama) require accepting a license on the HF model page.

**PDF extraction is empty**: Try installing `pdfplumber` as a fallback (`pip install pdfplumber`). Some scanned PDFs may need OCR preprocessing — this tool handles text-based PDFs only.

**Validation keeps failing**: Check `--verbose` output. If the model consistently fails on a field, consider relaxing the constraint in `schema.py` or providing a more specific `description`.

## License

MIT
