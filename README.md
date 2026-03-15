# Pediatric Genetic Report Extractor + REDCap Compatibility

Structured extraction for pediatric/NICU genetic testing reports with a schema-driven prompt, Pydantic validation, and optional OpenAI-compatible vLLM endpoint support.

The project now supports **two representations at once**:

1. A **linked, normalized report object** that keeps a test report together with its findings, phenotype note(s), diagnosis note, and diagnoses.
2. A **legacy REDCap-compatible flat layout** that can import from and export back to the current raw repeating-instrument structure.

`main.py` is the canonical CLI. `medace.py` is only a tiny backward-compatible shim.

## Why this structure

The current REDCap export stores these at the same logical level but in separate repeating instruments:

- `genetic_tests` → one row per test, with up to 20 finding slots inside the row
- `patient_phenotypes` → one row per phenotype/HPO note block
- `genetic_diagnoses` → one row per diagnosis-note block, with up to 3 diagnoses inside the row

That flat layout is fine for REDCap, but it loses the **report ↔ phenotype ↔ diagnosis-note ↔ diagnosis** association.

The normalized JSON keeps that association. The flat exporter degrades back to the current REDCap structure when needed.

## Supported inference backends

- **Local HuggingFace** loading (`torch` + `transformers`)
- **External vLLM endpoint** exposing an OpenAI-compatible `/v1/chat/completions` API

Known model aliases:

- `oss-120b` → `openai/gpt-oss-120b`
- `oss-20b` → `openai/gpt-oss-20b`
- `llama3.1-8b` → `meta-llama/Llama-3.1-8B-Instruct`

## Install

```bash
pip install pydantic

# local HF backend only
pip install torch transformers accelerate

# PDF support
pip install pymupdf
# or
pip install pdfplumber

# for deriving REDCap columns from the data dictionary
pip install openpyxl
```

## Main tasks

### 1) Extract from a PDF/TXT report into linked JSON

Local HF:

```bash
python main.py \
  --task extract \
  -i report.pdf \
  -m oss-20b
```

External vLLM endpoint:

```bash
python main.py \
  --task extract \
  -i report.pdf \
  -m openai/gpt-oss-120b \
  --backend vllm \
  --base-url http://127.0.0.1:8020
```

This writes a linked, single-report JSON by default:

```json
{
  "patient": { ... },
  "report_metadata": { ... },
  "test_info": { ... },
  "findings_list": [ ... ],
  "interpretation": { ... },
  "patient_phenotypes": [ ... ],
  "diagnoses": [ ... ],
  "diagnoses_note": "..."
}
```

### 2) Extract and also emit REDCap-compatible rows

Genetics-only flat rows:

```bash
python main.py \
  --task extract \
  -i report.pdf \
  -m oss-20b \
  --redcap-output report.redcap.csv
```

Full current-project column order using the REDCap data dictionary:

```bash
python main.py \
  --task extract \
  -i report.pdf \
  -m oss-20b \
  --data-dictionary "14499V2 Annotated Data Dictionary for CTSI Core 3_9_26.xlsx" \
  --redcap-output report.redcap.csv \
  --emit-base-row
```

If no data dictionary is provided, the flat export falls back to a genetics-focused header.

### 3) Import a raw REDCap export into linked patient-case JSON

```bash
python main.py \
  --task import-redcap \
  -i raw_redcap_export.csv \
  -o linked_cases.json
```

The importer groups rows by `mrn_id`, parses the genetics repeating instruments, keeps the original raw column order, and preserves untouched non-genetics repeating instruments for round-tripping.

### 4) Export linked JSON back to REDCap-compatible flat rows

From linked patient-case JSON:

```bash
python main.py \
  --task export-redcap \
  -i linked_cases.json \
  -o roundtrip.csv
```

Force genetics-only output:

```bash
python main.py \
  --task export-redcap \
  -i linked_cases.json \
  -o genetics_only.csv \
  --genetics-only
```

Use the REDCap data dictionary to rebuild the full current-project header order:

```bash
python main.py \
  --task export-redcap \
  -i linked_cases.json \
  -o roundtrip.csv \
  --data-dictionary "14499V2 Annotated Data Dictionary for CTSI Core 3_9_26.xlsx"
```

### 5) Print / persist the canonical REDCap column order from the data dictionary

```bash
python main.py \
  --task print-redcap-columns \
  --data-dictionary "14499V2 Annotated Data Dictionary for CTSI Core 3_9_26.xlsx" \
  -o redcap_columns.json
```

## Schema highlights

The extraction schema now explicitly covers the project’s genetics-facing pieces:

- `patient`
- `report_metadata`
- `test_info`
- `findings_list`
- `interpretation`
- `patient_phenotypes`
- `diagnoses`
- `diagnoses_note`

The REDCap-linked case model additionally preserves:

- `legacy_base_row`
- `legacy_other_repeat_rows`
- `redcap_column_order`
- unlinked phenotype rows and diagnosis-note rows when the original flat data is too ambiguous to attach safely

## Import/export behavior

### Linking when importing legacy flat REDCap rows

The importer links `patient_phenotypes` and `genetic_diagnoses` rows to `genetic_tests` rows when possible:

- if there is exactly one report, it links everything to that report
- otherwise it uses conservative date/type/locus heuristics
- if the match is ambiguous, it leaves the rows in `unlinked_patient_phenotypes` or `unlinked_diagnosis_groups`

That keeps the normalized JSON honest instead of inventing a linkage that REDCap never actually stored.

### Round-tripping legacy rows

When rows are imported from REDCap, the original row dictionaries are preserved on the linked objects. On export, those preserved rows are reused as the starting point whenever possible, so hidden/calculated fields and unrelated repeat instruments are not unnecessarily discarded.

## Prompt / validation loop

The extraction pipeline is still schema-driven:

1. Build prompt instructions from `schema.py`
2. Run the model
3. Parse JSON
4. Validate with Pydantic
5. Re-ask with validation errors if needed

So `schema.py` remains the single source of truth for the extraction object.

## Important semantics baked into the prompt/schema

- VUS-only, carrier-only, ROH/AOH-only, and negative results should **not** become diagnoses
- diagnoses should stay linked to the report that supports them
- phenotype/HPO note blocks should stay linked to the originating report when known
- a single report may have multiple findings and multiple diagnoses

## Notes on limits

- `genetic_tests` currently supports up to **20** findings per row, matching REDCap
- `genetic_diagnoses` currently supports up to **3** diagnoses per legacy row; the flat exporter chunks larger diagnosis lists into multiple rows
- if you export a stand-alone extracted report without the full case context, fields that depend on NICU/base-row context stay blank unless you provide that context yourself

## Backward compatibility

Existing scripts that still call:

```bash
python medace.py ...
```

will continue to work, because `medace.py` now simply forwards to `main.py`.
