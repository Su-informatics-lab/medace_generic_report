"""Pydantic schema for structured pediatric genetic test report extraction.

SINGLE SOURCE OF TRUTH — edit this file to change what gets extracted.
main.py imports `get_extraction_model()` and `build_format_instructions()`
and never hardcodes field names.

TARGET: Genetic test report PDFs and text for NICU/pediatric patients.
Extracts only what is present on the report itself.

REDCap mapping notes:
  - PatientDemographics  → demographics_birth_history_and_admission (subset)
  - TestInformation      → genetic_tests: test-level fields
  - GeneticFinding       → genetic_tests: finding1..finding20 (as List)
  - TestInterpretation   → genetic_tests: interpretation section
  - NOT extracted: NICU Course, NICU Diagnoses, Follow-up Patient Services,
    Vital Status, Birth History (chart-abstracted, not on reports)

HOW TO ADD FIELDS:
  1. Add a field to the appropriate sub-model (or create a new one).
  2. Give it a `description=` — this auto-populates the LLM prompt.
  3. Add validators (range, enum, regex) — these auto-trigger on re-ask.
  4. That's it.  The prompt and validation pipeline adapt automatically.

CONFIDENCE DESIGN:
  Every sub-model carries a `field_confidence` dict mapping field names to
  {confidence, interval, comment}.  The LLM self-reports extraction quality
  per field, enabling downstream QC without doubling field count.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums — constrained vocabularies aligned with REDCap codebook
# ---------------------------------------------------------------------------


class Sex(str, Enum):
    """REDCap [sex]: 1=Male, 2=Female, 95=Other."""

    male = "Male"
    female = "Female"
    other = "Other"


class TestType(str, Enum):
    """REDCap [type]: genetic test types."""

    fish = "FISH"
    karyotype = "Karyotype"
    microarray = "Microarray"
    targeted_variant = "Targeted Variant"
    single_gene_seq = "Single Gene (Sequencing, Del/Dup)"
    single_gene_repeat = "Single Gene (Repeat Analysis)"
    panel_or_exome_slice = "Panel or Exome Slice"
    exome = "Exome Sequencing"
    genome = "Genome Sequencing"
    mitochondrial = "Mitochondrial DNA"
    imprinting_methylation = "Imprinting/Methylation"
    other = "Other"
    not_reported = "Not Reported"


class TestLab(str, Enum):
    """REDCap [lab]: testing laboratories."""

    ambry = "Ambry"
    baylor = "Baylor"
    blueprint = "Blueprint"
    egl_eurofins = "EGL/Eurofins"
    genedx = "GeneDx"
    integrated_genetics = "Integrated Genetics"
    invitae = "Invitae"
    iu_cytogenetics = "IU Cytogenetics"
    iu_molecular = "IU Molecular"
    iu_ngs = "IU NGS"
    labcorp = "LabCorp"
    prevention = "Prevention"
    quest = "Quest"
    iu_diagnostic_genomics = "IU Diagnostic Genomics"
    other = "Other"
    not_reported = "Not Reported"


class AnalysisType(str, Enum):
    """REDCap [analysis]: how the sample was analyzed."""

    proband_only = "Proband Only"
    proband_with_segregation = "Proband Only w/ Segregation"
    duo = "Duo"
    trio = "Trio"
    other = "Other"
    not_reported = "Not Reported"


class SecondaryFindings(str, Enum):
    """REDCap [consent_second]: secondary findings consent."""

    opt_in = "Opt-IN"
    opt_out = "Opt-OUT"
    not_reported = "Not Reported or N/A"


class LabClassification(str, Enum):
    """REDCap [labclass]: ACMG 5-tier + extras."""

    pathogenic = "Pathogenic"
    likely_pathogenic = "Likely Pathogenic"
    vus = "VUS"
    abnormal_nos = "Abnormal NOS"
    likely_benign = "Likely Benign"
    benign = "Benign"
    normal_nos = "Normal NOS"
    other = "Other"
    not_reported = "Not Reported"


class Dosage(str, Enum):
    """REDCap [dose]: copy number / dosage."""

    deletion = "Deletion"
    duplication = "Duplication"
    other = "Other"
    not_reported = "Not Reported"


class Zygosity(str, Enum):
    """REDCap [zygosity]."""

    heterozygous = "Heterozygous"
    homozygous = "Homozygous"
    hemizygous = "Hemizygous"
    other = "Other"
    not_reported = "Not Reported"


class InheritancePattern(str, Enum):
    """REDCap [inheritance]."""

    autosomal_dominant = "Autosomal Dominant"
    autosomal_recessive = "Autosomal Recessive"
    x_linked = "X-Linked"
    ad_or_ar = "Autosomal Dominant OR Recessive"
    other = "Other"
    not_reported = "Not Reported"


class Segregation(str, Enum):
    """REDCap [segregation]: parental origin."""

    de_novo = "De Novo"
    maternal = "Maternal"
    paternal = "Paternal"
    not_maternal = "NOT Maternal"
    not_paternal = "NOT Paternal"
    maternal_and_paternal = "Maternal and Paternal"
    other = "Other"
    not_reported = "Not Reported"


class FindingClassification(str, Enum):
    """REDCap [findingclass]: clinical significance category."""

    primary = "Primary"
    secondary = "Secondary"
    incidental = "Incidental"
    unknown = "Unknown"


class TestInterpretationResult(str, Enum):
    """REDCap [testdx]: overall test interpretation."""

    nondiagnostic = "Nondiagnostic"
    diagnostic = "Diagnostic"
    not_reported = "Not Reported"


class ReferenceGenome(str, Enum):
    """REDCap [refseq]: genome build."""

    grch37_hg19 = "GRCh37/hg19"
    grch38_hg38 = "GRCh38/hg38"
    other = "Other"
    not_reported = "Not Reported"


# ---------------------------------------------------------------------------
# Confidence tracking — shared across all sub-models
# ---------------------------------------------------------------------------


class FieldConfidence(BaseModel):
    """Per-field extraction confidence reported by the LLM."""

    model_config = ConfigDict(extra="forbid")

    confidence: str = Field(
        description="Self-assessed confidence: high, moderate, or low"
    )
    interval: Optional[str] = Field(
        None, description="Estimated probability interval, e.g. '0.85-0.95'"
    )
    comment: Optional[str] = Field(
        None, description="Why confidence is not high, or notes on ambiguity"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        if v not in ("high", "moderate", "low"):
            raise ValueError("confidence must be high, moderate, or low")
        return v


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class PatientDemographics(BaseModel):
    """Patient identifiers as printed on the genetic test report.

    REDCap mapping:
      patient_name_last/first → [name] (Last, First format)
      date_of_birth           → [dob]
      mrn                     → [mrn_id]
      sex                     → [sex]
    """

    model_config = ConfigDict(extra="forbid")

    patient_name_last: Optional[str] = Field(
        None, description="Patient last (family) name as printed on report"
    )
    patient_name_first: Optional[str] = Field(
        None, description="Patient first (given) name as printed on report"
    )
    date_of_birth: Optional[str] = Field(
        None, description="Date of birth in YYYY-MM-DD format"
    )
    mrn: Optional[str] = Field(None, description="Medical record number")
    sex: Optional[Sex] = Field(None, description="Patient sex: Male, Female, or Other")
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description=(
            "Per-field confidence for each demographics field extracted. "
            "Keys are field names (e.g. 'patient_name_last', 'mrn')."
        ),
    )

    @field_validator("date_of_birth")
    @classmethod
    def validate_dob_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"date_of_birth must be YYYY-MM-DD, got '{v}'")
        return v


class TestInformation(BaseModel):
    """Test-level metadata from the report header/footer.

    REDCap mapping:
      test_type          → [type]
      test_name          → [testname]
      reanalysis         → [reanalysis]
      lab                → [lab]
      lab_other          → [lab_other]
      order_date         → [order]
      result_date        → [returndate]
      analysis_type      → [analysis]
      secondary_findings → [consent_second]
      findings_reported  → [findings]
    """

    model_config = ConfigDict(extra="forbid")

    test_type: Optional[TestType] = Field(
        None, description="Type of genetic test performed"
    )
    test_type_other: Optional[str] = Field(
        None, description="If test_type is Other, specify here"
    )
    test_name: Optional[str] = Field(
        None,
        description="Full test name as printed on report (include speed, e.g. 'Rapid Trio Genome Sequencing')",
    )
    reanalysis: Optional[bool] = Field(
        None, description="Whether this is a reanalysis of prior data"
    )
    lab: Optional[TestLab] = Field(None, description="Testing laboratory")
    lab_other: Optional[str] = Field(None, description="If lab is Other, specify here")
    order_date: Optional[str] = Field(
        None, description="Test order/collection date in YYYY-MM-DD format"
    )
    result_date: Optional[str] = Field(
        None, description="Result/report date in YYYY-MM-DD format"
    )
    analysis_type: Optional[AnalysisType] = Field(
        None, description="Analysis type: Proband Only, Duo, Trio, etc."
    )
    secondary_findings: Optional[SecondaryFindings] = Field(
        None, description="Secondary findings consent: Opt-IN, Opt-OUT, or N/A"
    )
    findings_reported: Optional[bool] = Field(
        None, description="Whether any findings were reported on this test"
    )
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for each test information field.",
    )

    @field_validator("order_date", "result_date")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD, got '{v}'")
        return v

    @model_validator(mode="after")
    def check_other_fields(self) -> "TestInformation":
        """If type is Other, test_type_other should be provided (warn, don't block)."""
        # Soft check — LLM may not always have this info
        return self


class GeneticFinding(BaseModel):
    """A single genetic finding/variant from the report.

    Corresponds to one of REDCap's finding1..finding20 blocks.
    Use a List[GeneticFinding] to capture all findings.

    REDCap mapping:
      gene_locus    → [genelocus]
      variant       → [dna] (HGVS c. or genomic coordinates)
      protein       → [protein] (HGVS p.)
      transcript    → [transcript]
      dosage        → [dose]
      roh           → [roh]
      classification→ [labclass]
      zygosity      → [zygosity]
      inheritance   → [inheritance]
      segregation   → [segregation]
      finding_class → [findingclass]
      additional    → [findingcomments]
    """

    model_config = ConfigDict(extra="forbid")

    gene_locus: Optional[str] = Field(
        None,
        description="Gene symbol, chromosomal locus, or chromosome (e.g. 'BRCA1', '17p13.1', 'chr21')",
    )
    variant: Optional[str] = Field(
        None,
        description="Variant in HGVS DNA nomenclature or genomic coordinates (e.g. 'c.123A>G' or 'chr1:12345-67890')",
    )
    protein: Optional[str] = Field(
        None,
        description="Protein change in HGVS nomenclature (e.g. 'p.Thr123Ala')",
    )
    transcript: Optional[str] = Field(
        None,
        description="Reference transcript (e.g. 'NM_012345.6')",
    )
    dosage: Optional[Dosage] = Field(
        None,
        description="Copy number/dosage: Deletion, Duplication, Other, or Not Reported",
    )
    roh: Optional[bool] = Field(
        None,
        description="Whether this is within a region of homozygosity (ROH)",
    )
    classification: Optional[LabClassification] = Field(
        None,
        description="ACMG classification: Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign, etc.",
    )
    zygosity: Optional[Zygosity] = Field(
        None,
        description="Zygosity: Heterozygous, Homozygous, Hemizygous",
    )
    inheritance: Optional[InheritancePattern] = Field(
        None,
        description="Mode of inheritance: AD, AR, X-Linked, etc.",
    )
    segregation: Optional[Segregation] = Field(
        None,
        description="Parental origin: De Novo, Maternal, Paternal, etc.",
    )
    finding_class: Optional[FindingClassification] = Field(
        None,
        description="Finding classification: Primary, Secondary, Incidental, Unknown",
    )
    additional_info: Optional[str] = Field(
        None,
        description="Any additional notes about this finding",
    )
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for each finding field.",
    )


class TestInterpretation(BaseModel):
    """Overall test interpretation section.

    REDCap mapping:
      result          → [testdx]
      reference_genome→ [refseq]
    """

    model_config = ConfigDict(extra="forbid")

    result: Optional[TestInterpretationResult] = Field(
        None,
        description="Overall test interpretation: Diagnostic, Nondiagnostic, or Not Reported",
    )
    reference_genome: Optional[ReferenceGenome] = Field(
        None,
        description="Reference genome build: GRCh37/hg19, GRCh38/hg38, Other, or Not Reported",
    )
    interpretation_summary: Optional[str] = Field(
        None,
        description="Brief summary of the lab's interpretation/conclusion as stated in the report",
    )
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for interpretation fields.",
    )


# ---------------------------------------------------------------------------
# Top-level extraction model
# ---------------------------------------------------------------------------


class GeneticTestExtraction(BaseModel):
    """Structured extraction from a single pediatric genetic test report PDF.

    Designed for the Indiana University NICU Genomics REDCap project (PID 31291).
    Extracts only fields present on the genetic test report itself — clinical
    context (NICU course, birth history, follow-up, vital status) is excluded
    as it requires chart abstraction.
    """

    model_config = ConfigDict(extra="forbid")

    patient: PatientDemographics = Field(
        default_factory=PatientDemographics,
        description="Patient demographics as printed on the report",
    )
    test_info: TestInformation = Field(
        default_factory=TestInformation,
        description="Test-level metadata (type, lab, dates, analysis)",
    )
    findings: Optional[List[GeneticFinding]] = Field(
        None,
        description=(
            "List of all genetic findings/variants reported. "
            "One entry per finding. Empty list or null if no findings reported."
        ),
    )
    interpretation: TestInterpretation = Field(
        default_factory=TestInterpretation,
        description="Overall test interpretation and reference genome",
    )
    extraction_confidence: Optional[str] = Field(
        None,
        description="Overall extraction confidence: high, moderate, or low",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Caveats, ambiguities, or unresolvable information from the report",
    )

    @model_validator(mode="after")
    def confidence_is_valid(self) -> "GeneticTestExtraction":
        if self.extraction_confidence and self.extraction_confidence not in (
            "high",
            "moderate",
            "low",
        ):
            raise ValueError("extraction_confidence must be high, moderate, or low")
        return self

    @model_validator(mode="after")
    def findings_consistent_with_test_info(self) -> "GeneticTestExtraction":
        """Cross-validate: if test_info says no findings, findings list should be empty."""
        if self.test_info.findings_reported is False and self.findings:
            raise ValueError(
                "test_info.findings_reported is False but findings list is non-empty. "
                "Set findings_reported=True or clear the findings list."
            )
        if self.test_info.findings_reported is True and not self.findings:
            raise ValueError(
                "test_info.findings_reported is True but findings list is empty. "
                "Add findings or set findings_reported=False."
            )
        return self


# ---------------------------------------------------------------------------
# Public API — main.py imports only these
# ---------------------------------------------------------------------------


def get_extraction_model() -> Type[BaseModel]:
    """Return the top-level extraction model class.

    main.py calls this so it never hardcodes the schema class name.
    To swap the entire schema, change this return value.
    """
    return GeneticTestExtraction


def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap Optional[X] → (X, True); non-optional → (annotation, False)."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union and type(None) in args:
        inner = [a for a in args if a is not type(None)]
        if len(inner) == 1:
            return inner[0], True
    return annotation, False


def describe_field(name: str, info, indent: int = 0) -> List[str]:
    """Recursively describe a Pydantic field for prompt instructions."""
    prefix = "  " * indent
    lines: List[str] = []
    desc = info.description or ""

    raw_ann = info.annotation
    inner, is_optional = unwrap_optional(raw_ann)

    # unwrap List[X]
    list_origin = get_origin(inner)
    if list_origin is list:
        list_args = get_args(inner)
        opt_tag = " (optional, list)" if is_optional else " (list)"
        if (
            list_args
            and isinstance(list_args[0], type)
            and issubclass(list_args[0], BaseModel)
        ):
            item_model = list_args[0]
            lines.append(f"{prefix}- {name}{opt_tag}: {desc}")
            lines.append(f"{prefix}  Each item in the list has these fields:")
            for sub_name, sub_info in item_model.model_fields.items():
                lines.extend(describe_field(sub_name, sub_info, indent + 2))
            return lines
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}")
        return lines

    opt_tag = " (optional)" if is_optional else ""

    # enum
    if isinstance(inner, type) and issubclass(inner, Enum):
        allowed = [e.value for e in inner]
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}. Allowed values: {allowed}")
        return lines

    # nested BaseModel (including Dict[str, FieldConfidence])
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}")
        for sub_name, sub_info in inner.model_fields.items():
            lines.extend(describe_field(sub_name, sub_info, indent + 1))
        return lines

    # Dict type — describe key/value
    if get_origin(inner) is dict:
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}")
        dict_args = get_args(inner)
        if dict_args and len(dict_args) == 2:
            val_type = dict_args[1]
            if isinstance(val_type, type) and issubclass(val_type, BaseModel):
                lines.append(f"{prefix}  Each value is an object with:")
                for sub_name, sub_info in val_type.model_fields.items():
                    lines.extend(describe_field(sub_name, sub_info, indent + 2))
        return lines

    # numeric constraints from metadata
    constraints = []
    if info.metadata:
        for m in info.metadata:
            if hasattr(m, "ge"):
                constraints.append(f">= {m.ge}")
            if hasattr(m, "le"):
                constraints.append(f"<= {m.le}")
    c_str = f" [{', '.join(constraints)}]" if constraints else ""

    lines.append(f"{prefix}- {name}{opt_tag}: {desc}{c_str}")
    return lines


def build_format_instructions(model: Optional[Type[BaseModel]] = None) -> str:
    """Walk the Pydantic schema and emit LLM-friendly format instructions.

    If no model is passed, uses get_extraction_model().
    """
    if model is None:
        model = get_extraction_model()

    lines = [
        "Respond with a single JSON object conforming to this schema:",
        "",
        f"Root model: {model.__name__}",
        "",
    ]
    for name, info in model.model_fields.items():
        lines.extend(describe_field(name, info, indent=0))
    lines.append("")
    lines.append(
        "Use null for any field whose value cannot be determined from the report."
    )
    lines.append(
        "Do not invent information. Extract only what is explicitly stated or clearly implied."
    )
    lines.append(
        "For EVERY sub-model, populate field_confidence with an entry for each "
        "field you extracted (not null fields). Each entry needs confidence "
        "(high/moderate/low), an interval (e.g. '0.90-0.99'), and an optional comment."
    )
    return "\n".join(lines)
