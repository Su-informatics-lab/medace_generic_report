"""Schema and mapping utilities for pediatric/NICU genetic report extraction.

The design now has two complementary representations:

1) A *linked, normalized* representation that keeps a genetic report together with
   its findings, phenotype note(s), diagnosis note, and diagnosis objects.
2) A *legacy REDCap-compatible* view that can round-trip the current flat export
   layout where `genetic_tests`, `patient_phenotypes`, and `genetic_diagnoses`
   live as separate repeating instruments.

`get_extraction_model()` returns the single-report extraction model used by the
LLM pipeline. `get_case_model()` returns the patient-level linked model used for
REDCap import/export.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Small helpers shared by extraction and REDCap import/export
# ---------------------------------------------------------------------------


DATE_FORMAT_HINT = "Use YYYY-MM-DD when a date can be determined."


def normalize_date(value: Any) -> Optional[str]:
    """Normalize common REDCap/report date strings to ISO YYYY-MM-DD.

    Accepts:
    - YYYY-MM-DD
    - M/D/YYYY
    - MM/DD/YYYY
    - M/D/YY
    - MM/DD/YY

    Returns None for blank values.
    Raises ValueError for malformed non-blank values.
    """

    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()

    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue

    # Handle single-digit month/day variants explicitly.
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 3 and all(part.strip() for part in parts):
            month, day, year = [part.strip() for part in parts]
            if len(year) == 2:
                year_int = int(year)
                year = f"20{year}" if year_int <= 68 else f"19{year}"
            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid date '{text}'") from exc

    raise ValueError(f"Date must be YYYY-MM-DD or M/D/YY style, got '{text}'")


def split_name(name: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Split a REDCap-style patient name.

    Prefers 'Last, First' when a comma is present; otherwise uses the first token
    as the first name and the final token as the last name when reasonable.
    """

    if not name:
        return None, None
    text = str(name).strip()
    if not text:
        return None, None
    if "," in text:
        last, first = text.split(",", 1)
        last = last.strip() or None
        first = first.strip() or None
        return last, first
    parts = text.split()
    if len(parts) == 1:
        return None, parts[0]
    return parts[-1], " ".join(parts[:-1])


def compose_name(
    name: Optional[str],
    patient_name_last: Optional[str],
    patient_name_first: Optional[str],
) -> Optional[str]:
    if name:
        return name
    if patient_name_last and patient_name_first:
        return f"{patient_name_last}, {patient_name_first}"
    if patient_name_last:
        return patient_name_last
    if patient_name_first:
        return patient_name_first
    return None


# ---------------------------------------------------------------------------
# Enums constrained to project-compatible vocabularies
# ---------------------------------------------------------------------------


class Sex(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"


class TestType(str, Enum):
    fish = "FISH"
    karyotype = "Karyotype"
    microarray = "Microarray"
    targeted_variant = "Targeted Variant"
    single_gene_seq_del_dup = "Single Gene (Sequencing, Del/Dup)"
    single_gene_repeat = "Single Gene (Repeat Analysis)"
    panel_or_exome_slice = "Panel or Exome Slice"
    exome = "Exome Sequencing"
    genome = "Genome Sequencing"
    mitochondrial = "Mitochondrial DNA"
    imprinting_methylation = "Imprinting/Methylation"
    other = "Other"
    not_reported = "Not Reported/Unknown"


class TestLab(str, Enum):
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
    not_reported = "Not Reported/Unknown"


class AnalysisType(str, Enum):
    proband_only = "Proband Only"
    proband_with_segregation = "Proband Only w/ Segregation"
    duo = "Duo"
    trio = "Trio"
    other = "Other"
    not_reported = "Not Reported/Unknown"


class SecondaryFindings(str, Enum):
    opt_in = "Opt-IN"
    opt_out = "Opt-OUT"
    not_reported = "Not Reported or N/A"


class LabClassification(str, Enum):
    pathogenic = "Pathogenic"
    likely_pathogenic = "Likely Pathogenic"
    vus = "Variant of Uncertain Significance (VUS)"
    abnormal_nos = "Abnormal NOS"
    likely_benign = "Likely Benign"
    benign = "Benign"
    normal_nos = "Normal NOS"
    other = "Other"
    not_reported = "Not Reported/Not Applicable"


class Dosage(str, Enum):
    x0 = "x0"
    x1 = "x1"
    x2 = "x2"
    x3 = "x3"
    x4 = "x4"
    copy_loss_nos = "Copy Loss NOS"
    copy_gain_nos = "Copy Gain NOS"
    deletion = "Deletion"
    duplication = "Duplication"
    other = "Other"
    not_reported = "Not Reported"


class Zygosity(str, Enum):
    heterozygous = "Heterozygous"
    homozygous = "Homozygous"
    hemizygous = "Hemizygous"
    other = "Other"
    not_reported = "Not Reported"


class InheritancePattern(str, Enum):
    autosomal_dominant = "Autosomal Dominant"
    autosomal_recessive = "Autosomal Recessive"
    x_linked = "X-Linked"
    ad_or_ar = "Autosomal Dominant OR Recessive"
    other = "Other"
    not_reported = "Not Reported"


class Segregation(str, Enum):
    de_novo = "De Novo"
    maternal = "Maternal"
    paternal = "Paternal"
    not_maternal = "NOT Maternal"
    not_paternal = "NOT Paternal"
    maternal_and_paternal = "Maternal and Paternal"
    other = "Other"
    not_reported = "Not Reported/Not Applicable"


class FindingClassification(str, Enum):
    primary = "Primary"
    secondary = "Secondary"
    incidental = "Incidental"
    unknown = "Unknown"


class TestInterpretationResult(str, Enum):
    nondiagnostic = "Nondiagnostic"
    diagnostic = "Diagnostic"
    not_reported = "Not Reported/Unknown"


class ReferenceGenome(str, Enum):
    grch37_hg19 = "GRCh37/hg19"
    grch38_hg38 = "GRCh38/hg38"
    other = "Other"
    not_reported = "Not Reported/Unknown"


class TestTimeframe(str, Enum):
    nicu_stay = "NICU Stay"
    pre_nicu_osh = "Pre-NICU Admission/OSH"
    prenatal = "Prenatal"
    post_nicu_discharge = "Post-NICU Discharge"
    postmortem = "Postmortem"
    other_unknown = "Other/Unknown"


class DiagnosisStudyPeriod(str, Enum):
    pre_nicu_stay = "Pre-NICU Stay"
    nicu_stay = "NICU Stay"
    year1_post_nicu = "Year 1 Post-NICU Stay"
    year2_post_nicu = "Year 2 Post-NICU Stay"
    other = "Other"
    unknown = "Unknown"


class NICUDiagnosisType(str, Enum):
    none_unknown = "None/Unknown"
    cytogenetic = "Cytogenetic"
    molecular = "Molecular Genetic"
    two_or_more = "Two or More Diagnoses"
    not_reported = "Not Reported"


# ---------------------------------------------------------------------------
# Confidence tracking shared by extraction-facing models
# ---------------------------------------------------------------------------


class FieldConfidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confidence: str = Field(
        description="Self-assessed confidence: high, moderate, or low"
    )
    interval: Optional[str] = Field(
        None, description="Estimated probability interval, e.g. '0.85-0.95'"
    )
    comment: Optional[str] = Field(
        None, description="Why confidence is not high, or important ambiguity notes"
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: str) -> str:
        if value not in {"high", "moderate", "low"}:
            raise ValueError("confidence must be high, moderate, or low")
        return value


# ---------------------------------------------------------------------------
# Core extraction-facing submodels
# ---------------------------------------------------------------------------


class PatientDemographics(BaseModel):
    """Patient identifiers used across extraction and REDCap compatibility."""

    model_config = ConfigDict(extra="forbid")

    mrn_id: Optional[str] = Field(None, description="Medical record number (MRN)")
    study_id: Optional[str] = Field(
        None, description="Study ID when present in the source context"
    )
    name: Optional[str] = Field(
        None,
        description="Full patient name exactly as printed, typically 'Last, First' when visible",
    )
    patient_name_last: Optional[str] = Field(
        None, description="Patient last/family name if it can be separated"
    )
    patient_name_first: Optional[str] = Field(
        None, description="Patient first/given name if it can be separated"
    )
    dob: Optional[str] = Field(None, description=f"Date of birth. {DATE_FORMAT_HINT}")
    sex: Optional[Sex] = Field(None, description="Patient sex")
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for non-null patient fields.",
    )

    @field_validator("dob")
    @classmethod
    def validate_dob(cls, value: Optional[str]) -> Optional[str]:
        return normalize_date(value)

    @model_validator(mode="after")
    def sync_names(self) -> "PatientDemographics":
        self.name = compose_name(
            self.name, self.patient_name_last, self.patient_name_first
        )
        if self.name and (not self.patient_name_last or not self.patient_name_first):
            last, first = split_name(self.name)
            self.patient_name_last = self.patient_name_last or last
            self.patient_name_first = self.patient_name_first or first
        return self


class ReportMetadata(BaseModel):
    """Report header fields useful for provenance, linking, and QA."""

    model_config = ConfigDict(extra="forbid")

    accession: Optional[str] = Field(None, description="Accession number")
    specimen_number: Optional[str] = Field(
        None, description="Specimen number or specimen ID"
    )
    specimen_type: Optional[str] = Field(
        None, description="Specimen type, e.g. blood or buccal swab"
    )
    collected: Optional[str] = Field(
        None, description=f"Collection date from the report header. {DATE_FORMAT_HINT}"
    )
    received: Optional[str] = Field(
        None, description=f"Received date from the report header. {DATE_FORMAT_HINT}"
    )
    report_date: Optional[str] = Field(
        None,
        description=f"Final report/sign-out/result date shown in the report header or signature. {DATE_FORMAT_HINT}",
    )
    physician_name: Optional[str] = Field(
        None, description="Ordering/referring physician name when stated"
    )
    reason_for_referral: Optional[str] = Field(
        None,
        description="Reason for referral or indication text from the report header",
    )
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for non-null report metadata fields.",
    )

    @field_validator("collected", "received", "report_date")
    @classmethod
    def validate_dates(cls, value: Optional[str]) -> Optional[str]:
        return normalize_date(value)


class TestInformation(BaseModel):
    """Test-level fields aligned to the REDCap genetic_tests repeating instrument."""

    model_config = ConfigDict(extra="forbid")

    type: Optional[TestType] = Field(
        None, description="Genetic test type ordered or performed"
    )
    type_other: Optional[str] = Field(
        None, description="Free-text detail if type is Other"
    )
    reanalysis: Optional[bool] = Field(
        None, description="Whether the report is explicitly a reanalysis"
    )
    testname: Optional[str] = Field(
        None,
        description="Full test name exactly as printed on the report, including speed when present",
    )
    geneticrec: Optional[bool] = Field(
        None,
        description="Whether the test was recommended by a genetics provider. Usually not explicit on the report; use null unless clearly stated from source context.",
    )
    resultavailable: Optional[bool] = Field(
        None,
        description="Whether the result is available. For an actual report PDF this is usually true.",
    )
    file: Optional[str] = Field(
        None,
        description="Filename or path of the source report when available from the runtime context",
    )
    decline: Optional[bool] = Field(
        None,
        description="Whether the test was declined. Usually null for an actual returned report unless explicitly discussed.",
    )
    lab: Optional[TestLab] = Field(None, description="Testing laboratory")
    lab_other: Optional[str] = Field(
        None, description="Free-text detail if lab is Other"
    )
    order: Optional[str] = Field(
        None,
        description=f"Test order date. Prefer explicit order date; if unavailable on the report, collection date may be used as a proxy. {DATE_FORMAT_HINT}",
    )
    returndate: Optional[str] = Field(
        None, description=f"Report/result return date. {DATE_FORMAT_HINT}"
    )
    tat: Optional[int] = Field(
        None,
        description="Turnaround time in days if stated or derivable from order and return dates",
    )
    timeframe: Optional[TestTimeframe] = Field(
        None,
        description="Timing of testing relative to NICU stay. Often unavailable from the report alone.",
    )
    analysis: Optional[AnalysisType] = Field(
        None,
        description="Analysis type such as Proband Only, Duo, Trio, or Proband Only w/ Segregation",
    )
    consent_second: Optional[SecondaryFindings] = Field(
        None,
        description="Secondary findings consent status: Opt-IN, Opt-OUT, or Not Reported/N/A",
    )
    findings: Optional[bool] = Field(
        None,
        description="Whether any findings are reported. Negative tests should usually be false.",
    )
    order_age: Optional[int] = Field(
        None,
        description="Age in days at order. Usually absent from the report itself unless provided from source context.",
    )
    return_age: Optional[int] = Field(
        None,
        description="Age in days at return date. Usually absent from the report itself unless provided from source context.",
    )
    time_admit: Optional[int] = Field(
        None,
        description="Days from NICU admission to order date. Usually absent from the report itself unless provided from source context.",
    )
    testcomments: Optional[str] = Field(
        None,
        description="Additional test-level note or shorthand result comment. Use for cytogenetic nomenclature such as 46,XX or 47,XX,+21 when helpful.",
    )
    order_post_death: Optional[str] = Field(
        None,
        description="Legacy calculated field for whether order occurred on/after death. Usually unavailable from report-only extraction.",
    )
    return_post_death: Optional[str] = Field(
        None,
        description="Legacy calculated field for whether result returned on/after death. Usually unavailable from report-only extraction.",
    )
    missing_gt: Optional[str] = Field(
        None,
        description="Legacy calculated missingness summary. Usually unavailable from report-only extraction.",
    )
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for non-null test fields.",
    )

    @field_validator("order", "returndate")
    @classmethod
    def validate_dates(cls, value: Optional[str]) -> Optional[str]:
        return normalize_date(value)

    @model_validator(mode="after")
    def fill_tat(self) -> "TestInformation":
        if self.tat is None and self.order and self.returndate:
            order_date = date.fromisoformat(self.order)
            return_date = date.fromisoformat(self.returndate)
            self.tat = (return_date - order_date).days
        if self.resultavailable is None and self.returndate:
            self.resultavailable = True
        return self


class GeneticFinding(BaseModel):
    """One finding slot from the REDCap genetic_tests form."""

    model_config = ConfigDict(extra="forbid")

    genelocus: Optional[str] = Field(
        None,
        description="Associated gene, locus, chromosome, or cytogenetic region for this finding",
    )
    dna: Optional[str] = Field(
        None,
        description="DNA-level HGVS string or genomic/cytogenetic coordinate/nomenclature",
    )
    protein: Optional[str] = Field(
        None, description="Protein-level HGVS notation when stated"
    )
    transcript: Optional[str] = Field(
        None, description="Reference transcript identifier when stated"
    )
    dose: Optional[Dosage] = Field(
        None, description="Dosage/copy-number state for CNV-style findings"
    )
    roh: Optional[bool] = Field(
        None,
        description="Whether the report explicitly describes this as a region of homozygosity/AOH-related finding",
    )
    labclass: Optional[LabClassification] = Field(
        None, description="Laboratory classification for this finding"
    )
    zygosity: Optional[Zygosity] = Field(None, description="Zygosity for the finding")
    inheritance: Optional[InheritancePattern] = Field(
        None, description="Inheritance pattern for the finding/disorder"
    )
    segregation: Optional[Segregation] = Field(
        None, description="Segregation or parental-origin assessment"
    )
    findingclass: Optional[FindingClassification] = Field(
        None, description="Primary, Secondary, Incidental, or Unknown"
    )
    findingcomments: Optional[str] = Field(
        None, description="Additional free-text note about the finding"
    )
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for non-null finding fields.",
    )


class TestInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    testdx: Optional[TestInterpretationResult] = Field(
        None, description="Overall test interpretation: diagnostic or nondiagnostic"
    )
    refseq: Optional[ReferenceGenome] = Field(
        None, description="Reference genome build when stated"
    )
    interpretation_text: Optional[str] = Field(
        None,
        description="Short free-text summary of the lab interpretation or conclusion",
    )
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for non-null interpretation fields.",
    )


class PatientPhenotypeNote(BaseModel):
    """Patient phenotype/HPO note block aligned to the separate patient_phenotypes form."""

    model_config = ConfigDict(extra="forbid")

    hpo_date: Optional[str] = Field(
        None,
        description="Phenotype test date. In REDCap this usually mirrors the genetic test order date. Use the best available date if explicit.",
    )
    hpo_terms: Optional[str] = Field(
        None,
        description="Free-text phenotype or HPO terms block exactly as written or clearly summarized from the report",
    )
    parsed_terms: Optional[List[str]] = Field(
        None,
        description="Optional parsed list of phenotype/HPO terms when a clear comma/semicolon-delimited list is present",
    )
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for non-null phenotype-note fields.",
    )

    @field_validator("hpo_date")
    @classmethod
    def validate_hpo_date(cls, value: Optional[str]) -> Optional[str]:
        return normalize_date(value)


class GeneticDiagnosis(BaseModel):
    """One diagnosis slot from the REDCap genetic_diagnoses repeating form."""

    model_config = ConfigDict(extra="forbid")

    dxname: Optional[str] = Field(
        None,
        description="Name of the confirmed diagnosis/disorder. Do not use a carrier state, ROH alone, or isolated VUS as a diagnosis.",
    )
    dx_test: Optional[TestType] = Field(
        None, description="Diagnostic test type responsible for this diagnosis"
    )
    dx_test_other: Optional[str] = Field(
        None, description="Free-text detail if dx_test is Other"
    )
    dxgenelocus: Optional[str] = Field(
        None,
        description="Associated diagnostic gene, locus, chromosome, or chromosomal region",
    )
    dxdate: Optional[str] = Field(
        None,
        description=f"Date of diagnosis/diagnostic report return. {DATE_FORMAT_HINT}",
    )
    eradx: Optional[DiagnosisStudyPeriod] = Field(
        None,
        description="Study period of diagnosis relative to the NICU stay",
    )
    dxfhx: Optional[bool] = Field(
        None,
        description="Whether relevant family history was known at the time of testing",
    )
    dxage: Optional[int] = Field(
        None, description="Age in days at diagnosis when available from source context"
    )
    dxtimeto: Optional[int] = Field(
        None,
        description="Days from NICU admission to diagnosis when available from source context",
    )
    dxomim: Optional[str] = Field(
        None, description="OMIM identifier or mapped phenotype MIM number"
    )
    dxorpha: Optional[str] = Field(None, description="ORPHA identifier when available")
    dxinfo_other: Optional[str] = Field(
        None, description="Additional free-text diagnosis note"
    )
    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description="Per-field confidence for non-null diagnosis fields.",
    )

    @field_validator("dxdate")
    @classmethod
    def validate_dx_date(cls, value: Optional[str]) -> Optional[str]:
        return normalize_date(value)


class GeneticReportExtraction(BaseModel):
    """Linked, single-report extraction model used by the LLM pipeline.

    One output JSON from `extract` corresponds to one genetic report/test and keeps
    together the report metadata, the test row, findings, phenotype note(s), and
    diagnosis note/diagnosis objects.
    """

    model_config = ConfigDict(extra="forbid")

    patient: PatientDemographics = Field(
        default_factory=PatientDemographics,
        description="Patient identifiers for this report",
    )
    report_metadata: ReportMetadata = Field(
        default_factory=ReportMetadata,
        description="Header/provenance metadata from the report",
    )
    test_info: TestInformation = Field(
        default_factory=TestInformation,
        description="Test-level REDCap-aligned information for this report",
    )
    findings_list: Optional[List[GeneticFinding]] = Field(
        None,
        description="All reported findings for this report, one object per finding. Use [] or null if there are no findings.",
    )
    interpretation: TestInterpretation = Field(
        default_factory=TestInterpretation,
        description="Overall test interpretation",
    )
    patient_phenotypes: Optional[List[PatientPhenotypeNote]] = Field(
        None,
        description="Phenotype/HPO note blocks associated with this report. Usually zero or one per report.",
    )
    diagnoses: Optional[List[GeneticDiagnosis]] = Field(
        None,
        description="Confirmed diagnoses attributable to this report. Leave empty/null when the report is nondiagnostic or does not state a confirmed diagnosis.",
    )
    diagnoses_note: Optional[str] = Field(
        None,
        description="Free-text diagnosis note/comment associated with this report's diagnoses",
    )
    extraction_confidence: Optional[str] = Field(
        None, description="Overall extraction confidence: high, moderate, or low"
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Caveats, unresolved ambiguities, or other extraction notes",
    )

    @model_validator(mode="after")
    def validate_global_consistency(self) -> "GeneticReportExtraction":
        if self.extraction_confidence and self.extraction_confidence not in {
            "high",
            "moderate",
            "low",
        }:
            raise ValueError(
                "extraction_confidence must be one of: high, moderate, low"
            )

        if self.test_info.findings is False and self.findings_list:
            raise ValueError(
                "test_info.findings is False but findings_list is non-empty"
            )
        return self


# ---------------------------------------------------------------------------
# Patient-level linked model for REDCap import/export
# ---------------------------------------------------------------------------


class LegacyRepeatRow(BaseModel):
    """Unmodeled repeat instrument row preserved verbatim for round-tripping."""

    model_config = ConfigDict(extra="forbid")

    redcap_repeat_instrument: Optional[str] = Field(
        None, description="Name of the REDCap repeating instrument"
    )
    redcap_repeat_instance: Optional[int] = Field(
        None, description="Repeat instance number within that instrument"
    )
    row_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Verbatim row dictionary preserved from the raw REDCap export",
    )
    source_index: Optional[int] = Field(
        None, description="Original row index in the imported file"
    )


class LegacyPhenotypeNote(PatientPhenotypeNote):
    source_repeat_instance: Optional[int] = Field(
        None, description="Original REDCap repeat instance if imported from legacy rows"
    )
    source_row: Optional[Dict[str, Any]] = Field(
        None,
        description="Verbatim patient_phenotypes row dictionary from legacy import",
    )


class GeneticDiagnosisGroup(BaseModel):
    """One legacy diagnosis-note row that may hold up to three diagnoses."""

    model_config = ConfigDict(extra="forbid")

    diagnoses: List[GeneticDiagnosis] = Field(
        default_factory=list,
        description="Diagnoses contained in this legacy diagnosis-note row",
    )
    diagnoses_note: Optional[str] = Field(
        None,
        description="Free-text diagnosis note/comment associated with this row",
    )
    source_repeat_instance: Optional[int] = Field(
        None, description="Original REDCap repeat instance if imported"
    )
    source_row: Optional[Dict[str, Any]] = Field(
        None,
        description="Verbatim genetic_diagnoses row dictionary from legacy import",
    )


class LinkedGeneticReport(GeneticReportExtraction):
    """Normalized report object with optional legacy-row provenance."""

    source_repeat_instance: Optional[int] = Field(
        None, description="Original genetic_tests repeat instance if imported"
    )
    source_row: Optional[Dict[str, Any]] = Field(
        None,
        description="Verbatim genetic_tests row dictionary from legacy import",
    )
    linked_patient_phenotypes: List[LegacyPhenotypeNote] = Field(
        default_factory=list,
        description="Legacy patient_phenotypes rows linked to this report during import",
    )
    linked_diagnosis_groups: List[GeneticDiagnosisGroup] = Field(
        default_factory=list,
        description="Legacy genetic_diagnoses rows linked to this report during import",
    )
    link_method: Optional[str] = Field(
        None,
        description="How diagnosis/phenotype rows were linked when imported from flat legacy rows",
    )


class LinkedGeneticsCase(BaseModel):
    """Patient-level linked case model for REDCap import/export and comparison.

    This keeps the high-resolution, report-centric association while remaining able to
    degrade back to the current flat REDCap export layout.
    """

    model_config = ConfigDict(extra="forbid")

    patient: PatientDemographics = Field(
        default_factory=PatientDemographics,
        description="Patient identifiers for the case",
    )
    nicu_flags: Optional[Dict[str, Any]] = Field(
        None,
        description="Convenience subset of genetics-related NICU/base-row flags when imported from REDCap",
    )
    reports: List[LinkedGeneticReport] = Field(
        default_factory=list,
        description="Linked report objects, one per genetic_tests repeat row or extracted report",
    )
    unlinked_patient_phenotypes: List[LegacyPhenotypeNote] = Field(
        default_factory=list,
        description="Phenotype-note rows that could not be linked to a specific report during legacy import",
    )
    unlinked_diagnosis_groups: List[GeneticDiagnosisGroup] = Field(
        default_factory=list,
        description="Diagnosis-note rows that could not be linked to a specific report during legacy import",
    )
    legacy_base_row: Optional[Dict[str, Any]] = Field(
        None,
        description="Verbatim non-repeating base row from the raw REDCap export",
    )
    legacy_other_repeat_rows: List[LegacyRepeatRow] = Field(
        default_factory=list,
        description="Other repeating-instrument rows preserved verbatim for round-tripping",
    )
    redcap_column_order: Optional[List[str]] = Field(
        None,
        description="Header order captured from the imported raw REDCap export",
    )
    source_file: Optional[str] = Field(
        None, description="Originating file path/name for the case import"
    )


# ---------------------------------------------------------------------------
# Public model accessors
# ---------------------------------------------------------------------------


def get_extraction_model() -> Type[BaseModel]:
    return GeneticReportExtraction


def get_case_model() -> Type[BaseModel]:
    return LinkedGeneticsCase


# ---------------------------------------------------------------------------
# Recursive prompt builder used by main.py
# ---------------------------------------------------------------------------


def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union and type(None) in args:
        inner = [arg for arg in args if arg is not type(None)]
        if len(inner) == 1:
            return inner[0], True
    return annotation, False


def describe_field(name: str, info, indent: int = 0) -> List[str]:
    prefix = "  " * indent
    lines: List[str] = []
    description = info.description or ""

    raw_annotation = info.annotation
    inner, is_optional = unwrap_optional(raw_annotation)
    optional_suffix = " (optional)" if is_optional else ""

    origin = get_origin(inner)

    if origin is list:
        args = get_args(inner)
        lines.append(f"{prefix}- {name}{optional_suffix}: {description}")
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            lines.append(f"{prefix}  Each list item contains:")
            for sub_name, sub_info in args[0].model_fields.items():
                lines.extend(describe_field(sub_name, sub_info, indent + 2))
        return lines

    if origin is dict:
        lines.append(f"{prefix}- {name}{optional_suffix}: {description}")
        args = get_args(inner)
        if (
            len(args) == 2
            and isinstance(args[1], type)
            and issubclass(args[1], BaseModel)
        ):
            lines.append(f"{prefix}  Each dictionary value contains:")
            for sub_name, sub_info in args[1].model_fields.items():
                lines.extend(describe_field(sub_name, sub_info, indent + 2))
        return lines

    if isinstance(inner, type) and issubclass(inner, Enum):
        allowed_values = [member.value for member in inner]
        lines.append(
            f"{prefix}- {name}{optional_suffix}: {description}. Allowed values: {allowed_values}"
        )
        return lines

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        lines.append(f"{prefix}- {name}{optional_suffix}: {description}")
        for sub_name, sub_info in inner.model_fields.items():
            lines.extend(describe_field(sub_name, sub_info, indent + 1))
        return lines

    lines.append(f"{prefix}- {name}{optional_suffix}: {description}")
    return lines


def build_format_instructions(model: Optional[Type[BaseModel]] = None) -> str:
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
    lines.extend(
        [
            "",
            "Use null for any field that cannot be determined from the report.",
            "Do not invent information.",
            "For each extraction-facing sub-model, populate field_confidence only for non-null fields you actually extracted.",
        ]
    )
    return "\n".join(lines)
