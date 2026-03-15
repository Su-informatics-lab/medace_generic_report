"""Pydantic schema for structured pediatric genetic test report extraction.

SINGLE SOURCE OF TRUTH — edit this file to change what gets extracted.
The extraction pipeline reads this schema at runtime, generates prompt instructions
from it, validates model output against it, and re-asks on validation failure.

Design goals for the IU pediatric/NICU genetic testing project:
- align closely to Caroline Parker's highlighted REDCap variables
- keep repeatable REDCap blocks as Python lists instead of *_2 ... *_20 fields
- preserve room for supplemental report-native content (for example phenotype text)
- make downstream comparison to human abstraction straightforward
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Base helpers
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Base model with strict extra-field handling."""

    model_config = ConfigDict(extra="forbid")


class FieldConfidence(StrictModel):
    """Per-field extraction confidence reported by the model."""

    confidence: str = Field(
        description="Self-assessed confidence: high, moderate, or low"
    )
    interval: Optional[str] = Field(
        None,
        description="Estimated probability interval, e.g. '0.85-0.95'",
    )
    comment: Optional[str] = Field(
        None,
        description="Why confidence is not high, or notes on ambiguity",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: str) -> str:
        allowed = {"high", "moderate", "low"}
        if value not in allowed:
            raise ValueError(f"confidence must be one of {sorted(allowed)}")
        return value


class ConfidenceTrackedModel(StrictModel):
    """Base class for sub-models that carry a field_confidence map."""

    field_confidence: Optional[Dict[str, FieldConfidence]] = Field(
        None,
        description=(
            "Per-field confidence for each extracted non-null field in this object. "
            "Keys should be field names from this object."
        ),
    )

    @model_validator(mode="after")
    def validate_field_confidence_keys(self) -> "ConfidenceTrackedModel":
        if not self.field_confidence:
            return self

        valid_keys = {
            name for name in self.__class__.model_fields if name != "field_confidence"
        }
        invalid_keys = sorted(set(self.field_confidence) - valid_keys)
        if invalid_keys:
            raise ValueError(
                f"field_confidence contains invalid keys for {self.__class__.__name__}: {invalid_keys}"
            )
        return self


# ---------------------------------------------------------------------------
# Enums aligned to highlighted REDCap fields where possible
# ---------------------------------------------------------------------------


class Sex(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"


class TestType(str, Enum):
    fish = "FISH"
    karyotype = "Karyotype (Chromosome Analysis)"
    microarray = "Microarray (CMA)"
    targeted_variant = "Targeted Variant"
    single_gene_seq = "Single Gene (Sequencing, Del/Dup)"
    single_gene_repeat = "Single Gene (Repeat Analysis)"
    panel_or_exome_slice = "Panel or Exome Slice"
    exome = "Exome Sequencing"
    genome = "Genome Sequencing"
    mitochondrial = "Mitochondrial DNA"
    imprinting_methylation = "Imprinting/Methylation"
    other = "Other"
    not_reported_unknown = "Not Reported/Unknown"


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
    not_reported_unknown = "Not Reported/Unknown"


class TestTimeframe(str, Enum):
    nicu_stay = "NICU Stay"
    pre_nicu_admission_osh = "Pre-NICU Admission/OSH"
    prenatal = "Prenatal"
    post_nicu_discharge = "Post-NICU Discharge"
    postmortem = "Postmortem"
    other_unknown = "Other/Unknown"


class AnalysisType(str, Enum):
    proband_only = "Proband Only"
    proband_only_w_segregation = "Proband Only w/ Segregation"
    duo = "Duo"
    trio = "Trio"
    other = "Other"
    not_reported_unknown = "Not Reported/Unknown"


class SecondaryFindings(str, Enum):
    opt_in = "Opt-IN"
    opt_out = "Opt-OUT"
    not_reported_or_na = "Not Reported or N/A"


class LabClassification(str, Enum):
    pathogenic = "Pathogenic"
    likely_pathogenic = "Likely Pathogenic"
    vus = "VUS"
    abnormal_nos = "Abnormal NOS"
    likely_benign = "Likely Benign"
    benign = "Benign"
    normal_nos = "Normal NOS"
    other = "Other"
    not_reported_not_applicable = "Not Reported/Not Applicable"


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
    not_reported_not_applicable = "Not Reported/Not Applicable"


class FindingClassification(str, Enum):
    primary = "Primary"
    secondary = "Secondary"
    incidental = "Incidental"
    unknown = "Unknown"


class TestInterpretationResult(str, Enum):
    nondiagnostic = "Nondiagnostic"
    diagnostic = "Diagnostic"
    not_reported_unknown = "Not Reported/Unknown"


class ReferenceGenome(str, Enum):
    grch37_hg19 = "GRCh37/hg19"
    grch38_hg38 = "GRCh38/hg38"
    other = "Other"
    not_reported_unknown = "Not Reported/Unknown"


class DiagnosisStudyPeriod(str, Enum):
    pre_nicu_stay = "Pre-NICU Stay"
    nicu_stay = "NICU Stay"
    year_1_post_nicu_stay = "Year 1 Post-NICU Stay"
    year_2_post_nicu_stay = "Year 2 Post-NICU Stay"
    other = "Other"
    unknown = "Unknown"


# ---------------------------------------------------------------------------
# Validators/utilities
# ---------------------------------------------------------------------------


def validate_iso_date(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return value
    value = value.strip()
    if not value:
        return None
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD, got '{value}'") from exc
    return value


def normalize_string_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return values
    cleaned: List[str] = []
    seen: set[str] = set()
    for item in values:
        if item is None:
            continue
        stripped = str(item).strip()
        if not stripped:
            continue
        if stripped not in seen:
            seen.add(stripped)
            cleaned.append(stripped)
    return cleaned or None


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class PatientDemographics(ConfidenceTrackedModel):
    """Patient identifiers printed on the report itself."""

    patient_name_last: Optional[str] = Field(
        None,
        description="Patient last/family name as printed on the report",
    )
    patient_name_first: Optional[str] = Field(
        None,
        description="Patient first/given name as printed on the report",
    )
    date_of_birth: Optional[str] = Field(
        None,
        description="Date of birth in YYYY-MM-DD format",
    )
    mrn: Optional[str] = Field(
        None,
        description="Medical record number / MRN printed on the report",
    )
    sex: Optional[Sex] = Field(
        None,
        description="Biological sex or gender as printed on the report",
    )

    @field_validator("date_of_birth")
    @classmethod
    def validate_date_of_birth(cls, value: Optional[str]) -> Optional[str]:
        return validate_iso_date(value, "date_of_birth")


class NICUFlags(ConfidenceTrackedModel):
    """Binary REDCap-style flags derived from timing and diagnostic status."""

    nicu_genetic_testing: Optional[bool] = Field(
        None,
        description=(
            "REDCap nicugt. True if the patient had ANY diagnostic genetic testing "
            "recommended, initiated, or completed during the NICU stay, prenatally, "
            "or at an outside hospital prior to/around NICU transfer. False for tests "
            "that occurred only after NICU discharge."
        ),
    )
    nicu_genetic_diagnosis: Optional[bool] = Field(
        None,
        description=(
            "REDCap nicudx. True only if a diagnostic genetic result relevant to the "
            "NICU period was made during the NICU stay, prenatally, or at an outside "
            "hospital before/around NICU transfer. VUS-only, carrier status, ROH-only, "
            "or post-discharge-only diagnoses should be False."
        ),
    )


class TestInformation(ConfidenceTrackedModel):
    """Test-level metadata aligned to highlighted REDCap genetic testing variables."""

    test_type: Optional[TestType] = Field(
        None,
        description="REDCap [type]. Type of genetic test ordered or performed.",
    )
    test_type_other: Optional[str] = Field(
        None,
        description="REDCap [type_other]. If test_type is Other, specify the test type.",
    )
    test_name: Optional[str] = Field(
        None,
        description=(
            "REDCap [testname]. Full test name as printed on the report. Include speed "
            "qualifier when present, for example 'Rapid Genome Sequencing' or 'Trio Rapid Genome Sequencing'."
        ),
    )
    result_available: Optional[bool] = Field(
        None,
        description=(
            "REDCap [resultavailable]. True when the report contains an actual test result. "
            "False when testing was recommended/initiated but no result is available or the test was declined."
        ),
    )
    test_declined: Optional[bool] = Field(
        None,
        description=(
            "REDCap [decline]. True only if the report/text explicitly indicates the test was declined. "
            "For an actual result PDF this is usually False."
        ),
    )
    reanalysis: Optional[bool] = Field(
        None,
        description="Whether the report explicitly describes this test as a reanalysis.",
    )
    lab: Optional[TestLab] = Field(
        None,
        description="REDCap [lab]. Testing laboratory.",
    )
    lab_other: Optional[str] = Field(
        None,
        description="REDCap [lab_other]. If lab is Other, specify the testing laboratory.",
    )
    order_date: Optional[str] = Field(
        None,
        description=(
            "REDCap [order]. Test order date in YYYY-MM-DD format. If order date is not "
            "explicitly stated on the report, collection date may be used as a proxy."
        ),
    )
    result_date: Optional[str] = Field(
        None,
        description="REDCap [returndate]. Result/report date in YYYY-MM-DD format.",
    )
    timeframe: Optional[TestTimeframe] = Field(
        None,
        description="REDCap [timeframe]. Timing of the test relative to the NICU stay.",
    )
    analysis_type: Optional[AnalysisType] = Field(
        None,
        description="REDCap [analysis]. Analysis type such as Proband Only, Duo, or Trio.",
    )
    secondary_findings: Optional[SecondaryFindings] = Field(
        None,
        description=(
            "REDCap [consent_second]. Secondary findings consent/selection status. "
            "Infer from report language when clearly stated."
        ),
    )
    findings_reported: Optional[bool] = Field(
        None,
        description=(
            "REDCap [findings]. True if the report contains any reported findings, including "
            "diagnostic findings, VUS, incidental findings, ROH, or secondary findings. False for truly negative reports."
        ),
    )

    @field_validator("order_date", "result_date")
    @classmethod
    def validate_dates(cls, value: Optional[str], info) -> Optional[str]:
        return validate_iso_date(value, info.field_name)


class PatientPhenotypes(ConfidenceTrackedModel):
    """Phenotype/HPO content carried on or derivable from the report text."""

    phenotype_test_date: Optional[str] = Field(
        None,
        description=(
            "REDCap [hpo_date]. Date associated with the phenotype/HPO terms, usually the same as order_date, in YYYY-MM-DD format."
        ),
    )
    clinical_indication_text: Optional[str] = Field(
        None,
        description=(
            "Raw clinical indication / phenotype summary text from the report, if present. Preserve report wording."
        ),
    )
    hpo_terms: Optional[List[str]] = Field(
        None,
        description=(
            "REDCap [hpo_terms]. Phenotype or HPO-like terms listed on the report. Preserve report wording and split into distinct items when clearly separated."
        ),
    )

    @field_validator("phenotype_test_date")
    @classmethod
    def validate_phenotype_test_date(cls, value: Optional[str]) -> Optional[str]:
        return validate_iso_date(value, "phenotype_test_date")

    @field_validator("hpo_terms")
    @classmethod
    def normalize_hpo_terms(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return normalize_string_list(value)


class GeneticFinding(ConfidenceTrackedModel):
    """One finding from the report. Replaces finding1 ... finding20 REDCap blocks."""

    condition_name: Optional[str] = Field(
        None,
        description=(
            "Disease/condition name for this finding as stated on the report, if present. "
            "Examples include a named syndrome, trisomy, or a CNV region/disease label from a findings table."
        ),
    )
    gene_locus: Optional[str] = Field(
        None,
        description=(
            "REDCap [genelocus]. Gene symbol, locus, chromosome, or chromosomal region for the finding."
        ),
    )
    variant: Optional[str] = Field(
        None,
        description=(
            "REDCap [dna]. Variant nomenclature, genomic coordinates, or cytogenetic nomenclature exactly as reported. "
            "Examples: 'c.123A>G', 'chr4:10000-49157256DUP', or '47,XX,+21'."
        ),
    )
    protein: Optional[str] = Field(
        None,
        description="REDCap [protein]. Protein nomenclature when applicable, for example 'p.Thr123Ala'.",
    )
    transcript: Optional[str] = Field(
        None,
        description="REDCap [transcript]. Transcript identifier when present, for example 'NM_000551.4'.",
    )
    dosage: Optional[Dosage] = Field(
        None,
        description=(
            "REDCap [dose]. Copy-number dosage or copy-state label. Use x3 for trisomy and x1 for monosomy when explicitly appropriate."
        ),
    )
    roh: Optional[bool] = Field(
        None,
        description="REDCap [roh]. True if this finding is a region of homozygosity (ROH/AOH).",
    )
    classification: Optional[LabClassification] = Field(
        None,
        description="REDCap [labclass]. Laboratory classification for this finding.",
    )
    zygosity: Optional[Zygosity] = Field(
        None,
        description="REDCap [zygosity]. Zygosity for this finding when reported.",
    )
    inheritance: Optional[InheritancePattern] = Field(
        None,
        description="REDCap [inheritance]. Inheritance pattern when reported.",
    )
    segregation: Optional[Segregation] = Field(
        None,
        description="REDCap [segregation]. Segregation/parental origin when reported.",
    )
    finding_class: Optional[FindingClassification] = Field(
        None,
        description=(
            "REDCap [findingclass]. Primary, Secondary, Incidental, or Unknown. "
            "Secondary findings are ACMG secondary findings; incidental findings are reported as unrelated additional/incidental findings."
        ),
    )
    additional_finding: Optional[bool] = Field(
        None,
        description=(
            "REDCap [second]. True if the report indicates there are additional findings after this one. "
            "For list-based output, this can usually be inferred from list length."
        ),
    )
    additional_info: Optional[str] = Field(
        None,
        description="REDCap [findingcomments]. Additional notes about the finding.",
    )


class TestInterpretation(ConfidenceTrackedModel):
    """Overall report interpretation block."""

    result: Optional[TestInterpretationResult] = Field(
        None,
        description="REDCap [testdx]. Overall test interpretation.",
    )
    reference_genome: Optional[ReferenceGenome] = Field(
        None,
        description="REDCap [refseq]. Reference genome / sequence build.",
    )
    interpretation_summary: Optional[str] = Field(
        None,
        description=(
            "Brief free-text summary of the laboratory's interpretation or conclusion, preserving the report's meaning."
        ),
    )


class GeneticDiagnosis(ConfidenceTrackedModel):
    """One confirmed diagnosis derived from this report. Replaces dxname_1..dxname_3 blocks."""

    diagnosis_name: Optional[str] = Field(
        None,
        description=(
            "REDCap [dxname]. Name of the diagnosis/condition. Do not create diagnoses for carrier status, ROH, negative results, or VUS without a disorder name."
        ),
    )
    diagnostic_test: Optional[TestType] = Field(
        None,
        description="REDCap [dx_test]. Test type that produced the diagnosis.",
    )
    diagnostic_test_other: Optional[str] = Field(
        None,
        description="REDCap [dx_test_other]. If diagnostic_test is Other, specify the test.",
    )
    associated_gene_locus: Optional[str] = Field(
        None,
        description=(
            "REDCap [dxgenelocus]. Associated gene, locus, chromosome, or chromosomal region for the diagnosis."
        ),
    )
    diagnosis_date: Optional[str] = Field(
        None,
        description=(
            "REDCap [dxdate]. Date of diagnosis in YYYY-MM-DD format. Usually the same as the diagnostic report result date."
        ),
    )
    study_period_of_diagnosis: Optional[DiagnosisStudyPeriod] = Field(
        None,
        description="REDCap [eradx]. Study period of diagnosis.",
    )
    family_history_known_at_testing: Optional[bool] = Field(
        None,
        description=(
            "REDCap [dxfhx]. True if the report clearly indicates a known family history or known familial variant at the time of testing."
        ),
    )
    phenotype_omim: Optional[str] = Field(
        None,
        description=(
            "REDCap [dxomim]. OMIM identifier for the diagnosis when explicitly stated. Usually null unless the report includes it."
        ),
    )
    orpha_number: Optional[str] = Field(
        None,
        description=(
            "REDCap [dxorpha]. ORPHA identifier for the diagnosis when explicitly stated. Usually null unless the report includes it."
        ),
    )

    @field_validator("diagnosis_date")
    @classmethod
    def validate_diagnosis_date(cls, value: Optional[str]) -> Optional[str]:
        return validate_iso_date(value, "diagnosis_date")


# ---------------------------------------------------------------------------
# Top-level extraction model
# ---------------------------------------------------------------------------


class GeneticTestExtraction(StrictModel):
    """Structured extraction for one pediatric/NICU genetic testing report."""

    patient: PatientDemographics = Field(
        default_factory=PatientDemographics,
        description="Patient demographics printed on the report",
    )
    nicu_flags: NICUFlags = Field(
        default_factory=NICUFlags,
        description="REDCap-style NICU flags derived from the report and timing",
    )
    test_info: TestInformation = Field(
        default_factory=TestInformation,
        description="Test-level metadata aligned to the genetic testing form",
    )
    patient_phenotypes: PatientPhenotypes = Field(
        default_factory=PatientPhenotypes,
        description="Phenotype/HPO-style content carried on the report",
    )
    findings: Optional[List[GeneticFinding]] = Field(
        None,
        description=(
            "List of all reported findings. One object per finding. Use [] or null when the report clearly has no findings."
        ),
    )
    interpretation: TestInterpretation = Field(
        default_factory=TestInterpretation,
        description="Overall laboratory interpretation",
    )
    diagnoses: Optional[List[GeneticDiagnosis]] = Field(
        None,
        description=(
            "List of confirmed diagnoses supported by this report. One object per diagnosis. "
            "Exclude carrier status, ROH-only findings, negative results, and VUS without a disorder name."
        ),
    )
    extraction_confidence: Optional[str] = Field(
        None,
        description="Overall extraction confidence: high, moderate, or low",
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Caveats, ambiguities, or unresolved issues from the report",
    )

    @field_validator("extraction_confidence")
    @classmethod
    def validate_extraction_confidence(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        allowed = {"high", "moderate", "low"}
        if value not in allowed:
            raise ValueError(f"extraction_confidence must be one of {sorted(allowed)}")
        return value

    @field_validator("findings")
    @classmethod
    def normalize_findings(
        cls, value: Optional[List[GeneticFinding]]
    ) -> Optional[List[GeneticFinding]]:
        if value == []:
            return []
        return value

    @field_validator("diagnoses")
    @classmethod
    def normalize_diagnoses(
        cls, value: Optional[List[GeneticDiagnosis]]
    ) -> Optional[List[GeneticDiagnosis]]:
        if value == []:
            return []
        return value

    @model_validator(mode="after")
    def auto_fill_derived_fields(self) -> "GeneticTestExtraction":
        # hpo_date is a calculated REDCap field tied to order_date.
        if (
            self.patient_phenotypes.phenotype_test_date is None
            and self.test_info.order_date is not None
        ):
            self.patient_phenotypes.phenotype_test_date = self.test_info.order_date

        # diagnosis fields can inherit report-wide defaults for a single-report extraction.
        if self.diagnoses:
            for diagnosis in self.diagnoses:
                if (
                    diagnosis.diagnosis_date is None
                    and self.test_info.result_date is not None
                ):
                    diagnosis.diagnosis_date = self.test_info.result_date
                if (
                    diagnosis.diagnostic_test is None
                    and self.test_info.test_type is not None
                ):
                    diagnosis.diagnostic_test = self.test_info.test_type
                if (
                    diagnosis.study_period_of_diagnosis is None
                    and self.test_info.timeframe is not None
                ):
                    if self.test_info.timeframe in {
                        TestTimeframe.prenatal,
                        TestTimeframe.pre_nicu_admission_osh,
                    }:
                        diagnosis.study_period_of_diagnosis = (
                            DiagnosisStudyPeriod.pre_nicu_stay
                        )
                    elif self.test_info.timeframe == TestTimeframe.nicu_stay:
                        diagnosis.study_period_of_diagnosis = (
                            DiagnosisStudyPeriod.nicu_stay
                        )

        nicu_like = {
            TestTimeframe.nicu_stay,
            TestTimeframe.pre_nicu_admission_osh,
            TestTimeframe.prenatal,
        }
        post_nicu_like = {
            TestTimeframe.post_nicu_discharge,
            TestTimeframe.postmortem,
        }

        if (
            self.nicu_flags.nicu_genetic_testing is None
            and self.test_info.timeframe is not None
        ):
            if self.test_info.timeframe in nicu_like:
                self.nicu_flags.nicu_genetic_testing = True
            elif self.test_info.timeframe in post_nicu_like:
                self.nicu_flags.nicu_genetic_testing = False

        if (
            self.nicu_flags.nicu_genetic_diagnosis is None
            and self.test_info.timeframe is not None
        ):
            if self.interpretation.result == TestInterpretationResult.diagnostic:
                self.nicu_flags.nicu_genetic_diagnosis = (
                    self.test_info.timeframe in nicu_like
                )
            elif self.interpretation.result == TestInterpretationResult.nondiagnostic:
                self.nicu_flags.nicu_genetic_diagnosis = False

        return self

    @model_validator(mode="after")
    def findings_consistent_with_test_info(self) -> "GeneticTestExtraction":
        if self.test_info.findings_reported is False and self.findings:
            raise ValueError(
                "test_info.findings_reported is False but findings list is non-empty"
            )
        if self.test_info.findings_reported is True and not self.findings:
            raise ValueError(
                "test_info.findings_reported is True but findings list is empty"
            )
        if (
            self.test_info.test_declined is True
            and self.test_info.result_available is True
        ):
            raise ValueError(
                "test_info.test_declined cannot be True when result_available is True"
            )
        return self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_extraction_model() -> Type[BaseModel]:
    """Return the top-level extraction model class."""

    return GeneticTestExtraction


# ---------------------------------------------------------------------------
# Prompt/schema rendering helpers
# ---------------------------------------------------------------------------


def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Unwrap Optional[X] -> (X, True); non-optional -> (annotation, False)."""

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union and type(None) in args:
        inner = [arg for arg in args if arg is not type(None)]
        if len(inner) == 1:
            return inner[0], True
    return annotation, False


def describe_field(name: str, info, indent: int = 0) -> List[str]:
    """Recursively describe a Pydantic field for prompt instructions."""

    prefix = "  " * indent
    lines: List[str] = []
    desc = info.description or ""

    raw_annotation = info.annotation
    inner, is_optional = unwrap_optional(raw_annotation)

    list_origin = get_origin(inner)
    if list_origin in (list, List):
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

    if isinstance(inner, type) and issubclass(inner, Enum):
        allowed = [member.value for member in inner]
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}. Allowed values: {allowed}")
        return lines

    if isinstance(inner, type) and issubclass(inner, BaseModel):
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}")
        for sub_name, sub_info in inner.model_fields.items():
            lines.extend(describe_field(sub_name, sub_info, indent + 1))
        return lines

    if get_origin(inner) in (dict, Dict):
        lines.append(f"{prefix}- {name}{opt_tag}: {desc}")
        dict_args = get_args(inner)
        if len(dict_args) == 2:
            value_type = dict_args[1]
            if isinstance(value_type, type) and issubclass(value_type, BaseModel):
                lines.append(f"{prefix}  Each value is an object with:")
                for sub_name, sub_info in value_type.model_fields.items():
                    lines.extend(describe_field(sub_name, sub_info, indent + 2))
        return lines

    constraints: List[str] = []
    if info.metadata:
        for metadata in info.metadata:
            if hasattr(metadata, "ge"):
                constraints.append(f">= {metadata.ge}")
            if hasattr(metadata, "le"):
                constraints.append(f"<= {metadata.le}")
    constraint_text = f" [{', '.join(constraints)}]" if constraints else ""

    lines.append(f"{prefix}- {name}{opt_tag}: {desc}{constraint_text}")
    return lines


def build_format_instructions(model: Optional[Type[BaseModel]] = None) -> str:
    """Walk the Pydantic schema and emit model-facing extraction instructions."""

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
            "Use null for any scalar field whose value cannot be determined from the report.",
            "Use [] only when the report clearly indicates that a list field has no items; otherwise use null when unknown.",
            "Do not invent information. Extract only what is explicitly stated or clearly implied by the report.",
            "For every confidence-tracked sub-model, populate field_confidence for each non-null extracted field.",
        ]
    )
    return "\n".join(lines)
