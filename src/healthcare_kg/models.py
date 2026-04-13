from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ErrorKind = Literal["structural", "semantic", "missing"]


@dataclass(frozen=True)
class Triple:
    """Clinical edge: subject IRI, relation local name, object IRI."""

    subject: str
    relation: str  # hasSymptom | treatedBy
    object: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.subject, self.relation, self.object)


@dataclass
class DetectedError:
    kind: ErrorKind
    code: str
    message: str
    triple: Triple | None = None
    nodes: tuple[str, ...] = ()


@dataclass
class LLMVerdict:
    provider: str
    raw_text: str
    valid: bool | None
    corrected_triple: tuple[str, str, str] | None
    confidence: float | None
    parsed: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    triple: Triple
    verdicts: list[LLMVerdict]
    accepted_correction: tuple[str, str, str] | None
    status: Literal["auto_accepted", "manual_review", "skipped_no_apis", "agreed_valid"]
    votes: dict[str, int] = field(default_factory=dict)


@dataclass
class CorrectionRecord:
    original: Triple
    final_triple: Triple | None
    status: str
    llm_attribution: dict[str, Any]
    manual: bool


@dataclass
class DiseasePrediction:
    disease_iri: str
    disease_label: str
    score: float
    jaccard: float
    hop_tier: int
    matched_symptoms: list[str]
