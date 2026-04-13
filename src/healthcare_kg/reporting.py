from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from healthcare_kg.metrics import MetricReport
from healthcare_kg.models import CorrectionRecord, DetectedError, EnsembleResult, Triple


def errors_to_jsonable(errors: list[DetectedError]) -> list[dict[str, Any]]:
    out = []
    for e in errors:
        row: dict[str, Any] = {
            "kind": e.kind,
            "code": e.code,
            "message": e.message,
            "nodes": list(e.nodes),
        }
        if e.triple:
            row["triple"] = {
                "subject": e.triple.subject,
                "relation": e.triple.relation,
                "object": e.triple.object,
            }
        out.append(row)
    return out


def ensemble_to_jsonable(r: EnsembleResult) -> dict[str, Any]:
    return {
        "original": {
            "subject": r.triple.subject,
            "relation": r.triple.relation,
            "object": r.triple.object,
        },
        "status": r.status,
        "accepted_correction": list(r.accepted_correction) if r.accepted_correction else None,
        "votes": r.votes,
        "verdicts": [
            {
                "provider": v.provider,
                "valid": v.valid,
                "corrected_triple": list(v.corrected_triple) if v.corrected_triple else None,
                "confidence": v.confidence,
                "parsed": v.parsed,
            }
            for v in r.verdicts
        ],
    }


def write_pre_correction_report(errors: list[DetectedError], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(errors_to_jsonable(errors), indent=2), encoding="utf-8")


def write_review_queue(results: list[EnsembleResult], path: str | Path) -> None:
    """Export triples that need human review (LLMs disagreed or no majority)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = [ensemble_to_jsonable(r) for r in results if r.status == "manual_review"]
    path.write_text(json.dumps(pending, indent=2), encoding="utf-8")


def write_ensemble_log(results: list[EnsembleResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [ensemble_to_jsonable(r) for r in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_kg_diff(
    records: list[CorrectionRecord],
    ensemble_results: list[EnsembleResult],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        rows.append(
            {
                "original": asdict(r.original),
                "final": asdict(r.final_triple) if r.final_triple else None,
                "status": r.status,
                "manual": r.manual,
                "llm_attribution": r.llm_attribution,
            }
        )
    for er in ensemble_results:
        if er.status == "manual_review":
            rows.append(
                {
                    "original": {
                        "subject": er.triple.subject,
                        "relation": er.triple.relation,
                        "object": er.triple.object,
                    },
                    "final": None,
                    "status": "manual_review_pending",
                    "manual": True,
                    "llm_attribution": {v.provider: v.parsed for v in er.verdicts},
                }
            )
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_metrics_table(report: MetricReport, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "micro_precision": report.precision_micro,
        "micro_recall": report.recall_micro,
        "micro_f1": report.f1_micro,
        "mrr": report.mrr,
        "model_top1_entropy": report.prediction_entropy,
        "random_precision_per_draw": report.random_precision,
        "random_top1_entropy": report.random_entropy,
        "per_disease": report.per_class,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def triple_from_dict(d: dict[str, Any]) -> Triple:
    return Triple(d["subject"], d["relation"], d["object"])
