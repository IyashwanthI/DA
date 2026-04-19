"""Manual correction loader and applier for curated triple edits."""

from __future__ import annotations

import json
from pathlib import Path
import shutil   
from healthcare_kg.corrections import add_triple_to_rdf, remove_triple_from_rdf, save_rdf
from healthcare_kg.loader import EX, load_owl
from healthcare_kg.models import CorrectionRecord, Triple


def _iri(part: str) -> str:
    p = part.strip()
    if p.startswith("http://") or p.startswith("https://"):
        return p
    return str(EX[p])


def triple_from_spec(d: dict) -> Triple:
    return Triple(_iri(d["subject"]), d["relation"].strip(), _iri(d["object"]))


def apply_manual_edits_file(
    owl_path: str | Path,
    edits_path: str | Path,
    out_ttl: str | Path,
    diff_append: str | Path | None = None,
) -> list[CorrectionRecord]:
    """
    Apply manual corrections from JSON. Each row:
    {
      "original": {subject, relation, object},
      "final": {subject, relation, object},
      "skip_if_missing": true  # optional, defaults to false
    }
    IRIs or ontology local names (e.g. brain_cancer, headache).
    """
    owl_path = Path(owl_path)
    edits_path = Path(edits_path)
    out_ttl = Path(out_ttl)
    kg = load_owl(owl_path)
    g = kg.rdf
    raw = json.loads(edits_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Manual edits JSON must be a list of edit objects")

    records: list[CorrectionRecord] = []
    diff_rows: list[dict] = []

    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            continue
        if "original" not in row or "final" not in row:
            continue
        orig = triple_from_spec(row["original"])
        fin = triple_from_spec(row["final"])
        skip_if_missing = bool(row.get("skip_if_missing", False))
        removed = remove_triple_from_rdf(g, orig)
        if not removed:
            if skip_if_missing:
                continue
            raise ValueError(f"Edit {i}: original triple not in graph: {orig}")
        add_triple_to_rdf(g, fin)
        rec = CorrectionRecord(
            original=orig,
            final_triple=fin,
            status="manual_applied",
            llm_attribution={},
            manual=True,
        )
        records.append(rec)
        diff_rows.append(
            {
                "original": {"subject": orig.subject, "relation": orig.relation, "object": orig.object},
                "final": {"subject": fin.subject, "relation": fin.relation, "object": fin.object},
                "status": "manual_applied",
                "manual": True,
                "llm_attribution": {},
            }
        )

    g.serialize(destination=str(owl_path), format="pretty-xml")
    if diff_append is not None and diff_rows:
        p = Path(diff_append)
        existing: list = []
        if p.is_file():
            existing = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(existing + diff_rows, indent=2), encoding="utf-8")

    return records

