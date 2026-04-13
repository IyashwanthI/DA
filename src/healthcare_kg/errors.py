from __future__ import annotations

from collections import defaultdict

from healthcare_kg.loader import KGSnapshot, local_name
from healthcare_kg.models import DetectedError, Triple


def detect_errors(kg: KGSnapshot) -> list[DetectedError]:
    """Phase 1: rule-based structural, semantic, and missing-link checks."""
    errors: list[DetectedError] = []
    g = kg.nx_graph
    types = kg.types
    triple_counts = kg.triple_counts

    # --- Structural: duplicate triples
    for t_tuple, count in triple_counts.items():
        if count > 1:
            s, rel, o = t_tuple
            errors.append(
                DetectedError(
                    kind="structural",
                    code="duplicate_triple",
                    message=f"Identical triple appears {count} times",
                    triple=Triple(s, rel, o),
                )
            )

    # --- Structural: self-loops on clinical relations
    for t in kg.triples:
        if t.subject == t.object:
            errors.append(
                DetectedError(
                    kind="structural",
                    code="self_loop",
                    message="Subject and object are the same IRI",
                    triple=t,
                )
            )

    # --- Structural: orphan nodes (no clinical edges)
    clinical_nodes = set(g.nodes())
    for uri, _t in types.items():
        if uri not in clinical_nodes:
            errors.append(
                DetectedError(
                    kind="structural",
                    code="orphan_node",
                    message=f"Typed {types[uri]} has no hasSymptom/treatedBy edges",
                    nodes=(uri,),
                )
            )

    # --- Semantic: domain/range style checks
    for t in kg.triples:
        st = types.get(t.subject)
        ot = types.get(t.object)
        if t.relation == "hasSymptom":
            if st and st != "Disease":
                errors.append(
                    DetectedError(
                        kind="semantic",
                        code="hasSymptom_wrong_subject_type",
                        message=f"hasSymptom subject typed as {st}, expected Disease",
                        triple=t,
                    )
                )
            if ot and ot != "Symptom":
                errors.append(
                    DetectedError(
                        kind="semantic",
                        code="hasSymptom_wrong_object_type",
                        message=f"hasSymptom object typed as {ot}, expected Symptom",
                        triple=t,
                    )
                )
        elif t.relation == "treatedBy":
            if st and st != "Disease":
                errors.append(
                    DetectedError(
                        kind="semantic",
                        code="treatedBy_wrong_subject_type",
                        message=f"treatedBy subject typed as {st}, expected Disease",
                        triple=t,
                    )
                )
            if ot and ot != "Drug":
                errors.append(
                    DetectedError(
                        kind="semantic",
                        code="treatedBy_wrong_object_type",
                        message=f"treatedBy object typed as {ot}, expected Drug (not Symptom)",
                        triple=t,
                    )
                )

    # --- Missing: diseases without symptoms
    disease_has_symptom: defaultdict[str, int] = defaultdict(int)
    for t in kg.triples:
        if t.relation == "hasSymptom" and types.get(t.subject) == "Disease":
            disease_has_symptom[t.subject] += 1

    for uri, tname in types.items():
        if tname != "Disease":
            continue
        if disease_has_symptom[uri] == 0:
            errors.append(
                DetectedError(
                    kind="missing",
                    code="disease_no_symptoms",
                    message=f"Disease has no hasSymptom edges: {kg.labels.get(uri, local_name(uri))}",
                    nodes=(uri,),
                )
            )

    # --- Missing: symptoms never linked from any disease
    symptom_linked: set[str] = set()
    for t in kg.triples:
        if t.relation == "hasSymptom":
            symptom_linked.add(t.object)

    for uri, tname in types.items():
        if tname != "Symptom":
            continue
        if uri not in symptom_linked:
            errors.append(
                DetectedError(
                    kind="missing",
                    code="symptom_no_disease",
                    message=f"Symptom never appears as object of hasSymptom: {kg.labels.get(uri, local_name(uri))}",
                    nodes=(uri,),
                )
            )

    return errors


def errors_to_suspicious_triples(errors: list[DetectedError]) -> list[Triple]:
    """Unique triples flagged by detectors (excludes node-only issues)."""
    seen: set[tuple[str, str, str]] = set()
    out: list[Triple] = []
    for e in errors:
        if e.triple is None:
            continue
        key = e.triple.as_tuple()
        if key in seen:
            continue
        seen.add(key)
        out.append(e.triple)
    return out
