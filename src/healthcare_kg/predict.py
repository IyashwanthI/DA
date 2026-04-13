from __future__ import annotations

from healthcare_kg.loader import KGSnapshot, local_name
from healthcare_kg.models import DiseasePrediction


def _resolve_symptom_iris(kg: KGSnapshot, names: list[str]) -> set[str]:
    """Map user tokens (labels or local names) to symptom IRIs."""
    name_lower = {n.strip().lower(): n for n in names}
    found: set[str] = set()
    for uri, lab in kg.labels.items():
        if kg.types.get(uri) != "Symptom":
            continue
        ln = local_name(uri).lower()
        ll = lab.lower()
        for key in name_lower:
            if key == ln or key == ll or key in ll or ll in key:
                found.add(uri)
    return found


def disease_symptoms(kg: KGSnapshot, disease_iri: str) -> set[str]:
    g = kg.nx_graph
    out: set[str] = set()
    if disease_iri not in g:
        return out
    for _, succ, data in g.out_edges(disease_iri, data=True):
        if data.get("relation") == "hasSymptom":
            out.add(succ)
    return out


def predict_diseases(
    kg: KGSnapshot,
    symptom_inputs: list[str],
    top_k: int = 10,
    hop2_penalty: float = 0.55,
    jaccard_weight: float = 0.65,
    coverage_weight: float = 0.35,
) -> list[DiseasePrediction]:
    """
    Multi-hop style ranking: tier-1 diseases link to any resolved input symptom;
    tier-2 diseases share a symptom with a tier-1 disease (excluding input set).
    """
    input_iris = _resolve_symptom_iris(kg, symptom_inputs)
    g = kg.nx_graph
    # Orphan symptoms (typed but no hasSymptom/treatedBy edges) are not in the graph
    input_iris = {s for s in input_iris if s in g}
    if not input_iris:
        return []

    tier1: set[str] = set()
    for s_iri in input_iris:
        for pred in g.predecessors(s_iri):
            edge_data = g.get_edge_data(pred, s_iri)
            if not edge_data:
                continue
            for _k, data in edge_data.items():
                if data.get("relation") == "hasSymptom":
                    tier1.add(pred)
                    break

    tier2: set[str] = set()
    for d1 in tier1:
        for sym in disease_symptoms(kg, d1):
            if sym in input_iris:
                continue
            if sym not in g:
                continue
            for pred in g.predecessors(sym):
                ed = g.get_edge_data(pred, sym)
                if not ed:
                    continue
                ok = any(x.get("relation") == "hasSymptom" for x in ed.values())
                if ok and pred not in tier1:
                    tier2.add(pred)

    candidates: dict[str, int] = {}
    for d in tier1:
        candidates[d] = 1
    for d in tier2:
        candidates[d] = 2

    preds: list[DiseasePrediction] = []
    for d_iri, tier in candidates.items():
        if kg.types.get(d_iri) != "Disease":
            continue
        s_set = disease_symptoms(kg, d_iri)
        matched = sorted(s_set & input_iris, key=lambda u: kg.labels.get(u, u))
        inter = len(s_set & input_iris)
        union = len(s_set | input_iris)
        jacc = inter / union if union else 0.0
        cov = inter / len(input_iris) if input_iris else 0.0
        hop_factor = 1.0 if tier == 1 else hop2_penalty
        score = (jaccard_weight * jacc + coverage_weight * cov) * hop_factor
        preds.append(
            DiseasePrediction(
                disease_iri=d_iri,
                disease_label=kg.labels.get(d_iri, local_name(d_iri)),
                score=round(score, 6),
                jaccard=round(jacc, 6),
                hop_tier=tier,
                matched_symptoms=[kg.labels.get(m, local_name(m)) for m in matched],
            )
        )

    preds.sort(key=lambda p: (-p.score, -p.jaccard, p.disease_label))
    return preds[:top_k]
