"""Rule-based disease-to-drug recommendation over the clinical graph."""

from __future__ import annotations

import re

from healthcare_kg.loader import KGSnapshot, local_name
from healthcare_kg.models import DrugRecommendation


def _resolve_disease_iris(kg: KGSnapshot, names: list[str]) -> set[str]:
    """Map user tokens (labels, local names, or raw IDs) to disease IRIs."""
    name_lower = {n.strip().lower() for n in names}
    found: set[str] = set()

    for uri, label in kg.labels.items():
        t = kg.types.get(uri, "")
        if t != "Disease" and not str(t).endswith("Disease"):
            continue

        ln = local_name(uri).lower()
        ll = label.lower()
        raw_id = ln.split("_", 1)[-1] if "_" in ln else ln

        for key in name_lower:
            if key == ln or key == ll or key == raw_id:
                found.add(uri)
            elif re.search(r"\b" + re.escape(key) + r"\b", ll):
                found.add(uri)

    return found


def disease_symptoms(kg: KGSnapshot, disease_iri: str) -> set[str]:
    g = kg.nx_graph
    out: set[str] = set()
    if disease_iri not in g:
        return out
    for _, succ, data in g.out_edges(disease_iri, data=True):
        rel = str(data.get("relation", ""))
        if rel == "hasSymptom" or rel.endswith("hasSymptom"):
            out.add(succ)
    return out


def disease_drugs(kg: KGSnapshot, disease_iri: str) -> set[str]:
    g = kg.nx_graph
    out: set[str] = set()
    if disease_iri not in g:
        return out

    for _, succ, data in g.out_edges(disease_iri, data=True):
        rel = str(data.get("relation", ""))
        if rel == "treatedBy" or rel.endswith("treatedBy"):
            out.add(succ)

    for pred, _, data in g.in_edges(disease_iri, data=True):
        rel = str(data.get("relation", ""))
        if rel == "treats" or rel.endswith("treats"):
            out.add(pred)
    return out


def drug_treated_diseases(kg: KGSnapshot, drug_iri: str) -> set[str]:
    g = kg.nx_graph
    out: set[str] = set()
    if drug_iri not in g:
        return out

    for _, succ, data in g.out_edges(drug_iri, data=True):
        rel = str(data.get("relation", ""))
        if rel == "treats" or rel.endswith("treats"):
            out.add(succ)

    for pred, _, data in g.in_edges(drug_iri, data=True):
        rel = str(data.get("relation", ""))
        if rel == "treatedBy" or rel.endswith("treatedBy"):
            out.add(pred)
    return out


def drug_symptoms(kg: KGSnapshot, drug_iri: str) -> set[str]:
    out: set[str] = set()
    for disease_iri in drug_treated_diseases(kg, drug_iri):
        out |= disease_symptoms(kg, disease_iri)
    return out


def predict_drugs_for_disease(
    kg: KGSnapshot,
    disease_query: str,
    top_k: int = 10,
    hop2_penalty: float = 0.55,
    jaccard_weight: float = 0.65,
    coverage_weight: float = 0.35,
) -> list[DrugRecommendation]:
    input_disease_iris = _resolve_disease_iris(kg, [disease_query])
    g = kg.nx_graph
    input_disease_iris = {d for d in input_disease_iris if d in g}
    if not input_disease_iris:
        return []

    # For a single query we use the best lexical match deterministically.
    query_disease_iri = sorted(input_disease_iris, key=lambda d: kg.labels.get(d, local_name(d)))[0]
    query_symptoms = disease_symptoms(kg, query_disease_iri)

    tier1 = disease_drugs(kg, query_disease_iri)

    related_diseases: set[str] = set()
    for s_iri in query_symptoms:
        for pred in g.predecessors(s_iri):
            edge_data = g.get_edge_data(pred, s_iri)
            if not edge_data:
                continue
            ok = any(
                x.get("relation") == "hasSymptom" or str(x.get("relation", "")).endswith("hasSymptom")
                for x in edge_data.values()
            )
            if ok and pred != query_disease_iri:
                related_diseases.add(pred)

    tier2: set[str] = set()
    for d_iri in related_diseases:
        for drug_iri in disease_drugs(kg, d_iri):
            if drug_iri not in tier1:
                tier2.add(drug_iri)

    candidates: dict[str, int] = {}
    for d in tier1:
        candidates[d] = 1
    for d in tier2:
        candidates[d] = 2

    preds: list[DrugRecommendation] = []
    for drug_iri, tier in candidates.items():
        t = kg.types.get(drug_iri, "")
        if t != "Drug" and not str(t).endswith("Drug"):
            continue

        drug_sym_set = drug_symptoms(kg, drug_iri)
        inter = len(drug_sym_set & query_symptoms)
        union = len(drug_sym_set | query_symptoms)

        jacc = inter / union if union else 0.0
        cov = inter / len(query_symptoms) if query_symptoms else 0.0
        hop_factor = 1.0 if tier == 1 else hop2_penalty
        score = (jaccard_weight * jacc + coverage_weight * cov) * hop_factor

        preds.append(
            DrugRecommendation(
                drug_iri=drug_iri,
                drug_label=kg.labels.get(drug_iri, local_name(drug_iri)).title(),
                score=round(score, 6),
            )
        )

    preds.sort(key=lambda p: (-p.score, p.drug_label))
    if not preds:
        return []

    top_score = preds[0].score
    drop_off_ratio = 0.40
    dynamic_threshold = top_score * drop_off_ratio
    filtered_preds = [p for p in preds if p.score >= dynamic_threshold]

    return filtered_preds[:top_k]

