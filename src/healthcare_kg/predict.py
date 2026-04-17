from __future__ import annotations
import re
from healthcare_kg.loader import KGSnapshot, local_name
from healthcare_kg.models import DiseasePrediction

def _resolve_symptom_iris(kg: KGSnapshot, names: list[str]) -> set[str]:
    """Map user tokens (labels, local names, or raw IDs) to symptom IRIs."""
    name_lower = {n.strip().lower() for n in names}
    found: set[str] = set()
    
    for uri, lab in kg.labels.items():
        # FIX 1: Handle full URIs for types (e.g., http://.../ontology/Symptom)
        t = kg.types.get(uri, "")
        if t != "Symptom" and not str(t).endswith("Symptom"):
            continue
            
        ln = local_name(uri).lower() # e.g., 'symptom_d003371'
        ll = lab.lower()             # e.g., 'cough'
        
        # FIX 2: Extract the raw ID so users can input "D003371" directly
        raw_id = ln.split("_", 1)[-1] if "_" in ln else ln
        
        for key in name_lower:
            # 1. Exact match on local name, raw ID, or full label
            if key == ln or key == ll or key == raw_id:
                found.add(uri)
            # 2. Strict word boundary match on the label
            elif re.search(r'\b' + re.escape(key) + r'\b', ll):
                found.add(uri)
                
    return found

def disease_symptoms(kg: KGSnapshot, disease_iri: str) -> set[str]:
    g = kg.nx_graph
    out: set[str] = set()
    if disease_iri not in g:
        return out
    for _, succ, data in g.out_edges(disease_iri, data=True):
        rel = data.get("relation", "")
        # FIX 3: Handle full URIs for relations
        if rel == "hasSymptom" or str(rel).endswith("hasSymptom"):
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
    
    input_iris = _resolve_symptom_iris(kg, symptom_inputs)
    g = kg.nx_graph
    
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
                rel = data.get("relation", "")
                # FIX 3: Handle full URIs for relations
                if rel == "hasSymptom" or str(rel).endswith("hasSymptom"):
                    tier1.add(pred)
                    break

    tier2: set[str] = set()
    for d1 in tier1:
        for sym in disease_symptoms(kg, d1):
            if sym in input_iris or sym not in g:
                continue
            for pred in g.predecessors(sym):
                ed = g.get_edge_data(pred, sym)
                if not ed:
                    continue
                # FIX 3: Handle full URIs for relations
                ok = any(x.get("relation") == "hasSymptom" or str(x.get("relation", "")).endswith("hasSymptom") for x in ed.values())
                if ok and pred not in tier1:
                    tier2.add(pred)

    candidates: dict[str, int] = {}
    for d in tier1: candidates[d] = 1
    for d in tier2: candidates[d] = 2

    preds: list[DiseasePrediction] = []
    for d_iri, tier in candidates.items():
        # FIX 1: Handle full URIs for types
        t = kg.types.get(d_iri, "")
        if t != "Disease" and not str(t).endswith("Disease"):
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
    
    if not preds:
        return []

    top_score = preds[0].score
    
    # FIX 4: Lower the drop-off ratio. Better ID mapping means more symptoms per disease, 
    # which naturally lowers Jaccard scores. 0.70 is too aggressive now.
    drop_off_ratio = 0.40 
    dynamic_threshold = top_score * drop_off_ratio

    filtered_preds = [p for p in preds if p.score >= dynamic_threshold]

    return filtered_preds[:top_k]