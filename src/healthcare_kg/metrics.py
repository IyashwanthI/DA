"""Evaluation metrics for disease prediction quality and baselines."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from healthcare_kg.loader import KGSnapshot, local_name
from healthcare_kg.predict import predict_diseases


@dataclass
class EvalExample:
    symptoms: list[str]
    gold_disease_iri: str | None = None
    """Full IRI or local name of the correct disease."""


@dataclass
class MetricReport:
    precision_micro: float
    recall_micro: float
    f1_micro: float
    mrr: float
    prediction_entropy: float
    random_precision: float
    random_entropy: float
    per_class: dict[str, dict[str, float]]


def _normalize_disease_ref(kg: KGSnapshot, ref: str) -> str | None:
    ref = str(ref).strip()
    
    # 1. Direct IRI match
    str_types = {str(k): str(v) for k, v in kg.types.items()}
    if ref in str_types:
        return ref
        
    target = ref.lower()
    
    # NEW FIX: Create a space-separated version to match RDF Labels
    # (Turns "ulcerative_colitis" into "ulcerative colitis")
    target_spaced = target.replace("_", " ")
    
    raw_id = target.split("_", 1)[-1] if "_" in target else target
    raw_id = raw_id.replace(":", "_")
    
    for uri, t in kg.types.items():
        uri_str = str(uri) 
        t_str = str(t)     
        
        if t_str != "Disease" and not t_str.endswith("Disease"):
            continue
            
        ln = local_name(uri_str).lower()
        
        # 2. Match on exact local name
        if ln == target:
            return uri_str
            
        # 3. Match on raw ID
        ln_raw_id = ln.split("_", 1)[-1] if "_" in ln else ln
        if ln_raw_id == raw_id:
            return uri_str
            
        # 4. Match on human-readable label (Comparing against the spaced version!)
        lab = str(kg.labels.get(uri, "")).lower() 
        if lab == target or lab == target_spaced:
            return uri_str
            
    return None


def _rank_of_gold(ranked_iris: list[str], gold: str) -> int | None:
    try:
        return ranked_iris.index(gold) + 1
    except ValueError:
        return None


def _binary_entropy(p: float) -> float:
    p = min(1.0, max(0.0, p))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def evaluate(
    kg: KGSnapshot,
    examples: list[EvalExample],
    top_k: int = 5,
    random_trials: int = 50,
    seed: int = 42,
    predict_fn: Callable[..., list] | None = None,
) -> MetricReport:
    """
    Phase 4: precision/recall/F1 (micro), MRR, entropy of top-1 hit rate vs random baseline.
    """
    predict_fn = predict_fn or (lambda **kw: predict_diseases(**kw))

    rng = random.Random(seed)
    all_diseases = [u for u, t in kg.types.items() if t == "Disease"]

    tp = fp = fn = 0
    rr_sum = 0.0
    n_mrr = 0
    top1_hits = 0

    random_tp = 0
    random_top1 = 0
    per_tp: defaultdict[str, int] = defaultdict(int)
    per_fp: defaultdict[str, int] = defaultdict(int)
    per_fn: defaultdict[str, int] = defaultdict(int)

    resolved_examples: list[tuple[EvalExample, str]] = []
    for ex in examples:
        gold = _normalize_disease_ref(kg, ex.gold_disease_iri or "")
        if not gold:
            continue
        resolved_examples.append((ex, gold))
    # --- DIAGNOSTIC PRINT ---
    print(f"\n[DEBUG] Total examples in JSON: {len(examples)}")
    print(f"[DEBUG] Successfully resolved to graph URIs: {len(resolved_examples)}")
    if len(resolved_examples) == 0:
        print("[DEBUG] 🚨 STOP! All ground truth examples failed to match graph nodes.")
    # ------------------------

    for ex, gold_iri in resolved_examples:
        ranked = predict_fn(kg=kg, symptom_inputs=ex.symptoms, top_k=top_k)
        
        # Force string cast on predictions to match gold_iri
        ranked_iris = [str(p.disease_iri) for p in ranked] 
        pred_set = set(ranked_iris)
        
        is_hit = gold_iri in pred_set
        
        # --- DIAGNOSTIC PRINT (Will print the first example to verify logic) ---
        if tp == 0 and fn == 0 and len(resolved_examples) > 0:
            print(f"[DEBUG] First Test Case Gold IRI: {gold_iri}")
            print(f"[DEBUG] Predictions found: {len(ranked_iris)}")
        # ------------------------

        if is_hit:
            tp += 1
            per_tp[gold_iri] += 1
        else:
            fn += 1
            per_fn[gold_iri] += 1
            
        fp += len(pred_set) - (1 if is_hit else 0)
        
        # ... [Rest of your evaluate loop remains exactly the same] ...
    for ex, gold_iri in resolved_examples:
        ranked = predict_fn(kg=kg, symptom_inputs=ex.symptoms, top_k=top_k)
        ranked_iris = [p.disease_iri for p in ranked]
        pred_set = set(ranked_iris)
        is_hit = gold_iri in pred_set
        if is_hit:
            tp += 1
            per_tp[gold_iri] += 1
        else:
            fn += 1
            per_fn[gold_iri] += 1
        fp += len(pred_set) - (1 if is_hit else 0)
        for d in pred_set:
            if d == gold_iri:
                continue
            per_fp[d] += 1

        rnk = _rank_of_gold(ranked_iris, gold_iri)
        if rnk is not None:
            rr_sum += 1.0 / rnk
            n_mrr += 1
        if ranked_iris and ranked_iris[0] == gold_iri:
            top1_hits += 1

        for _ in range(random_trials):
            guess = rng.choice(all_diseases) if all_diseases else None
            if guess is None:
                continue
            if guess == gold_iri:
                random_tp += 1
            if guess == gold_iri:
                random_top1 += 1

    n = len(resolved_examples)
    prec_denom = tp + fp
    rec_denom = tp + fn
    precision_micro = tp / prec_denom if prec_denom else 0.0
    recall_micro = tp / rec_denom if rec_denom else 0.0
    f1_micro = (
        2 * precision_micro * recall_micro / (precision_micro + recall_micro)
        if (precision_micro + recall_micro)
        else 0.0
    )
    mrr = rr_sum / n_mrr if n_mrr else 0.0
    p_top1 = top1_hits / n if n else 0.0
    prediction_entropy = _binary_entropy(p_top1)

    rnd_p = random_tp / (n * random_trials) if n and random_trials else 0.0
    rnd_top1 = random_top1 / (n * random_trials) if n and random_trials else 0.0
    random_entropy = _binary_entropy(rnd_top1)

    per_class: dict[str, dict[str, float]] = {}
    all_gold = {g for _, g in resolved_examples}
    for g in all_gold:
        tpc = per_tp[g]
        fpc = per_fp.get(g, 0)
        fnc = per_fn.get(g, 0)
        pg = tpc / (tpc + fpc) if (tpc + fpc) else 0.0
        rg = tpc / (tpc + fnc) if (tpc + fnc) else 0.0
        fg = 2 * pg * rg / (pg + rg) if (pg + rg) else 0.0
        per_class[g] = {"precision": pg, "recall": rg, "f1": fg}

    return MetricReport(
        precision_micro=precision_micro,
        recall_micro=recall_micro,
        f1_micro=f1_micro,
        mrr=mrr,
        prediction_entropy=prediction_entropy,
        random_precision=rnd_p,
        random_entropy=random_entropy,
        per_class=per_class,
    )

