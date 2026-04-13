from __future__ import annotations

from pathlib import Path

from rdflib import Graph, URIRef

from healthcare_kg.loader import CLINICAL_RELATIONS, EX, KGSnapshot
from healthcare_kg.models import CorrectionRecord, EnsembleResult, Triple


def _predicate_for_relation(rel: str):
    if rel == "hasSymptom":
        return EX.hasSymptom
    if rel == "treatedBy":
        return EX.treatedBy
    raise ValueError(f"Unknown relation {rel}")


def remove_triple_from_rdf(g: Graph, t: Triple) -> bool:
    p = _predicate_for_relation(t.relation)
    trip = (URIRef(t.subject), p, URIRef(t.object))
    if trip not in g:
        return False
    g.remove(trip)
    return True


def add_triple_to_rdf(g: Graph, t: Triple) -> None:
    p = _predicate_for_relation(t.relation)
    g.add((URIRef(t.subject), p, URIRef(t.object)))


def apply_ensemble_to_rdf(
    rdf: Graph,
    original: Triple,
    result: EnsembleResult,
) -> CorrectionRecord | None:
    """Apply auto-accepted correction to RDF graph. Returns record or None."""
    if result.status == "agreed_valid" or result.status == "skipped_no_apis":
        return CorrectionRecord(
            original=original,
            final_triple=None,
            status=result.status,
            llm_attribution={v.provider: v.parsed for v in result.verdicts},
            manual=False,
        )
    if result.status != "auto_accepted" or not result.accepted_correction:
        return None

    from healthcare_kg.llm_ensemble import corrected_triple_to_rdf_triple

    new_t = corrected_triple_to_rdf_triple(result.accepted_correction)
    remove_triple_from_rdf(rdf, original)
    add_triple_to_rdf(rdf, new_t)
    return CorrectionRecord(
        original=original,
        final_triple=new_t,
        status="auto_accepted",
        llm_attribution={v.provider: v.parsed for v in result.verdicts},
        manual=False,
    )


def rebuild_snapshot_from_rdf(rdf: Graph) -> KGSnapshot:
    """Rebuild KGSnapshot after RDF edits (re-parse clinical edges)."""
    triples: list[Triple] = []
    import networkx as nx
    from collections import Counter

    nx_graph: nx.MultiDiGraph = nx.MultiDiGraph()
    for s, p, o in rdf.triples((None, None, None)):
        if p not in CLINICAL_RELATIONS:
            continue
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        rel = CLINICAL_RELATIONS[p]
        t = Triple(str(s), rel, str(o))
        triples.append(t)
        nx_graph.add_edge(t.subject, t.object, relation=rel)

    types: dict[str, str] = {}
    from rdflib import RDF

    for tname in ("Disease", "Symptom", "Drug"):
        type_uri = EX[tname]
        for subj in rdf.subjects(RDF.type, type_uri):
            types[str(subj)] = tname

    from rdflib import RDFS

    labels: dict[str, str] = {}
    for s, _, o in rdf.triples((None, RDFS.label, None)):
        if isinstance(o, URIRef):
            continue
        labels[str(s)] = str(o)
    for node in nx_graph.nodes():
        if node not in labels:
            from healthcare_kg.loader import local_name

            labels[node] = local_name(node)

    triple_counts = Counter(t.as_tuple() for t in triples)
    return KGSnapshot(
        rdf=rdf,
        nx_graph=nx_graph,
        labels=labels,
        types=types,
        triples=triples,
        triple_counts=triple_counts,
    )


def save_rdf(g: Graph, path: str | Path, fmt: str = "turtle") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=path.as_posix(), format=fmt)
