"""OWL loader and graph snapshot construction helpers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
from rdflib import RDF, RDFS, Graph, Namespace, URIRef

from healthcare_kg.models import Triple

EX = Namespace("http://healthcarekg.org/ontology/")
OWL_THING = URIRef("http://www.w3.org/2002/07/owl#Thing")

CLINICAL_RELATIONS = {
    EX.hasSymptom: "hasSymptom",
    EX.treatedBy: "treatedBy",
    EX.isSymptomOf: "isSymptomOf", 
    EX.treats: "treats",
}

ENTITY_TYPES = ("Disease", "Symptom", "Drug")


@dataclass
class KGSnapshot:
    """RDF graph + NetworkX clinical view + lookup tables."""

    rdf: Graph
    nx_graph: nx.MultiDiGraph
    labels: dict[str, str]
    types: dict[str, str]
    triples: list[Triple]
    triple_counts: Counter[tuple[str, str, str]]


def local_name(uri: str) -> str:
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


def load_owl(path: str | Path) -> KGSnapshot:
    path = Path(path).resolve()
    g = Graph()
    g.parse(path.as_posix(), format="xml")

    types: dict[str, str] = {}
    for tname in ENTITY_TYPES:
        type_uri = EX[tname]
        for s in g.subjects(RDF.type, type_uri):
            types[str(s)] = tname

    labels: dict[str, str] = {}
    for s, _, o in g.triples((None, RDFS.label, None)):
        if isinstance(o, URIRef):
            continue
        labels[str(s)] = str(o)

    triples: list[Triple] = []
    nx_graph: nx.MultiDiGraph = nx.MultiDiGraph()

    for s, p, o in g.triples((None, None, None)):
        if p not in CLINICAL_RELATIONS:
            continue
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        rel = CLINICAL_RELATIONS[p]
        t = Triple(str(s), rel, str(o))
        triples.append(t)
        nx_graph.add_edge(t.subject, t.object, relation=rel)

    triple_counts = Counter(t.as_tuple() for t in triples)

    for node in nx_graph.nodes():
        if node not in labels:
            labels[node] = local_name(node)

    return KGSnapshot(
        rdf=g,
        nx_graph=nx_graph,
        labels=labels,
        types=types,
        triples=triples,
        triple_counts=triple_counts,
    )


def clinical_triples_from_graph(g: nx.MultiDiGraph) -> list[Triple]:
    out: list[Triple] = []
    for u, v, data in g.edges(data=True):
        rel = data.get("relation")
        if rel:
            out.append(Triple(u, rel, v))
    return out


def disease_seeded_clinical_subgraph(
    snapshot: KGSnapshot,
    n_diseases: int = 20,
    *,
    relations: frozenset[str] | None = None,
) -> nx.MultiDiGraph:
    """Subgraph over the first ``n_diseases`` diseases (IRIs sorted lexically).

    Includes only outgoing edges from those diseases whose relation is in
    ``relations`` (default: ``hasSymptom`` and ``treatedBy`` — the latter is the
    disease→drug link named ``treatedBy`` in ``hckg.owl``).
    """
    rels = relations or frozenset({"hasSymptom", "treatedBy"})
    disease_uris = sorted(uri for uri, t in snapshot.types.items() if t == "Disease")
    seeds = disease_uris[: max(0, n_diseases)]
    seed_set = set(seeds)
    g = nx.MultiDiGraph()
    for d in seeds:
        g.add_node(
            d,
            node_type="Disease",
            label=snapshot.labels.get(d, local_name(d)),
        )
    for u, v, key, data in snapshot.nx_graph.out_edges(seed_set, keys=True, data=True):
        rel = data.get("relation")
        if rel not in rels:
            continue
        nt = snapshot.types.get(v, "Unknown")
        g.add_node(
            v,
            node_type=nt,
            label=snapshot.labels.get(v, local_name(v)),
        )
        g.add_edge(u, v, key=key, relation=rel)
    return g


def subgraph_to_elements_json(g: nx.MultiDiGraph) -> dict:
    """Cytoscape.js-style ``elements`` document (nodes + edges)."""
    nodes: list[dict] = []
    for n, attr in g.nodes(data=True):
        nodes.append(
            {
                "data": {
                    "id": n,
                    "label": attr.get("label", local_name(n)),
                    "type": attr.get("node_type", "?"),
                }
            }
        )
    edges: list[dict] = []
    for u, v, key, data in g.edges(keys=True, data=True):
        rel = data.get("relation", "")
        eid = f"{u}|{rel}|{v}|{key}"
        edges.append(
            {
                "data": {
                    "id": eid,
                    "source": u,
                    "target": v,
                    "label": rel,
                }
            }
        )
    return {"elements": {"nodes": nodes, "edges": edges}}

