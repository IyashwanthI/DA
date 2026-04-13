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
