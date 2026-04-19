"""Symptom-to-disease ranking logic using graph traversal and overlap scoring."""
from __future__ import annotations

import json
from pathlib import Path
import csv
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rdflib import Graph,URIRef
from healthcare_kg.loader import local_name
from healthcare_kg.corrections import apply_ensemble_to_rdf, rebuild_snapshot_from_rdf, save_rdf
from healthcare_kg.errors import detect_errors, errors_to_suspicious_triples
from healthcare_kg.explain import explain_with_gemini
from healthcare_kg.loader import KGSnapshot, load_owl
from healthcare_kg.llm_ensemble import ensemble_triple
from healthcare_kg.metrics import EvalExample, evaluate
from healthcare_kg.metrics_hckg import evaluatehckg
from healthcare_kg.metrics_healthcare import evaluatehealthcare
from healthcare_kg.models import CorrectionRecord, EnsembleResult
from healthcare_kg.predict import predict_diseases
from healthcare_kg.manual import apply_manual_edits_file
from healthcare_kg.reporting import (
    write_ensemble_log,
    write_kg_diff,
    write_metrics_table,
    write_pre_correction_report,
    write_review_queue,
)
from rdflib.namespace import RDFS

def _default_owl() -> Path:
    return Path(__file__).resolve().parents[2] / "hckg.owl"
pth=_default_owl()
kg=load_owl(pth)
from rdflib.namespace import RDFS

def export_triples_to_csv(owl_file, output_csv):
    g = Graph()
    g.parse(owl_file)

    relations = {
        URIRef("http://healthcarekg.org/ontology/hasSymptom"),
        URIRef("http://healthcarekg.org/ontology/treatedBy"),
    }

    # Build label map
    label_map = {}
    for s, p, o in g:
        if p == RDFS.label:
            label_map[s] = str(o)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "predicate", "object"])

        count = 0
        for s, p, o in g:
            if p in relations:
                sub = label_map.get(s, local_name(s))
                obj = label_map.get(o, local_name(o))

                writer.writerow([sub, local_name(p), obj])
                count += 1

    print(f"Exported {count} filtered triples")
export_triples_to_csv(_default_owl(), "hh.csv")