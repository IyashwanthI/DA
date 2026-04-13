from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from healthcare_kg.corrections import apply_ensemble_to_rdf, rebuild_snapshot_from_rdf, save_rdf
from healthcare_kg.errors import detect_errors, errors_to_suspicious_triples
from healthcare_kg.explain import explain_with_claude
from healthcare_kg.loader import KGSnapshot, load_owl
from healthcare_kg.llm_ensemble import ensemble_triple
from healthcare_kg.metrics import EvalExample, evaluate
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

load_dotenv()

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _default_owl() -> Path:
    return Path(__file__).resolve().parents[2] / "healthcare.owl"


@app.command("detect")
def cmd_detect(
    owl: Path = typer.Option(_default_owl, "--owl", help="Path to healthcare.owl"),
    out: Path = typer.Option(
        Path("outputs/pre_correction_errors.json"),
        "--out",
        help="Write baseline error list JSON",
    ),
) -> None:
    """Phase 1: rule-based error detection (baseline)."""
    kg = load_owl(owl)
    errors = detect_errors(kg)
    write_pre_correction_report(errors, out)
    console.print(f"[green]Wrote[/green] {out} ({len(errors)} issues)")
    by_kind: dict[str, int] = {}
    for e in errors:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    console.print(by_kind)


@app.command("correct")
def cmd_correct(
    owl: Path = typer.Option(_default_owl, "--owl"),
    limit: int = typer.Option(20, "--limit", help="Max suspicious triples to send to LLMs"),
    out_ttl: Path = typer.Option(Path("outputs/corrected_kg.ttl"), "--out-ttl"),
    diff_path: Path = typer.Option(Path("outputs/kg_diff_report.json"), "--diff"),
    log_path: Path = typer.Option(Path("outputs/ensemble_log.json"), "--log"),
    review_path: Path = typer.Option(
        Path("outputs/review_queue.json"),
        "--review-out",
        help="Triples flagged for manual correction",
    ),
) -> None:
    """Phase 2: ensemble LLM correction + KG diff (requires API keys for live calls)."""
    kg = load_owl(owl)
    errors = detect_errors(kg)
    suspicious = errors_to_suspicious_triples(errors)[:limit]
    rdf = kg.rdf
    records: list[CorrectionRecord] = []
    ensemble_results: list[EnsembleResult] = []

    for t in suspicious:
        result = ensemble_triple(t)
        ensemble_results.append(result)
        if result.status == "agreed_valid":
            records.append(
                CorrectionRecord(
                    original=t,
                    final_triple=None,
                    status="agreed_valid",
                    llm_attribution={v.provider: v.parsed for v in result.verdicts},
                    manual=False,
                )
            )
            continue
        rec = apply_ensemble_to_rdf(rdf, t, result)
        if rec:
            records.append(rec)

    write_ensemble_log(ensemble_results, log_path)
    write_review_queue(ensemble_results, review_path)
    write_kg_diff(records, ensemble_results, diff_path)
    save_rdf(rdf, out_ttl)
    console.print(
        f"[green]Saved[/green] {out_ttl}, {diff_path}, {log_path}, {review_path}"
    )


def _load_kg(owl: Path, corrected: Path | None) -> KGSnapshot:
    from rdflib import Graph

    if corrected is not None and corrected.is_file():
        g = Graph()
        g.parse(corrected.as_posix(), format="turtle")
        return rebuild_snapshot_from_rdf(g)
    return load_owl(owl)


@app.command("predict")
def cmd_predict(
    symptoms: str = typer.Argument(..., help="Comma-separated symptom labels or IDs"),
    owl: Path = typer.Option(_default_owl, "--owl"),
    corrected: Path | None = typer.Option(
        None,
        "--corrected-ttl",
        help="Use corrected TTL from Phase 2 instead of original OWL",
    ),
    top_k: int = typer.Option(10, "--top-k"),
    explain: bool = typer.Option(False, "--explain", help="Optional Claude rationale for top candidates"),
) -> None:
    """Phase 3: symptom → ranked diseases (multi-hop + Jaccard scoring)."""
    kg = _load_kg(owl, corrected)

    sym_list = [s.strip() for s in symptoms.split(",") if s.strip()]
    preds = predict_diseases(kg, sym_list, top_k=top_k)
    table = Table(title="Ranked diseases")
    table.add_column("Rank")
    table.add_column("Disease")
    table.add_column("Score")
    table.add_column("Jaccard")
    table.add_column("Tier")
    table.add_column("Matched symptoms")
    for i, p in enumerate(preds, start=1):
        table.add_row(
            str(i),
            p.disease_label,
            str(p.score),
            str(p.jaccard),
            str(p.hop_tier),
            ", ".join(p.matched_symptoms[:6])
            + ("…" if len(p.matched_symptoms) > 6 else ""),
        )
    console.print(table)
    if explain and preds:
        console.print("[bold]LLM note[/bold]")
        console.print(explain_with_claude(sym_list, preds))


@app.command("apply-manual")
def cmd_apply_manual(
    owl: Path = typer.Option(_default_owl, "--owl"),
    edits: Path = typer.Option(
        Path("data/manual_corrections.json"),
        "--edits",
        help="JSON list of {original, final} triple specs (IRIs or local names)",
    ),
    out_ttl: Path = typer.Option(Path("outputs/manual_corrected_kg.ttl"), "--out-ttl"),
    diff_append: Path | None = typer.Option(
        None,
        "--append-diff",
        help="Optional JSON file to append manual diff rows to",
    ),
) -> None:
    """Apply domain-expert manual triple edits to a copy of the OWL-derived graph."""
    apply_manual_edits_file(owl, edits, out_ttl, diff_append)
    console.print(f"[green]Wrote[/green] {out_ttl}")


@app.command("export-edges")
def cmd_export_edges(
    owl: Path = typer.Option(_default_owl, "--owl"),
    out: Path = typer.Option(Path("outputs/clinical_edges.csv"), "--out"),
) -> None:
    """Export hasSymptom and treatedBy edges as CSV for spreadsheets or Neo4j import."""
    kg = load_owl(owl)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["subject_iri,relation,object_iri,subject_label,object_label\n"]
    for t in kg.triples:
        ls = kg.labels.get(t.subject, t.subject)
        lo = kg.labels.get(t.object, t.object)
        lines.append(
            f"{t.subject},{t.relation},{t.object},"
            f'"{ls.replace(chr(34), chr(39))}","{lo.replace(chr(34), chr(39))}"\n'
        )
    out.write_text("".join(lines), encoding="utf-8")
    console.print(f"[green]Wrote[/green] {out} ({len(kg.triples)} edges)")


@app.command("evaluate")
def cmd_evaluate(
    owl: Path = typer.Option(_default_owl, "--owl"),
    corrected: Path | None = typer.Option(
        None,
        "--corrected-ttl",
        help="Evaluate using corrected Turtle instead of --owl",
    ),
    ground_truth: Path = typer.Option(
        Path("data/ground_truth.json"),
        "--ground-truth",
    ),
    out: Path = typer.Option(Path("outputs/metrics.json"), "--out"),
    top_k: int = typer.Option(20, "--top-k"),
) -> None:
    """Phase 4–5: metrics vs ground truth + random baseline."""
    kg = _load_kg(owl, corrected)
    raw = json.loads(ground_truth.read_text(encoding="utf-8"))
    examples = [
        EvalExample(symptoms=row["symptoms"], gold_disease_iri=row["gold_disease"]) for row in raw
    ]
    report = evaluate(kg, examples, top_k=top_k)
    write_metrics_table(report, out)
    console.print_json(data={
        "micro_precision": report.precision_micro,
        "micro_recall": report.recall_micro,
        "micro_f1": report.f1_micro,
        "mrr": report.mrr,
        "model_top1_entropy": report.prediction_entropy,
        "random_precision_per_draw": report.random_precision,
        "random_top1_entropy": report.random_entropy,
    })
    console.print(f"[green]Wrote[/green] {out}")


@app.command("pipeline")
def cmd_pipeline(
    owl: Path = typer.Option(_default_owl, "--owl"),
    ground_truth: Path = typer.Option(Path("data/ground_truth.json"), "--ground-truth"),
    errors_out: Path = typer.Option(
        Path("outputs/pre_correction_errors.json"), "--errors-out"
    ),
    metrics_out: Path = typer.Option(Path("outputs/metrics.json"), "--metrics-out"),
    summary_out: Path = typer.Option(Path("outputs/pipeline_summary.json"), "--summary-out"),
    top_k: int = typer.Option(20, "--top-k"),
) -> None:
    """Run Phase 1 (detect) + Phase 4–5 (evaluate) on healthcare.owl; write summary JSON."""
    kg = load_owl(owl)
    errors = detect_errors(kg)
    write_pre_correction_report(errors, errors_out)

    raw = json.loads(ground_truth.read_text(encoding="utf-8"))
    examples = [
        EvalExample(symptoms=row["symptoms"], gold_disease_iri=row["gold_disease"]) for row in raw
    ]
    report = evaluate(kg, examples, top_k=top_k)
    write_metrics_table(report, metrics_out)

    by_kind: dict[str, int] = {}
    for e in errors:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1

    summary = {
        "owl": str(owl.resolve()),
        "clinical_edges": len(kg.triples),
        "typed_entities": len(kg.types),
        "errors_by_kind": by_kind,
        "artifacts": {
            "pre_correction_errors": str(errors_out),
            "metrics": str(metrics_out),
        },
        "metrics": {
            "micro_precision": report.precision_micro,
            "micro_recall": report.recall_micro,
            "micro_f1": report.f1_micro,
            "mrr": report.mrr,
            "model_top1_entropy": report.prediction_entropy,
            "random_precision_per_draw": report.random_precision,
            "random_top1_entropy": report.random_entropy,
        },
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"[green]Pipeline done.[/green] Summary: {summary_out}")
    console.print_json(data=summary["metrics"])


def main() -> None:
    app()


if __name__ == "__main__":
    main()
