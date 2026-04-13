# Healthcare Knowledge Graph (OWL)

Python pipeline for loading a **healthcare ontology** (`healthcare.owl`), **detecting KG errors**, optional **multi-LLM ensemble correction**, **symptom-based disease ranking** over the graph, and **evaluation** against a small ground-truth set with a random baseline.

The project follows a phased design: data ingestion → rule-based QA → (optional) LLM fixes → graph-based prediction → metrics and reports.

---

## Features

| Phase | What it does |
|-------|----------------|
| **0 — Data** | Load `healthcare.owl` (RDF/XML) with [RDFLib](https://rdflib.readthedocs.io/), build a [NetworkX](https://networkx.org/) view of clinical edges. |
| **1 — Detection** | Rule-based checks: structural (orphans, self-loops), semantic (wrong `rdf:type` for `hasSymptom` / `treatedBy`), missing links (diseases without symptoms, symptoms without diseases). |
| **2 — LLM ensemble** | Send suspicious triples to Claude, OpenAI, and Gemini; majority vote; auto-apply agreed corrections; queue disagreements for manual review. |
| **3 — Prediction** | Map user symptoms to ontology nodes, traverse `hasSymptom` (tier-1 and expanded tier-2 neighbors), score diseases with **Jaccard** overlap and a **hop penalty**. |
| **4 — Metrics** | Micro precision/recall/F1, MRR, binary top-1 entropy vs uniform random disease baseline. |
| **5 — Outputs** | JSON error report, KG diff / ensemble log, review queue, metrics table, optional CSV edge export, optional Streamlit UI. |

**Ontology conventions** (as used in `healthcare.owl`):

- Namespace: `http://healthcarekg.org/ontology/`
- Types: `Disease`, `Symptom`, `Drug`
- Properties: `hasSymptom` (Disease → Symptom), `treatedBy` (Disease → Drug)

---

## Requirements

- **Python 3.10+**
- **`healthcare.owl`** at the **repository root** (default), or pass `--owl` to CLI commands.

---

## Installation

From the project root:

```bash
pip install -e .
```

Optional UI:

```bash
pip install -e ".[ui]"
```

---

## Environment (LLM phase only)

Copy `.env.example` to `.env` and set keys if you use `correct` or `predict --explain`:

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude (`correct`, `--explain`) |
| `OPENAI_API_KEY` | GPT-style model in ensemble |
| `GOOGLE_API_KEY` | Gemini in ensemble |
| `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `GOOGLE_MODEL` | Optional overrides |

Without keys, ensemble calls **skip** live APIs (logged in `ensemble_log.json`).

---

## CLI

Entry point: **`healthcare-kg`** (or `python -m healthcare_kg.cli`).  
Use `--help` on any subcommand for options.

### Quick baseline (Phase 1 + 4–5)

```bash
python -m healthcare_kg.cli pipeline
```

Writes:

- `outputs/pre_correction_errors.json` — full issue list  
- `outputs/metrics.json` — metrics + per-disease breakdown  
- `outputs/pipeline_summary.json` — graph stats + roll-up  

### Phase 1 only

```bash
python -m healthcare_kg.cli detect
```

### Phase 2 — ensemble correction

```bash
python -m healthcare_kg.cli correct --limit 20
```

Produces `outputs/corrected_kg.ttl`, `kg_diff_report.json`, `ensemble_log.json`, `review_queue.json`.

### Phase 3 — prediction

```bash
python -m healthcare_kg.cli predict "fever, cough, fatigue" --top-k 10
python -m healthcare_kg.cli predict "nausea, headache" --corrected-ttl outputs/corrected_kg.ttl
python -m healthcare_kg.cli predict "fever, diarrhea" --explain
```

### Manual triple edits

1. Copy `data/manual_corrections.example.json` → `data/manual_corrections.json`  
2. Each item: `original` and `final` triples (`subject`, `relation`, `object`) as **full IRIs** or **local names** (e.g. `narcolepsy`, `fever`).

```bash
python -m healthcare_kg.cli apply-manual --edits data/manual_corrections.json
```

### Evaluation on corrected graph

```bash
python -m healthcare_kg.cli evaluate --corrected-ttl outputs/corrected_kg.ttl
```

### Export edges (CSV)

```bash
python -m healthcare_kg.cli export-edges --out outputs/clinical_edges.csv
```

### Streamlit UI

From project root:

```bash
streamlit run src/healthcare_kg/streamlit_app.py
```

---

## Codebase layout

```
DA-PROJECT/
├── healthcare.owl              # Source KG (RDF/XML)
├── pyproject.toml
├── requirements.txt
├── .env.example
├── data/
│   ├── ground_truth.json       # Eval: symptom lists → gold disease (local name or IRI)
│   └── manual_corrections.example.json
├── outputs/                    # Generated (gitignored recommended)
└── src/healthcare_kg/
    ├── __init__.py
    ├── cli.py                  # Typer CLI (all commands)
    ├── loader.py               # load_owl → KGSnapshot (RDF + NetworkX)
    ├── errors.py               # Phase 1 detection
    ├── llm_ensemble.py         # Phase 2: Claude / OpenAI / Gemini + voting
    ├── corrections.py          # Apply LLM/manual edits to RDF; rebuild snapshot
    ├── manual.py               # JSON-driven manual corrections
    ├── predict.py              # Phase 3 ranking
    ├── metrics.py              # Phase 4 evaluation + random baseline
    ├── reporting.py            # JSON writers (errors, diff, metrics, review queue)
    ├── explain.py              # Optional Claude explanation for top hits
    ├── models.py               # Dataclasses (triples, errors, predictions)
    └── streamlit_app.py        # Optional web UI
```

**Data flow (conceptual):**

1. `loader.load_owl` parses OWL and extracts `hasSymptom` / `treatedBy` into `KGSnapshot` (`rdf`, `nx_graph`, `labels`, `types`, `triples`).  
2. `errors.detect_errors` reads that snapshot and emits `DetectedError` records.  
3. `llm_ensemble.ensemble_triple` optionally rewrites triples; `corrections.apply_ensemble_to_rdf` mutates the RDF graph.  
4. `predict.predict_diseases` uses only nodes present in `nx_graph` (orphan typed entities are ignored for traversal).  
5. `metrics.evaluate` compares predictions to `data/ground_truth.json`.

---

## Ground truth format

`data/ground_truth.json` is a JSON array:

```json
[
  {
    "symptoms": ["fever", "diarrhea"],
    "gold_disease": "ulcerative_colitis"
  }
]
```

`gold_disease` may be a **local name** or a **full disease IRI**.

---

## Academic / report notes

- Treat `pre_correction_errors.json` as the **pre-correction baseline** for KG quality.  
- Use `kg_diff_report.json` and `ensemble_log.json` to attribute **automatic vs manual** changes.  
- Report **metrics.json** alongside a description of **ground-truth size** and **`--top-k`**, since micro precision is sensitive to how many diseases are returned per query.

---

## License

Add a license file if you distribute the project; the code and your `healthcare.owl` usage are your responsibility for compliance with data and API terms.
