"""
Disease prediction UI (Phase 5). Run from project root:

    pip install -e ".[ui]"
    streamlit run src/healthcare_kg/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from healthcare_kg.loader import load_owl
from healthcare_kg.predict import predict_diseases

DEFAULT_OWL = Path(__file__).resolve().parents[2] / "healthcare.owl"


def main() -> None:
    st.set_page_config(page_title="Healthcare KG — Disease prediction", layout="wide")
    st.title("Healthcare knowledge graph")
    st.caption("Symptom → ranked diseases (Jaccard + hop-weighted scoring on healthcare.owl)")

    col1, col2 = st.columns([2, 1])
    with col1:
        owl_path = st.text_input("OWL / RDF path", value=str(DEFAULT_OWL))
    with col2:
        top_k = st.number_input("Top k", min_value=1, max_value=50, value=10)

    symptoms = st.text_input(
        "Symptoms (comma-separated)",
        value="fever, diarrhea, abdominal pain",
        help="Labels or ontology local names, e.g. fever, nausea",
    )

    if st.button("Predict", type="primary"):
        path = Path(owl_path)
        if not path.is_file():
            st.error(f"File not found: {path}")
            return
        with st.spinner("Loading graph…"):
            kg = load_owl(path)
        tokens = [s.strip() for s in symptoms.split(",") if s.strip()]
        preds = predict_diseases(kg, tokens, top_k=int(top_k))
        if not preds:
            st.warning("No diseases matched. Try different symptom names.")
            return
        rows = [
            {
                "rank": i + 1,
                "disease": p.disease_label,
                "score": p.score,
                "jaccard": p.jaccard,
                "tier": p.hop_tier,
                "matched": ", ".join(p.matched_symptoms[:8]),
            }
            for i, p in enumerate(preds)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


main()
