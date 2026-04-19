"""
Disease prediction UI (Phase 5). Run from project root:

    pip install -e ".[ui]"
    streamlit run src/healthcare_kg/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from healthcare_kg.drug_predict import predict_drugs_for_disease
from healthcare_kg.explain import explain_with_gemini
from healthcare_kg.gnn_predict import GNNDrugRecommender
from healthcare_kg.loader import load_owl
from healthcare_kg.predict import predict_diseases

DEFAULT_OWL = Path(__file__).resolve().parents[2] / "healthcare.owl"
DEFAULT_GNN_DIR = Path(__file__).resolve().parents[3] / "gnn"


@st.cache_resource(show_spinner=False)
def load_gnn_recommender(artifacts_dir: str) -> GNNDrugRecommender:
    return GNNDrugRecommender.from_artifacts(artifacts_dir)


def main() -> None:
    st.set_page_config(page_title="Healthcare KG — Disease prediction", layout="wide")
    st.title("Healthcare knowledge graph")
    tab1, tab2 = st.tabs(["Symptoms -> Diseases", "Disease -> Drugs (GNN)"])

    with tab1:
        st.caption("Symptom -> ranked diseases (Jaccard + hop-weighted scoring on healthcare.owl)")

        col1, col2 = st.columns([2, 1])
        with col1:
            owl_path = st.text_input("OWL / RDF path", value=str(DEFAULT_OWL), key="owl_input")
        with col2:
            top_k = st.number_input("Top k", min_value=1, max_value=50, value=10, key="disease_top_k")

        symptoms = st.text_input(
            "Symptoms (comma-separated)",
            value="fever, diarrhea, abdominal pain",
            help="Labels or ontology local names, e.g. fever, nausea",
            key="symptoms_input",
        )

        if st.button("Predict diseases", type="primary"):
            path = Path(owl_path)
            if not path.is_file():
                st.error(f"File not found: {path}")
                return
            with st.spinner("Loading graph..."):
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
            st.session_state["last_symptoms"] = tokens
            st.session_state["last_disease_preds"] = preds

        if st.button("Explain with GEMINI", key="explain_disease_btn"):
            preds = st.session_state.get("last_disease_preds", [])
            tokens = st.session_state.get("last_symptoms", [])
            if not preds:
                st.info("Run disease prediction first to generate explanation context.")
            else:
                try:
                    with st.spinner("Generating explanation with GEMINI..."):
                        note = explain_with_gemini(tokens, preds)
                    st.markdown("### GEMINIExplanation")
                    st.write(note)
                except Exception as exc:
                    st.error(str(exc))

    with tab2:
        st.caption("Disease -> drugs using both KG rules and GNN, with comparison")
        owl_path_for_drugs = st.text_input(
            "OWL / RDF path for KG comparison",
            value=str(DEFAULT_OWL),
            key="owl_drugs_input",
        )
        artifacts_dir = st.text_input(
            "GNN artifacts folder",
            value=str(DEFAULT_GNN_DIR),
            help="Folder containing entity_maps.pkl, healthcare_gnn.pth, hckg_with_inverses.owl",
            key="gnn_dir_input",
        )
        disease_name = st.text_input(
            "Disease name",
            value="brain cancer",
            help="Use disease label from the ontology training labels.",
            key="disease_name_input",
        )
        gnn_top_k = st.number_input("Top k drugs", min_value=1, max_value=50, value=5, key="gnn_top_k")

        if st.button("Predict drugs", type="primary"):
            try:
                kg_path = Path(owl_path_for_drugs)
                if not kg_path.is_file():
                    st.error(f"File not found: {kg_path}")
                    return
                with st.spinner("Loading KG for rule-based drug prediction..."):
                    kg = load_owl(kg_path)
                rule_recs = predict_drugs_for_disease(kg, disease_name, top_k=int(gnn_top_k))

                with st.spinner("Loading GNN model artifacts..."):
                    recommender = load_gnn_recommender(artifacts_dir)
                with st.spinner("Running GNN inference..."):
                    gnn_recs = recommender.recommend(disease_name, top_k=int(gnn_top_k))
            except Exception as exc:
                st.error(str(exc))
                return

            if not rule_recs and not gnn_recs:
                st.warning("No drug recommendations were produced by either method.")
                return

            col_rule, col_gnn = st.columns(2)
            with col_rule:
                st.subheader("KG rule-based")
                rows = [
                    {"rank": i + 1, "drug": r.drug_label, "score": r.score, "drug_iri": r.drug_iri}
                    for i, r in enumerate(rule_recs)
                ]
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No direct treatedBy/treats drug edges found for this disease.")

            with col_gnn:
                st.subheader("GNN")
                rows = [
                    {"rank": i + 1, "drug": r.drug_label, "score": r.score, "drug_iri": r.drug_iri}
                    for i, r in enumerate(gnn_recs)
                ]
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No GNN recommendations produced.")

            rule_iris = {r.drug_iri for r in rule_recs}
            gnn_iris = {r.drug_iri for r in gnn_recs}
            overlap_iris = rule_iris & gnn_iris
            union_iris = rule_iris | gnn_iris
            jaccard = (len(overlap_iris) / len(union_iris)) if union_iris else 0.0

            st.subheader("Comparison")
            st.metric("Overlap (Jaccard)", f"{jaccard:.3f}")
            overlap_rows = [
                {"rank": i + 1, "drug": r.drug_label, "score": r.score, "drug_iri": r.drug_iri}
                for i, r in enumerate(rule_recs)
                if r.drug_iri in overlap_iris
            ]
            if overlap_rows:
                st.write("Common drugs in both KG and GNN results:")
                st.dataframe(overlap_rows, use_container_width=True, hide_index=True)
            else:
                st.write("No overlap between KG and GNN top-k predictions.")


main()
