"""GNN-based drug recommender loading model artifacts for inference."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdflib import Graph

from healthcare_kg.models import DrugRecommendation

try:
    import torch
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import SAGEConv, to_hetero
except Exception as exc:  # pragma: no cover - optional dependency guard
    torch = None
    HeteroData = None
    SAGEConv = None
    to_hetero = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


DEFAULT_MODEL_FILE = "healthcare_gnn.pth"
DEFAULT_MAPS_FILE = "entity_maps.pkl"
DEFAULT_OWL_FILE = "hckg_with_inverses.owl"


def _ensure_runtime() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "GNN inference requires 'torch' and 'torch-geometric'. "
            "Install them before using this feature."
        ) from _IMPORT_ERROR


def load_owl_with_saved_maps(file_path: Path, saved_maps: dict[str, dict[str, int]]) -> HeteroData:
    _ensure_runtime()

    g = Graph()
    g.parse(file_path.as_posix(), format="xml")
    data = HeteroData()

    edges: dict[tuple[str, str, str], list[list[int]]] = {
        ("disease", "hasSymptom", "symptom"): [],
        ("disease", "treatedBy", "drug"): [],
        ("symptom", "isSymptomOf", "disease"): [],
        ("drug", "treats", "disease"): [],
    }

    for s, p, o in g:
        rel = str(p).split("/")[-1]
        s_str, o_str = str(s), str(o)

        if rel == "hasSymptom" and s_str in saved_maps["disease"] and o_str in saved_maps["symptom"]:
            edges[("disease", "hasSymptom", "symptom")].append(
                [saved_maps["disease"][s_str], saved_maps["symptom"][o_str]]
            )
        elif rel == "treatedBy" and s_str in saved_maps["disease"] and o_str in saved_maps["drug"]:
            edges[("disease", "treatedBy", "drug")].append(
                [saved_maps["disease"][s_str], saved_maps["drug"][o_str]]
            )
        elif rel == "isSymptomOf" and s_str in saved_maps["symptom"] and o_str in saved_maps["disease"]:
            edges[("symptom", "isSymptomOf", "disease")].append(
                [saved_maps["symptom"][s_str], saved_maps["disease"][o_str]]
            )
        elif rel == "treats" and s_str in saved_maps["drug"] and o_str in saved_maps["disease"]:
            edges[("drug", "treats", "disease")].append(
                [saved_maps["drug"][s_str], saved_maps["disease"][o_str]]
            )

    for category, mapping in saved_maps.items():
        data[category].num_nodes = len(mapping)

    for edge_type, edge_list in edges.items():
        if edge_list:
            data[edge_type].edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

    return data


if torch is not None:
    class GNNEncoder(torch.nn.Module):
        def __init__(self, hidden_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), out_channels)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x


    class HeteroModel(torch.nn.Module):
        def __init__(
            self,
            metadata: tuple[list[str], list[tuple[str, str, str]]],
            num_nodes_dict: dict[str, int],
            hidden_channels: int,
            out_channels: int,
        ) -> None:
            super().__init__()
            self.emb_dict = torch.nn.ModuleDict(
                {
                    node_type: torch.nn.Embedding(num_nodes, hidden_channels)
                    for node_type, num_nodes in num_nodes_dict.items()
                }
            )
            self.encoder = to_hetero(GNNEncoder(hidden_channels, out_channels), metadata)

        def forward(self, edge_index_dict: dict, edge_label_index: torch.Tensor) -> torch.Tensor:
            x_dict = {node_type: self.emb_dict[node_type].weight for node_type in self.emb_dict.keys()}
            x_dict = self.encoder(x_dict, edge_index_dict)
            disease_x = x_dict["disease"][edge_label_index[0]]
            drug_x = x_dict["drug"][edge_label_index[1]]
            return (disease_x * drug_x).sum(dim=-1)
else:  # pragma: no cover - optional dependency guard
    HeteroModel = Any


@dataclass
class GNNDrugRecommender:
    model: HeteroModel
    data: HeteroData
    entity_maps: dict[str, dict[str, int]]
    label_to_uri: dict[str, str]

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str | Path,
        owl_file_name: str = DEFAULT_OWL_FILE,
        model_file_name: str = DEFAULT_MODEL_FILE,
        maps_file_name: str = DEFAULT_MAPS_FILE,
    ) -> "GNNDrugRecommender":
        _ensure_runtime()

        base = Path(artifacts_dir).resolve()
        owl_path = base / owl_file_name
        model_path = base / model_file_name
        maps_path = base / maps_file_name

        for p in (owl_path, model_path, maps_path):
            if not p.is_file():
                raise FileNotFoundError(f"Missing required artifact: {p}")

        with maps_path.open("rb") as f:
            saved_data = pickle.load(f)

        entity_maps = saved_data["maps"]
        label_to_uri = saved_data["labels"]
        data = load_owl_with_saved_maps(owl_path, entity_maps)

        num_nodes_dict = {
            "disease": len(entity_maps["disease"]),
            "drug": len(entity_maps["drug"]),
            "symptom": len(entity_maps["symptom"]),
        }

        model = HeteroModel(data.metadata(), num_nodes_dict, hidden_channels=64, out_channels=32)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        return cls(model=model, data=data, entity_maps=entity_maps, label_to_uri=label_to_uri)

    def recommend(self, disease_query: str, top_k: int = 5) -> list[DrugRecommendation]:
        query_clean = disease_query.lower().replace("_", " ").strip()
        disease_uri = self.label_to_uri.get(query_clean)
        if not disease_uri:
            raise ValueError(f"Disease '{disease_query}' not found in labels.")

        disease_id = self.entity_maps["disease"].get(disease_uri)
        if disease_id is None:
            raise ValueError(f"'{disease_query}' exists but is not mapped as a disease.")

        with torch.no_grad():
            x_dict = {node_type: self.model.emb_dict[node_type].weight for node_type in self.model.emb_dict.keys()}
            node_embeddings = self.model.encoder(x_dict, self.data.edge_index_dict)

            disease_emb = node_embeddings["disease"][disease_id].unsqueeze(0)
            drug_embs = node_embeddings["drug"]

            scores = torch.sigmoid(torch.matmul(disease_emb, drug_embs.t()).squeeze(0))
            top_probs, top_indices = torch.topk(scores, k=min(top_k, len(drug_embs)))

        inv_drug_map = {v: k for k, v in self.entity_maps["drug"].items()}
        uri_to_label = {v: k for k, v in self.label_to_uri.items()}

        recs: list[DrugRecommendation] = []
        for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
            drug_uri = inv_drug_map.get(int(idx), "Unknown")
            drug_label = uri_to_label.get(drug_uri, drug_uri.rsplit("/", 1)[-1])
            recs.append(
                DrugRecommendation(
                    drug_iri=drug_uri,
                    drug_label=drug_label.title(),
                    score=round(float(prob), 6),
                )
            )
        return recs

