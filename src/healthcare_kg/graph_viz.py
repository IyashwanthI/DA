"""Standalone HTML graph visualization (vis-network, embedded data)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any


def _b64_json_embed(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def write_disease_subgraph_html(payload: dict[str, Any], path: str | Path) -> None:
    """Write a single HTML file that opens offline; graph JSON is base64-embedded."""
    path = Path(path)
    b64 = _b64_json_embed(payload)
    meta = payload.get("meta") or {}
    title = "HCKG disease subgraph"
    if meta.get("n_seed_diseases"):
        title += f" ({meta['n_seed_diseases']} diseases)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; font-family: system-ui, Segoe UI, Roboto, sans-serif; background: #1a1d23; color: #e8eaed; }}
    #toolbar {{
      padding: 10px 14px; background: #252830; border-bottom: 1px solid #3d424d;
      display: flex; flex-wrap: wrap; align-items: center; gap: 12px 20px; font-size: 13px;
    }}
    #toolbar h1 {{ margin: 0; font-size: 15px; font-weight: 600; }}
    #toolbar .meta {{ opacity: 0.85; }}
    #legend {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }}
    #legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; }}
    #graph {{ width: 100%; height: calc(100vh - 52px); background: #12141a; }}
    a {{ color: #8ab4f8; }}
  </style>
</head>
<body>
  <div id="toolbar">
    <h1>{title}</h1>
    <span class="meta" id="stats"></span>
    <div id="legend">
      <span><i class="dot" style="background:#e74c3c"></i> Disease</span>
      <span><i class="dot" style="background:#3498db"></i> Symptom</span>
      <span><i class="dot" style="background:#27ae60"></i> Drug</span>
      <span style="border-left:1px solid #555;padding-left:14px">Edges: <span style="color:#95a5a6">hasSymptom</span> · <span style="color:#e67e22">treatedBy</span></span>
    </div>
  </div>
  <div id="graph"></div>
  <script>
(function() {{
  const payload = JSON.parse(atob("{b64}"));
  const el = payload.elements || {{}};
  const rawNodes = el.nodes || [];
  const rawEdges = el.edges || [];

  const COLORS = {{ Disease: "#e74c3c", Symptom: "#3498db", Drug: "#27ae60", Unknown: "#9b59b6" }};
  const nodes = rawNodes.map(function (n) {{
    const d = n.data || {{}};
    const t = d.type || "Unknown";
    return {{
      id: d.id,
      label: d.label || String(d.id).split("/").pop(),
      title: d.type + "\\n" + d.label,
      color: {{ background: COLORS[t] || COLORS.Unknown, border: "#2c3e50", highlight: {{ background: "#fff", border: "#2c3e50" }} }},
      font: {{ color: "#fff", size: 13 }},
      margin: 8,
      shape: t === "Drug" ? "box" : "dot",
      size: t === "Disease" ? 22 : (t === "Symptom" ? 10 : 14),
    }};
  }});

  const edges = rawEdges.map(function (e, i) {{
    const d = e.data || {{}};
    const rel = d.label || "";
    const isSymptom = rel === "hasSymptom";
    return {{
      id: "e" + i,
      from: d.source,
      to: d.target,
      label: rel,
      title: rel,
      color: {{ color: isSymptom ? "#7f8c8d" : "#e67e22" }},
      dashes: isSymptom,
      arrows: "to",
      font: {{ align: "middle", size: 10, color: "#bdc3c7", strokeWidth: 0 }},
      smooth: {{ type: "continuous", roundness: 0.35 }},
    }};
  }});

  const container = document.getElementById("graph");
  const data = {{ nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) }};
  const meta = payload.meta || {{}};
  document.getElementById("stats").textContent =
    (meta.node_count != null ? meta.node_count : nodes.length) + " nodes · " +
    (meta.edge_count != null ? meta.edge_count : edges.length) + " edges";

  const options = {{
    physics: {{
      enabled: true,
      barnesHut: {{
        gravitationalConstant: -12000,
        centralGravity: 0.35,
        springLength: 140,
        springConstant: 0.06,
        damping: 0.5,
      }},
      stabilization: {{ iterations: 200, updateInterval: 25 }},
    }},
    interaction: {{ hover: true, tooltipDelay: 120, zoomView: true, dragView: true }},
    layout: {{ improvedLayout: true }},
  }};

  const network = new vis.Network(container, data, options);
  network.once("stabilizationIterationsDone", function () {{
    network.setOptions({{ physics: false }});
  }});
}})();
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
