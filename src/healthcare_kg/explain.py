from __future__ import annotations

import os

import httpx

from healthcare_kg.models import DiseasePrediction


def explain_with_claude(
    symptoms: list[str],
    predictions: list[DiseasePrediction],
) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return "Set ANTHROPIC_API_KEY to enable LLM explanation."
    lines = [
        f"{i+1}. {p.disease_label} (score={p.score}, jaccard={p.jaccard}, tier={p.hop_tier})"
        for i, p in enumerate(predictions[:5])
    ]
    user = (
        f"Patient symptoms: {', '.join(symptoms)}.\n"
        f"Top KG-ranked diseases:\n" + "\n".join(lines) + "\n"
        "Which single diagnosis is most plausible and why? Answer in 3–5 sentences."
    )
    payload = {
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        "max_tokens": 400,
        "messages": [{"role": "user", "content": user}],
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
    print(data)
    return "".join(b.get("text", "") for b in data.get("content", []) if isinstance(b, dict))
