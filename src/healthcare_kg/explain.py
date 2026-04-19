"""LLM explanation helpers for summarizing top prediction rationale."""

from __future__ import annotations

import os
import time

import httpx

from healthcare_kg.models import DiseasePrediction
def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or default

def explain_with_gemini(
    symptoms: list[str],
    predictions: list[DiseasePrediction],
) -> str:
    key = <"insert API HERE"
    model = "gemini-2.5-flash"
    lines = [
        f"{i+1}. {p.disease_label} (score={p.score}, jaccard={p.jaccard}, tier={p.hop_tier})"
        for i, p in enumerate(predictions[:5])
    ]
    user = (
        f"Patient symptoms: {', '.join(symptoms)}.\n"
        f"Top KG-ranked diseases:\n" + "\n".join(lines) + "\n"
        "Which single diagnosis is most plausible and why? Answer in 3–5 sentences."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.4},
    }
    with httpx.Client(timeout=120.0) as client:
        max_retries = 5
        for attempt in range(max_retries):
            r = client.post(url, params={"key": key}, json=payload)

            # Retry with exponential backoff on API rate limits.
            if r.status_code == 429:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue

            r.raise_for_status()
            data = r.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))

        return "Gemini API rate limit reached repeatedly. Please retry in a few moments."


def explain_with_claude(
    symptoms: list[str],
    predictions: list[DiseasePrediction],
) -> str:
    """Backward compatible alias; now uses Gemini."""
    return explain_with_gemini(symptoms, predictions)

