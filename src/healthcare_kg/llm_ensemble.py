from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any

import httpx

from healthcare_kg.loader import EX, local_name
from healthcare_kg.models import EnsembleResult, LLMVerdict, Triple

SYSTEM = (
    "You are a clinical knowledge graph validator. "
    "Respond with JSON only, no markdown. Keys: valid (boolean), "
    "corrected_triple (array of 3 strings: disease_id, relation, entity_id), "
    "confidence (0-1 float). Use relation exactly 'hasSymptom' or 'treatedBy'. "
    "Use short snake_case IDs like the ontology (e.g. brain_cancer, headache)."
)


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or default


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("No JSON object found in model output")
    return json.loads(m.group(0))


def _triple_prompt(t: Triple) -> str:
    subj = local_name(t.subject)
    obj = local_name(t.object)
    return (
        f'Given the medical triple ({subj}, {t.relation}, {obj}), is this valid? '
        "If not, what is the correct relation or entity? "
        'Respond in JSON: {"valid": bool, "corrected_triple": [str, str, str], "confidence": float}'
    )


def _normalize_corrected(ct: Any) -> tuple[str, str, str] | None:
    if not isinstance(ct, (list, tuple)) or len(ct) != 3:
        return None
    a, b, c = ct
    if not all(isinstance(x, str) for x in (a, b, c)):
        return None
    rel = b.strip()
    if rel not in ("hasSymptom", "treatedBy"):
        return None
    return (a.strip(), rel, c.strip())


def _iri_for_local(part: str) -> str:
    if part.startswith("http://") or part.startswith("https://"):
        return part
    return str(EX[part])


def call_claude(user_prompt: str) -> LLMVerdict:
    key = _env("ANTHROPIC_API_KEY")
    model = _env("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    if not key:
        return LLMVerdict(
            provider="claude",
            raw_text="",
            valid=None,
            corrected_triple=None,
            confidence=None,
            parsed={"skipped": "no ANTHROPIC_API_KEY"},
        )
    payload = {
        "model": model,
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": f"{SYSTEM}\n\n{user_prompt}"},
        ],
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
    text = "".join(b.get("text", "") for b in data.get("content", []) if isinstance(b, dict))
    return _parse_verdict("claude", text)


def call_openai(user_prompt: str) -> LLMVerdict:
    key = _env("OPENAI_API_KEY")
    model = _env("OPENAI_MODEL", "gpt-4o")
    if not key:
        return LLMVerdict(
            provider="openai",
            raw_text="",
            valid=None,
            corrected_triple=None,
            confidence=None,
            parsed={"skipped": "no OPENAI_API_KEY"},
        )
    payload = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
        )
        r.raise_for_status()
        data = r.json()
    text = data["choices"][0]["message"]["content"]
    return _parse_verdict("openai", text)


def call_gemini(user_prompt: str) -> LLMVerdict:
    key = _env("GOOGLE_API_KEY")
    model = _env("GOOGLE_MODEL", "gemini-1.5-flash")
    if not key:
        return LLMVerdict(
            provider="gemini",
            raw_text="",
            valid=None,
            corrected_triple=None,
            confidence=None,
            parsed={"skipped": "no GOOGLE_API_KEY"},
        )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM}\n\n{user_prompt}"}]}],
        "generationConfig": {"temperature": 0.4},
    }
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, params={"key": key}, json=payload)
        r.raise_for_status()
        data = r.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return _parse_verdict("gemini", text)


def _parse_verdict(provider: str, text: str) -> LLMVerdict:
    try:
        obj = _extract_json_object(text)
    except (json.JSONDecodeError, ValueError) as e:
        return LLMVerdict(
            provider=provider,
            raw_text=text,
            valid=None,
            corrected_triple=None,
            confidence=None,
            parsed={"parse_error": str(e)},
        )
    valid = obj.get("valid")
    if not isinstance(valid, bool):
        valid = None
    ct = _normalize_corrected(obj.get("corrected_triple"))
    conf = obj.get("confidence")
    if isinstance(conf, (int, float)):
        confidence = float(conf)
    else:
        confidence = None
    return LLMVerdict(
        provider=provider,
        raw_text=text,
        valid=valid,
        corrected_triple=ct,
        confidence=confidence,
        parsed=obj,
    )


def ensemble_triple(t: Triple) -> EnsembleResult:
    prompt = _triple_prompt(t)
    verdicts = [
        call_claude(prompt),
        call_openai(prompt),
        call_gemini(prompt),
    ]
    active = [v for v in verdicts if v.valid is not None]
    if not active:
        return EnsembleResult(
            triple=t,
            verdicts=verdicts,
            accepted_correction=None,
            status="skipped_no_apis",
            votes={},
        )

    valid_votes = sum(1 for v in active if v.valid is True)
    invalid_votes = sum(1 for v in active if v.valid is False)

    if valid_votes >= 2:
        return EnsembleResult(
            triple=t,
            verdicts=verdicts,
            accepted_correction=None,
            status="agreed_valid",
            votes={"valid": valid_votes, "invalid": invalid_votes},
        )

    if invalid_votes < 2:
        return EnsembleResult(
            triple=t,
            verdicts=verdicts,
            accepted_correction=None,
            status="manual_review",
            votes={"valid": valid_votes, "invalid": invalid_votes},
        )

    corrected_counts: Counter[tuple[str, str, str]] = Counter()
    for v in active:
        if v.valid is False and v.corrected_triple:
            corrected_counts[v.corrected_triple] += 1

    if corrected_counts:
        best, cnt = corrected_counts.most_common(1)[0]
        if cnt >= 2:
            return EnsembleResult(
                triple=t,
                verdicts=verdicts,
                accepted_correction=best,
                status="auto_accepted",
                votes=dict(corrected_counts),
            )

    return EnsembleResult(
        triple=t,
        verdicts=verdicts,
        accepted_correction=None,
        status="manual_review",
        votes=dict(corrected_counts),
    )


def corrected_triple_to_rdf_triple(ct: tuple[str, str, str]) -> Triple:
    s, rel, o = ct
    return Triple(_iri_for_local(s), rel, _iri_for_local(o))
