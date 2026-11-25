"""Enrichment: one chat call that makes the prompt more precise.

The call goes to LLM_BASE_URL (the gateway) with the calling person's own
JWT as Bearer, so the gateway checks the person again and LiteLLM records
the call on them. It is not a safety step, so it fails open: on any error
the original prompt is used and the response says so.
"""
from __future__ import annotations

import httpx

from .config import Settings

SYSTEM = ("You rewrite prompts for an image model. Make the prompt clear and concrete "
          "(subject, style, light, composition). Keep its meaning. Answer with the prompt only.")


async def enrich_prompt(prompt: str, *, user_token: str, settings: Settings,
                        http: httpx.AsyncClient) -> tuple[str, bool]:
    """Return (prompt to use, True if it was enriched)."""
    try:
        r = await http.post(
            f"{settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {user_token}"},
            json={"model": settings.enrich_model,
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": prompt}]},
            timeout=settings.enrich_timeout_seconds)
        if r.status_code != 200:
            return prompt, False
        text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
        return prompt, False
    if not text or len(text) > settings.max_prompt_chars:
        return prompt, False
    return text, True
