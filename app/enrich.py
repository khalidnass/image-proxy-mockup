"""Enrichment: one chat call that makes the prompt more precise.

The call goes straight to the model server (LLM_BASE_URL, an OpenAI-compatible
server such as vLLM), with the server's own key when LLM_API_KEY is set and no
Authorization header when it is blank. It names the person in the body field
`user`, so the model server's own log can show who asked. It carries no
person's token. It is not a safety step, so it fails open: on any error the
original prompt is used and the response says so.
"""
from __future__ import annotations

import httpx

from .config import Settings

SYSTEM = ("You rewrite prompts for an image model. Make the prompt clear and concrete "
          "(subject, style, light, composition). Keep its meaning. Answer with the prompt only.")


async def enrich_prompt(prompt: str, *, user_id: str, settings: Settings,
                        http: httpx.AsyncClient) -> tuple[str, bool]:
    """Return (prompt to use, True if it was enriched)."""
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"} if settings.llm_api_key else {}
    try:
        r = await http.post(
            f"{settings.llm_base_url}/chat/completions", headers=headers,
            json={"model": settings.enrich_model, "user": user_id,
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
