"""Guardrails: simple rules that fail closed.

In: the prompt is checked before anything else runs. A blocked term or an
over-long prompt is a 400, and no LLM or image call is made.
Out: every image the image code returned must be a real image and not too
big, or the request is a 502.

Replace these rules with a call to your own guardrail service if you have
one: keep the same contract (raise ProxyError to stop, return to go on).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .errors import ProxyError

_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"RIFF")  # PNG, JPEG, WebP (RIFF....WEBP)


@lru_cache(maxsize=4)
def blocked_terms(path: str) -> tuple[re.Pattern, ...]:
    p = Path(path)
    if not p.exists():
        return ()
    terms = [t.strip().lower() for t in p.read_text(encoding="utf-8").splitlines()]
    return tuple(re.compile(rf"\b{re.escape(t)}\b") for t in terms if t and not t.startswith("#"))


def check_prompt(prompt: str, *, max_chars: int, terms_file: str) -> None:
    if len(prompt) > max_chars:
        raise ProxyError(400, f"The prompt is longer than {max_chars} characters.",
                         code="prompt_too_long", param="prompt")
    low = prompt.lower()
    if any(t.search(low) for t in blocked_terms(terms_file)):
        # Never say which term matched: the answer must not echo the prompt.
        raise ProxyError(400, "The prompt was refused by the content rules.",
                         code="content_policy_violation", param="prompt")


def check_images(images: list[bytes], *, max_bytes: int) -> None:
    if not images:
        raise ProxyError(502, "The image service returned no image.", type_="api_error",
                         code="no_image")
    for img in images:
        if len(img) > max_bytes:
            raise ProxyError(502, "The image service returned an image that is too large.",
                             type_="api_error", code="image_too_large")
        if not img.startswith(_MAGIC) or (img.startswith(b"RIFF") and img[8:12] != b"WEBP"):
            raise ProxyError(502, "The image service returned something that is not an image.",
                             type_="api_error", code="not_an_image")
