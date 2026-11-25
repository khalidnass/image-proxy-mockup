"""Extra parameters: the few non-OpenAI parameters your image service takes.

You declare them in `app/your_image_api.py` (EXTRA_PARAMS). A caller sends them
one of two ways:

  1. in the body: a top-level JSON field on create, a form field on edit;
  2. in one header (EXTRA_PARAMS_HEADER, default `x-image-params`) holding a
     JSON object, at most 4 KB. If a name comes both ways, the header wins.

Everything fails closed (400) and no answer ever echoes a value: an unknown
name, a wrong type, a string longer than MAX_PROMPT_CHARS, a number that is not
finite, a header that is not a JSON object or is too big. Every string value
passes the same content rules as the prompt. The values go to your image code
only: never to the LLM, never to a log line.
"""
from __future__ import annotations

import json
import math
import re

from . import guardrails
from .config import ConfigError, Settings
from .errors import ProxyError

MAX_HEADER_BYTES = 4096
ALLOWED_TYPES = (int, float, bool, str)

# The OpenAI image fields a body may carry besides your declared ones. The
# proxy reads model, prompt, n, size and response_format and ignores the rest.
# (LiteLLM itself adds `user`, the caller's end-user id: it is one of them.)
STANDARD_CREATE = frozenset({
    "model", "prompt", "n", "size", "quality", "style", "response_format", "user", "background",
    "moderation", "output_compression", "output_format", "partial_images", "stream"})
STANDARD_EDIT = frozenset({
    "model", "prompt", "image", "image[]", "mask", "n", "size", "quality", "response_format", "user",
    "background", "input_fidelity", "output_compression", "output_format", "partial_images", "stream"})

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")


def check_declared(declared: dict) -> dict:
    """Refuse to start on a declaration the proxy cannot check."""
    for name, typ in declared.items():
        if not isinstance(name, str) or not _SAFE_NAME.match(name):
            raise ConfigError(f"EXTRA_PARAMS: {name!r} is not a plain parameter name")
        if typ not in ALLOWED_TYPES:
            raise ConfigError(f"EXTRA_PARAMS: {name} must be int, float, bool or str")
        if name in STANDARD_CREATE or name in STANDARD_EDIT:
            raise ConfigError(f"EXTRA_PARAMS: {name} is a standard OpenAI image field")
    return declared


def _named(name: str) -> str:
    return name if _SAFE_NAME.match(name) else "a parameter"


def _bad(name: str, why: str) -> ProxyError:
    # the name may be shown; the value never is
    return ProxyError(400, f"Extra parameter {_named(name)}: {why}.", code="invalid_extra_parameter",
                      param=_named(name))


def _coerce(name: str, value, typ, *, from_form: bool):
    if from_form and isinstance(value, str) and typ is not str:
        text = value.strip()
        try:
            if typ is bool:
                if text.lower() not in ("true", "false", "1", "0"):
                    raise ValueError
                value = text.lower() in ("true", "1")
            elif typ is int:
                value = int(text)
            else:
                value = float(text)
        except ValueError:
            raise _bad(name, f"must be {typ.__name__}")
    if typ is bool:
        ok = isinstance(value, bool)
    elif typ is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif typ is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value) if ok else value
    else:
        ok = isinstance(value, str)
    if not ok:
        raise _bad(name, f"must be {typ.__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise _bad(name, "must be a finite number")
    return value


def _check_string(name: str, value, settings: Settings) -> None:
    if not isinstance(value, str):
        return
    if len(value) > settings.max_prompt_chars:
        raise _bad(name, f"is longer than {settings.max_prompt_chars} characters")
    try:
        guardrails.check_prompt(value, max_chars=settings.max_prompt_chars,
                                terms_file=settings.blocked_terms_file)
    except ProxyError:
        raise ProxyError(400, f"Extra parameter {_named(name)} was refused by the content rules.",
                         code="content_policy_violation", param=_named(name))


def from_header(raw: str | None, declared: dict) -> dict:
    """The header's JSON object, names checked, values not yet typed."""
    if raw is None or raw.strip() == "":
        return {}
    if len(raw.encode("utf-8")) > MAX_HEADER_BYTES:
        raise ProxyError(400, f"The extra-parameters header is larger than {MAX_HEADER_BYTES} bytes.",
                         code="invalid_extra_parameter")
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = None
    if not isinstance(obj, dict):
        raise ProxyError(400, "The extra-parameters header must hold one JSON object.",
                         code="invalid_extra_parameter")
    for name in obj:
        if name not in declared:
            raise _bad(name, "is not a parameter this model takes")
    return obj


def collect(*, body: dict, header: str | None, kind: str, declared: dict, settings: Settings,
            from_form: bool) -> dict:
    """The checked extra parameters of one request: declared names only, typed, content-checked."""
    standard = STANDARD_EDIT if kind == "edit" else STANDARD_CREATE
    for name in body:
        if name not in standard and name not in declared:
            raise _bad(name, "is not a parameter this model takes")
    from_body = {k: body[k] for k in body if k in declared}
    head = from_header(header, declared)
    out = {}
    for name, value in from_body.items():
        out[name] = _coerce(name, value, declared[name], from_form=from_form)
    for name, value in head.items():  # the header wins
        out[name] = _coerce(name, value, declared[name], from_form=False)
    for name, value in out.items():
        _check_string(name, value, settings)
    return out
