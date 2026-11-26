"""A custom route of your own. COPY THIS FILE for each API of yours.

`POST /v1/customline` is an example of a route that is not an OpenAI call.
It gets the same protection as the image routes, before your code runs:

  - the caller must be LiteLLM: `Authorization: Bearer <PROXY_API_KEY>` (else 401);
  - the person must be named by the gateway's user id header (else 403; a
    malformed id, 400);
  - every text value in the JSON body passes the prompt's content rules (else 400);
  - the body must be one JSON object (else 400); errors are OpenAI-shaped.

The log line holds the request id, the route, the status, the time and the
user id: never a value from the body.

To add your own: copy this file, change ROUTE and the body of `handle`, and
add the router in app/main.py beside this one (`app.include_router(...)`).
Then give the path a way in at LiteLLM and at the gateway (see the README).
"""
from __future__ import annotations

import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from . import guardrails, logs
from .auth import check_caller, header_id, user_from
from .config import Settings
from .errors import ProxyError

ROUTE = "/v1/customline"
MAX_BODY_BYTES = 64 * 1024


def _texts(value):
    """Every string inside a JSON value, at any depth."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _texts(v)
    elif isinstance(value, list):
        for v in value:
            yield from _texts(v)


# ============================================================================
# ===              PUT YOUR OWN ROUTE HERE  (the shipped example)          ===
# ===  `body` is the checked JSON object, `user` the user id. Return a     ===
# ===  JSON-able dict, or raise ProxyError(status, "short message").       ===
# ===  The shipped body only echoes the user id and the inputs' names.     ===
# ============================================================================
async def handle(body: dict, user: str, request: Request, settings: Settings) -> dict:
    return {"object": "customline", "user": user, "inputs": sorted(body)}


def router(settings: Settings) -> APIRouter:
    r = APIRouter()

    @r.post(ROUTE)
    async def customline(request: Request):
        started = time.perf_counter()
        rid = request.headers.get("x-request-id", "") or "-"
        user_name = None
        try:
            check_caller(request, settings)
            user_name = header_id(request, settings)
            raw = await request.body()
            if len(raw) > MAX_BODY_BYTES:
                raise ProxyError(400, f"The body is larger than {MAX_BODY_BYTES} bytes.")
            try:
                body = json.loads(raw or b"{}")
            except ValueError:
                raise ProxyError(400, "The body must be JSON.")
            if not isinstance(body, dict):
                raise ProxyError(400, "The body must be a JSON object.")
            user = user_from(request, body.get("user"), settings)
            user_name = user.name
            for text in _texts(body):
                guardrails.check_prompt(text, max_chars=settings.max_prompt_chars,
                                        terms_file=settings.blocked_terms_file)
            answer = await handle(body, user.name, request, settings)
        except ProxyError as e:
            logs.event("customline", request_id=rid, status=f"refused_{e.status}", user=user_name,
                       ms=(time.perf_counter() - started) * 1000)
            return e.response()
        logs.event("customline", request_id=rid, status="ok", user=user_name,
                   ms=(time.perf_counter() - started) * 1000)
        return JSONResponse(answer)

    return r
