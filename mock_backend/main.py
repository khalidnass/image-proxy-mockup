"""The mock backend: a stand-in for your custom image endpoint, and a tiny
chat model so the enrichment step needs no real LLM.

It speaks the OpenAI shapes. Images are real PNGs drawn from a hash of the
inputs, so the same request always gives the same picture. The chat answer
is a fixed rewrite of the prompt. It never logs a prompt, an answer or an
image.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
import sys
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from .png import gradient, stripes

log = logging.getLogger("mock_backend")
if not log.handlers:
    class _Stdout(logging.StreamHandler):
        @property
        def stream(self):
            return sys.stdout

        @stream.setter
        def stream(self, _value):
            pass

    _h = _Stdout()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False

app = FastAPI(title="mock-image-backend", docs_url=None, redoc_url=None, openapi_url=None)
_SIZE = re.compile(r"^(\d{2,4})x(\d{2,4})$")
MAX_SIDE = 1024  # the size asked for, up to this on each side; above it, 400


class _TooBig(ValueError):
    pass


def _dims(size: str | None) -> tuple[int, int]:
    """The width and height asked for (256x256 when none or not WxH)."""
    m = _SIZE.match(size or "")
    if not m:
        return 256, 256
    w, h = int(m.group(1)), int(m.group(2))
    if not (1 <= w <= MAX_SIDE and 1 <= h <= MAX_SIDE):
        raise _TooBig
    return w, h


def _answer(images: list[bytes]) -> JSONResponse:
    return JSONResponse({"created": int(time.time()),
                         "data": [{"b64_json": base64.b64encode(i).decode()} for i in images]})


def _bad(msg: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": {"message": msg, "type": "invalid_request_error"}})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


REWRITE_SUFFIX = ", clear subject, soft light, simple composition"


@app.post("/v1/chat/completions")
async def chat(request: Request):
    """A tiny OpenAI chat model, so the whole mockup runs with no real LLM.

    It answers with a canned, deterministic rewrite of the last user message.
    Non-streaming only. It logs nothing about the text. With
    MOCK_LOG_TOKEN_HEADER=true it logs one line per call saying whether an
    `x-user-token` header arrived (yes/no, never the value): a check that
    the user's token does NOT reach models outside LiteLLM's forward list."""
    try:
        body = await request.json()
        messages = body.get("messages") or []
        last = next((m.get("content") for m in reversed(messages)
                     if isinstance(m, dict) and m.get("role") == "user"), None)
    except (ValueError, AttributeError):
        body, messages, last = {}, [], None
    if isinstance(last, list):  # content parts: keep the text parts
        last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
    if not isinstance(last, str) or not last.strip():
        return _bad("a user message is required")
    if body.get("stream"):
        return _bad("streaming is not supported by the mock")
    if os.environ.get("MOCK_LOG_TOKEN_HEADER", "").lower() == "true":
        log.info("mock chat: x-user-token header present=%s",
                 "yes" if request.headers.get("x-user-token") else "no")
    answer = last.strip().rstrip(".") + REWRITE_SUFFIX
    prompt_tokens = sum(len(str(m.get("content") or "").split()) for m in messages if isinstance(m, dict))
    completion_tokens = len(answer.split())
    return JSONResponse({
        "id": "chatcmpl-mock-" + hashlib.sha256(answer.encode()).hexdigest()[:12],
        "object": "chat.completion", "created": int(time.time()),
        "model": str(body.get("model") or "mock-llm"),
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": answer}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens}})


EXTRAS = ("steps", "seed", "negative_prompt")  # the example extra parameters


def _extras_seed(fields) -> str:
    """The example extras, folded into the picture's hash so a test can see they arrived."""
    return "|".join(f"{k}={fields.get(k)}" for k in EXTRAS if fields.get(k) is not None)


@app.post("/v1/images/generations")
async def generations(request: Request):
    body = await request.json()
    prompt = str(body.get("prompt") or "")
    if not prompt:
        return _bad("prompt is required")
    n = max(1, min(int(body.get("n") or 1), 4))
    try:
        w, h = _dims(body.get("size"))
    except _TooBig:
        return _bad(f"size must be at most {MAX_SIDE}x{MAX_SIDE}")
    extra = _extras_seed(body)
    return _answer([gradient(f"{prompt}|{i}|{extra}".encode(), w, h) for i in range(n)])


@app.post("/v1/images/edits")
async def edits(request: Request):
    form = await request.form()
    prompt = str(form.get("prompt") or "")
    image = form.get("image")
    if not prompt or not isinstance(image, UploadFile):
        return _bad("prompt and image are required")
    data = await image.read()
    n = max(1, min(int(form.get("n") or 1), 4))
    try:
        w, h = _dims(form.get("size"))
    except _TooBig:
        return _bad(f"size must be at most {MAX_SIDE}x{MAX_SIDE}")
    digest = hashlib.sha256(data).digest()
    extra = _extras_seed(form)
    return _answer([stripes(f"{prompt}|{i}|{extra}".encode() + digest, w, h) for i in range(n)])
