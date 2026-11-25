"""The image proxy: OpenAI image calls in, the pipeline, OpenAI image answers out.

    POST /v1/images/generations   JSON
    POST /v1/images/edits         multipart/form-data
    GET  /v1/models               the names this proxy serves
    GET  /healthz                 no auth
"""
from __future__ import annotations

import base64
import re
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from . import logs, pipeline
from .auth import TokenVerifier, check_caller
from .config import Settings, from_env
from .errors import ProxyError

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_SIZE = re.compile(r"^\d{2,4}x\d{2,4}$")


def _request_id(request: Request) -> str:
    rid = request.headers.get("x-request-id", "")
    return rid if _SAFE_ID.match(rid) else uuid.uuid4().hex


def create_app(settings: Settings | None = None, http: httpx.AsyncClient | None = None) -> FastAPI:
    settings = settings or from_env()
    logs.setup()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        own = http is None
        app.state.http = http or httpx.AsyncClient()
        app.state.verifier = TokenVerifier(settings, app.state.http)
        logs.event("started", request_id="-", status="ok")
        yield
        if own:
            await app.state.http.aclose()

    app = FastAPI(title="image-proxy-mockup", lifespan=lifespan, docs_url=None,
                  redoc_url=None, openapi_url=None)

    @app.exception_handler(ProxyError)
    async def _proxy_error(request: Request, exc: ProxyError):
        return exc.response(getattr(request.state, "request_id", ""))

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError):
        # FastAPI's default answer echoes the input (the prompt): not here.
        return ProxyError(400, "The request body is not valid.").response()

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request):
        check_caller(request, settings)
        return {"object": "list", "data": [
            {"id": m, "object": "model", "created": 0, "owned_by": "image-proxy-mockup"}
            for m in settings.served_models]}

    async def _handle(request: Request, kind: str, read_job) -> JSONResponse:
        rid = request.state.request_id = _request_id(request)
        started = time.perf_counter()
        user_name = None
        try:
            check_caller(request, settings)
            user = await request.app.state.verifier.user_from(request)
            user_name = user.name
            job = await read_job(request)
            if job.model not in settings.served_models:
                raise ProxyError(400, "This proxy does not serve that model.",
                                 code="model_not_found", param="model")
            ctx = pipeline.Context(request_id=rid, user=user, settings=settings,
                                   http=request.app.state.http)
            await pipeline.run(job, ctx)
        except ProxyError as e:
            logs.event(kind, request_id=rid, status=f"refused_{e.status}", user=user_name,
                       ms=(time.perf_counter() - started) * 1000)
            raise
        logs.event(kind, request_id=rid, status="ok", user=user_name,
                   ms=(time.perf_counter() - started) * 1000)
        revised = job.revised_prompt or job.prompt
        return JSONResponse(
            headers={"x-request-id": rid, "x-prompt-enriched": "true" if job.enriched else "false"},
            content={"created": int(time.time()), "data": [
                {"b64_json": base64.b64encode(img).decode(), "revised_prompt": revised}
                for img in job.images]})

    def _common(prompt, model, n, size, response_format) -> tuple[str, str, int, str | None]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProxyError(400, "A prompt is required.", param="prompt")
        if not isinstance(model, str) or not model:
            raise ProxyError(400, "A model is required.", param="model")
        try:
            n = int(n if n is not None else 1)
        except (TypeError, ValueError):
            raise ProxyError(400, "n must be a whole number.", param="n")
        if not 1 <= n <= settings.max_images:
            raise ProxyError(400, f"n must be between 1 and {settings.max_images}.", param="n")
        if size is not None and (not isinstance(size, str) or not _SIZE.match(size)):
            raise ProxyError(400, "size must look like 1024x1024.", param="size")
        if response_format not in (None, "b64_json"):
            raise ProxyError(400, "Only response_format b64_json is supported.",
                             param="response_format")
        return prompt, model, n, size

    async def _read_generation(request: Request) -> pipeline.Job:
        try:
            body = await request.json()
        except ValueError:
            raise ProxyError(400, "The body must be JSON.")
        if not isinstance(body, dict):
            raise ProxyError(400, "The body must be a JSON object.")
        prompt, model, n, size = _common(body.get("prompt"), body.get("model"), body.get("n"),
                                         body.get("size"), body.get("response_format"))
        return pipeline.Job(kind="create", prompt=prompt, model=model, n=n, size=size)

    async def _read_file(form, *names: str) -> bytes | None:
        for name in names:
            item = form.get(name)
            if isinstance(item, UploadFile):
                data = await item.read(settings.max_image_bytes + 1)
                if len(data) > settings.max_image_bytes:
                    raise ProxyError(413, "The uploaded image is too large.", param=name)
                return data
        return None

    async def _read_edit(request: Request) -> pipeline.Job:
        if not request.headers.get("content-type", "").startswith("multipart/form-data"):
            raise ProxyError(400, "Image edits must be sent as multipart/form-data.")
        try:
            form = await request.form(max_files=4, max_fields=20)
        except Exception:
            raise ProxyError(400, "The multipart body could not be read.")
        prompt, model, n, size = _common(form.get("prompt"), form.get("model"), form.get("n"),
                                         form.get("size"), form.get("response_format"))
        image = await _read_file(form, "image", "image[]")
        if not image:
            raise ProxyError(400, "An image file is required.", param="image")
        mask = await _read_file(form, "mask")
        return pipeline.Job(kind="edit", prompt=prompt, model=model, n=n, size=size,
                            image=image, mask=mask)

    @app.post("/v1/images/generations")
    async def generations(request: Request):
        return await _handle(request, "create", _read_generation)

    @app.post("/v1/images/edits")
    async def edits(request: Request):
        return await _handle(request, "edit", _read_edit)

    return app


def app() -> FastAPI:  # uvicorn --factory app.main:app
    return create_app()
