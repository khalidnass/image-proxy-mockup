"""The steps every image request runs, in order.

    guardrail_in  ->  enrich  ->  image (your_image_api.py)  ->  guardrail_out

Each step is a small async function that takes the job and the context.
It changes the job, or raises ProxyError to stop the request. To add a
step (a second check, a translation, a tool call, a review of the image by
a vision model...), write one more function like these and put it in STEPS
where it belongs. Each step is logged with its name, status and duration,
never with its data.
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx

from . import guardrails, your_image_api
from .auth import User
from .config import Settings
from .enrich import enrich_prompt
from .errors import ProxyError
from .logs import event


@dataclass
class Job:
    kind: str  # "create" or "edit"
    prompt: str
    model: str
    n: int = 1
    size: str | None = None
    image: bytes | None = None  # edit only
    mask: bytes | None = None   # edit only
    extra: dict = field(default_factory=dict)  # checked extra parameters: image code only
    body_user: object = None  # the body's `user`: used only with TRUST_BODY_USER
    # Filled in by the steps:
    revised_prompt: str | None = None
    enriched: bool = False
    images: list[bytes] = field(default_factory=list)


@dataclass
class Context:
    request_id: str
    user: User
    settings: Settings
    http: httpx.AsyncClient


Step = Callable[[Job, Context], Awaitable[None]]


async def guardrail_in(job: Job, ctx: Context) -> None:
    guardrails.check_prompt(job.prompt, max_chars=ctx.settings.max_prompt_chars,
                            terms_file=ctx.settings.blocked_terms_file)


async def enrich(job: Job, ctx: Context) -> None:
    better, ok = await enrich_prompt(job.prompt, user_id=ctx.user.name,
                                     settings=ctx.settings, http=ctx.http)
    if ok:
        try:  # the enriched text must pass the same rules as the original
            guardrails.check_prompt(better, max_chars=ctx.settings.max_prompt_chars,
                                    terms_file=ctx.settings.blocked_terms_file)
        except ProxyError:
            ok = False
    if ok:
        job.prompt, job.revised_prompt, job.enriched = better, better, True


async def image(job: Job, ctx: Context) -> None:
    # the image code gets the settings without the proxy's key and the LLM's key
    # (it keeps IMAGE_API_KEY: it is the one that calls the image service)
    safe = dataclasses.replace(ctx.settings, proxy_api_key="", llm_api_key="")
    common = dict(prompt=job.prompt, model=job.model, n=job.n, size=job.size, extra=dict(job.extra),
                  user=ctx.user.name, http=ctx.http, settings=safe)
    try:
        if job.kind == "edit":
            out = await your_image_api.edit_image(image=job.image, mask=job.mask, **common)
        else:
            out = await your_image_api.create_image(**common)
    except your_image_api.ImageBackendError as e:
        raise ProxyError(502, str(e) or "The image service failed.", type_="api_error",
                         code="image_backend_error")
    except (ProxyError, asyncio.CancelledError):
        raise
    except Exception:
        raise ProxyError(502, "The image service failed.", type_="api_error",
                         code="image_backend_error")
    job.images = list(out.images)
    if out.revised_prompt:
        job.revised_prompt = out.revised_prompt


async def guardrail_out(job: Job, ctx: Context) -> None:
    guardrails.check_images(job.images, max_bytes=ctx.settings.max_image_bytes)


STEPS: list[tuple[str, Step]] = [
    ("guardrail_in", guardrail_in),
    ("enrich", enrich),
    ("image", image),
    ("guardrail_out", guardrail_out),
]


async def run(job: Job, ctx: Context, steps: list[tuple[str, Step]] | None = None) -> Job:
    async def _all() -> None:
        for name, step in (steps if steps is not None else STEPS):
            started = time.perf_counter()
            status = "ok"
            try:
                await step(job, ctx)
                if name == "enrich" and not job.enriched:
                    status = "skipped"
            except ProxyError as e:
                status = f"refused_{e.status}"
                raise
            except BaseException:
                status = "error"
                raise
            finally:
                event("step", request_id=ctx.request_id, step=name, status=status,
                      ms=(time.perf_counter() - started) * 1000, user=ctx.user.name)

    try:
        await asyncio.wait_for(_all(), timeout=ctx.settings.request_deadline_seconds)
    except asyncio.TimeoutError:
        raise ProxyError(504, "The image request took too long.", type_="api_error",
                         code="deadline_exceeded")
    return job
