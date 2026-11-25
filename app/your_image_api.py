"""The image call. THIS IS THE FILE YOU CHANGE.

The proxy has already checked the caller, verified the person and run the
guardrail and the enrichment. What reaches the two functions below is a
clean request. What they return is checked again (guardrail out) and sent
back to LiteLLM in the OpenAI shape.

THE CONTRACT (the same for both functions)

  Inputs   keyword arguments only:
           prompt     str          the prompt to draw (already enriched)
           model      str          the model name LiteLLM sent
           n          int          how many images (1 to MAX_IMAGES)
           size       str | None  for example "1024x1024", or None
           extra      dict         your EXTRA_PARAMS (below) the caller sent, already
                                   checked and typed; only declared names, maybe empty
           user       str          the verified user name (for your own audit,
                                   not a secret; the user's token is NOT given here)
           http       httpx.AsyncClient   a shared client; pass your own timeout
           settings   Settings     the proxy's settings (IMAGE_BACKEND_URL, ...)
           edit_image only:
           image      bytes        the input image as uploaded
           mask       bytes | None the mask, if one was sent

  Return   ImageOutput(images=[png_or_jpeg_or_webp_bytes, ...],
                       revised_prompt="the prompt you really used" or None)
           Return the raw image bytes, not base64. If your code changes the
           prompt (for example it enriches it itself), put the prompt it
           used in revised_prompt: it reaches the response's `revised_prompt`.

  Raise    ImageBackendError("short message")  when your backend fails.
           The caller gets a 502 with that message, so never put the prompt,
           a key or the backend's raw answer in it.
           Any other exception is also a 502 (with a generic message).
           The proxy's deadline (REQUEST_DEADLINE_SECONDS) covers your call.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx

from .config import Settings


@dataclass
class ImageOutput:
    images: list[bytes]
    revised_prompt: str | None = None


class ImageBackendError(Exception):
    """Your backend failed. The message is shown to the caller: keep it short and clean."""


# ============================================================================
# ===   YOUR EXTRA PARAMETERS: the non-OpenAI parameters your image      ===
# ===   service takes. Name -> type (int, float, bool or str).           ===
# ===   A caller sends them as body fields (JSON on create, form fields  ===
# ===   on edit) or in the x-image-params header as one JSON object;     ===
# ===   the header wins. They reach create_image/edit_image as `extra`,  ===
# ===   checked and typed. They never go to the LLM or to a log line.    ===
# ===   These three are an example: change them to yours.               ===
# ============================================================================
EXTRA_PARAMS: dict[str, type] = {
    "steps": int,
    "seed": int,
    "negative_prompt": str,
}


# ============================================================================
# ============================================================================
# ===                                                                      ===
# ===        PUT YOUR API CALL OR YOUR OWN CODE HERE  (create)             ===
# ===                                                                      ===
# ===  Way 1, your own endpoint: point IMAGE_BACKEND_URL at it and change   ===
# ===         the request body and the response mapping below.             ===
# ===  Way 2, your own code: replace the body of this function (a local    ===
# ===         model, an SDK, a queue...) and return ImageOutput.           ===
# ===  You may also enrich the prompt here yourself: then return the       ===
# ===  prompt you used as revised_prompt, and remove the "enrich" step     ===
# ===  from STEPS in app/pipeline.py.                                      ===
# ===                                                                      ===
# ============================================================================
# ============================================================================
async def create_image(*, prompt: str, model: str, n: int, size: str | None, extra: dict,
                       user: str, http: httpx.AsyncClient, settings: Settings) -> ImageOutput:
    # The shipped body calls the mock backend (mock_backend/), which speaks
    # the OpenAI image shape plus the three example extras. Add your
    # endpoint's own auth header here, and map `extra` to its own names.
    body = {"prompt": prompt, "n": n, "size": size or "256x256", **extra}
    r = await _post(http, f"{settings.image_backend_url}/v1/images/generations",
                    settings, json=body)
    return _openai_shape_to_output(r)


# ============================================================================
# ============================================================================
# ===                                                                      ===
# ===        PUT YOUR API CALL OR YOUR OWN CODE HERE  (edit)               ===
# ===                                                                      ===
# ===  Same two ways and the same contract as create_image above, plus     ===
# ===  `image` (bytes) and `mask` (bytes or None).                         ===
# ===                                                                      ===
# ============================================================================
# ============================================================================
async def edit_image(*, prompt: str, image: bytes, mask: bytes | None, model: str, n: int,
                     size: str | None, extra: dict, user: str, http: httpx.AsyncClient,
                     settings: Settings) -> ImageOutput:
    files = {"image": ("image.png", image, "image/png")}
    if mask is not None:
        files["mask"] = ("mask.png", mask, "image/png")
    data = {"prompt": prompt, "n": str(n), "size": size or "256x256",
            **{k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in extra.items()}}
    r = await _post(http, f"{settings.image_backend_url}/v1/images/edits",
                    settings, data=data, files=files)
    return _openai_shape_to_output(r)


# --- helpers for the shipped body; delete them if you replace it -------------

async def _post(http: httpx.AsyncClient, url: str, settings: Settings, **kwargs) -> httpx.Response:
    try:
        r = await http.post(url, timeout=settings.image_timeout_seconds, **kwargs)
    except httpx.TimeoutException:
        raise ImageBackendError("The image service did not answer in time.")
    except httpx.HTTPError:
        raise ImageBackendError("The image service could not be reached.")
    if r.status_code != 200:
        # The status only: the backend's own text may echo the prompt.
        raise ImageBackendError(f"The image service answered {r.status_code}.")
    return r


def _openai_shape_to_output(r: httpx.Response) -> ImageOutput:
    try:
        payload = r.json()
        images = [base64.b64decode(item["b64_json"], validate=True) for item in payload["data"]]
        revised = next((i.get("revised_prompt") for i in payload["data"] if i.get("revised_prompt")), None)
    except (ValueError, KeyError, TypeError):
        raise ImageBackendError("The image service sent an answer this proxy cannot read.")
    return ImageOutput(images=images, revised_prompt=revised)
