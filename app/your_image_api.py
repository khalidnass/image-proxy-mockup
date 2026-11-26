"""The image call. THIS IS THE FILE YOU CHANGE.

The proxy has already checked the caller, found the user id, and run the
guardrail and the enrichment. What reaches the two functions below is a
clean request. What they return is checked again (guardrail out) and sent
back to LiteLLM in the OpenAI shape.

As shipped, each function calls an image server in the OpenAI format (for
example vLLM serving Qwen-Image): create at IMAGE_CREATE_URL, edit at
IMAGE_EDIT_URL, each with its own model name and optional key.

THE CONTRACT (the same for both functions)

  Inputs   keyword arguments only:
           prompt     str          the prompt to draw (already enriched)
           model      str          the model name LiteLLM sent (the shipped body
                                   sends the configured IMAGE_*_MODEL instead)
           n          int          how many images (1 to MAX_IMAGES)
           size       str | None  for example "1024x1024", or None (server default)
           extra      dict         your EXTRA_PARAMS (below) the caller sent, already
                                   checked and typed; only declared names, maybe empty
           user       str          the user id the gateway sent: always passed on
           http       httpx.AsyncClient   a shared client; pass your own timeout
           settings   Settings     the proxy's settings, holding only the key of the
                                   image server this function calls
           edit_image only:
           image      bytes        the input image as uploaded (always present)
           mask       bytes | None the mask, if one was sent

  Return   ImageOutput(images=[png_or_jpeg_or_webp_bytes, ...],
                       revised_prompt="the prompt you really used" or None)
           Return the raw image bytes, not base64. If your code changes the
           prompt (for example it enriches it itself), put the prompt it
           used in revised_prompt: it reaches the response's `revised_prompt`.

  Raise    ImageBackendError("short message")  when your image server fails.
           The caller gets a 502 with that message, so never put the prompt,
           a key or the server's raw answer in it.
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
    """Your image server failed. The message is shown to the caller: keep it short and clean."""


# ============================================================================
# ===   YOUR EXTRA PARAMETERS: the inputs your image model takes beside  ===
# ===   the OpenAI fields. Name -> type (int, float, bool or str).       ===
# ===   A caller sends them as body fields (JSON on create, form fields  ===
# ===   on edit) or in the x-image-params header as one JSON object;     ===
# ===   the header wins. Any other name is refused (400). They reach     ===
# ===   create_image/edit_image as `extra`, checked and typed, and are   ===
# ===   passed to the image server as they are named here. They never   ===
# ===   go to the LLM or to a log line.                                  ===
# ===   As shipped: Qwen-Image's own inputs, as vLLM names them.         ===
# ============================================================================
EXTRA_PARAMS: dict[str, type] = {
    "negative_prompt": str,       # what to keep out; Qwen needs one for true_cfg_scale to act
    "true_cfg_scale": float,      # Qwen-Image's guidance strength
    "num_inference_steps": int,   # denoising steps
    "seed": int,                  # the same seed gives the same picture
}


# ============================================================================
# ============================================================================
# ===                                                                      ===
# ===        PUT YOUR API CALL OR YOUR OWN CODE HERE  (create)             ===
# ===                                                                      ===
# ===  Way 1, your own server: set IMAGE_CREATE_URL and IMAGE_CREATE_MODEL ===
# ===         (OpenAI format), or change the request below to fit yours.   ===
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
    body = {
        "model": settings.image_create_model,
        "prompt": prompt,
        "n": n,
        "response_format": "b64_json",
        "user": user,                 # the user name, always passed on
        **extra,                      # Qwen's inputs, as declared above
    }
    if size:
        body["size"] = size
    r = await _post(http, f"{settings.image_create_url}/v1/images/generations",
                    settings.image_create_api_key, settings, json=body)
    return _images_from(r)


# ============================================================================
# ============================================================================
# ===                                                                      ===
# ===        PUT YOUR API CALL OR YOUR OWN CODE HERE  (edit)               ===
# ===                                                                      ===
# ===  Same two ways and the same contract as create_image above, plus     ===
# ===  `image` (bytes) and `mask` (bytes or None). The server is           ===
# ===  IMAGE_EDIT_URL with IMAGE_EDIT_MODEL (OpenAI format: multipart).    ===
# ===                                                                      ===
# ============================================================================
# ============================================================================
async def edit_image(*, prompt: str, image: bytes, mask: bytes | None, model: str, n: int,
                     size: str | None, extra: dict, user: str, http: httpx.AsyncClient,
                     settings: Settings) -> ImageOutput:
    form = {
        "model": settings.image_edit_model,
        "prompt": prompt,
        "n": str(n),
        "response_format": "b64_json",
        "user": user,                 # the user name, always passed on
        **{k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in extra.items()},
    }
    if size:
        form["size"] = size
    files = {"image": ("image.png", image, "image/png")}
    if mask is not None:
        files["mask"] = ("mask.png", mask, "image/png")
    r = await _post(http, f"{settings.image_edit_url}/v1/images/edits",
                    settings.image_edit_api_key, settings, data=form, files=files)
    return _images_from(r)


# --- two small helpers for the shipped bodies -------------------------------

async def _post(http: httpx.AsyncClient, url: str, key: str, settings: Settings,
                **kwargs) -> httpx.Response:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        r = await http.post(url, headers=headers, timeout=settings.image_timeout_seconds, **kwargs)
    except httpx.TimeoutException:
        raise ImageBackendError("The image server did not answer in time.")
    except httpx.HTTPError:
        raise ImageBackendError("The image server could not be reached.")
    if r.status_code != 200:
        # The status only: the server's own text may echo the prompt.
        raise ImageBackendError(f"The image server answered {r.status_code}.")
    return r


def _images_from(r: httpx.Response) -> ImageOutput:
    """The OpenAI answer: data[].b64_json (and an optional revised_prompt)."""
    try:
        data = r.json()["data"]
        images = [base64.b64decode(item["b64_json"], validate=True) for item in data]
        revised = next((i.get("revised_prompt") for i in data if i.get("revised_prompt")), None)
    except (ValueError, KeyError, TypeError):
        raise ImageBackendError("The image server sent an answer this proxy cannot read.")
    return ImageOutput(images=images, revised_prompt=revised)
