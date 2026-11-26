"""Settings, read once from the environment. A bad setting stops the start."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    """A setting is missing or unsafe. The proxy refuses to start."""


def _list(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def _bool(name: str, value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    raise ConfigError(f"{name} must be true or false")


_HEADER = re.compile(r"^[a-z0-9-]{1,64}$")


@dataclass(frozen=True)
class Settings:
    # Who may call the proxy: LiteLLM, with this key as Bearer.
    proxy_api_key: str
    # Who the person is: the header the gateway writes from the checked sign-in
    # (and LiteLLM forwards to this model). No id: 403.
    user_id_header: str = "x-litellm-customer-id"
    # Use the body's `user` when the header is absent. A caller can write the
    # body's `user` itself, so this can be faked: keep it false unless the
    # caller is trusted.
    trust_body_user: bool = False
    # The header that may carry extra parameters as one JSON object.
    extra_params_header: str = "x-image-params"
    # The enrichment call: straight to the model server (an OpenAI-compatible
    # server such as vLLM), with its own key if it has one (blank: no header).
    llm_base_url: str = ""
    llm_api_key: str = ""
    enrich_model: str = ""
    enrich_timeout_seconds: float = 20.0
    # The model names this proxy answers to (as LiteLLM sends them).
    served_models: tuple[str, ...] = ("image-proxy-mockup",)
    # The image service the shipped code in your_image_api.py calls, and its key.
    image_backend_url: str = "http://mock-backend:8081"
    image_api_key: str = ""
    image_timeout_seconds: float = 180.0
    # Whole-request deadline. Keep it below LiteLLM's own timeout (300 s).
    request_deadline_seconds: float = 240.0
    # Guardrails.
    blocked_terms_file: str = str(Path(__file__).with_name("blocked_terms.txt"))
    max_prompt_chars: int = 4000
    max_image_bytes: int = 20 * 1024 * 1024
    max_images: int = 4

    def check(self) -> "Settings":
        if len(self.proxy_api_key) < 16:
            raise ConfigError("PROXY_API_KEY must be set (16 characters or more)")
        if not self.llm_base_url or not self.enrich_model:
            raise ConfigError("LLM_BASE_URL and ENRICH_MODEL must be set")
        if self.enrich_model in self.served_models:
            raise ConfigError(
                f"ENRICH_MODEL {self.enrich_model!r} is a model this proxy serves: "
                "the enrichment call would loop back here")
        if not _HEADER.match(self.user_id_header) or not _HEADER.match(self.extra_params_header):
            raise ConfigError("USER_ID_HEADER and EXTRA_PARAMS_HEADER must be plain header names")
        if self.extra_params_header == self.user_id_header:
            raise ConfigError("EXTRA_PARAMS_HEADER must not be USER_ID_HEADER")
        if not self.served_models:
            raise ConfigError("SERVED_MODELS must name at least one model")
        if not 0 < self.request_deadline_seconds < 300:
            raise ConfigError("REQUEST_DEADLINE_SECONDS must be above 0 and below 300")
        if self.enrich_timeout_seconds >= self.request_deadline_seconds:
            raise ConfigError("ENRICH_TIMEOUT_SECONDS must be below REQUEST_DEADLINE_SECONDS")
        return self

    def secrets(self) -> tuple[str, ...]:
        """Every key this proxy holds: never in a log line or an answer."""
        return tuple(s for s in (self.proxy_api_key, self.llm_api_key, self.image_api_key) if s)


def from_env(env: dict | None = None) -> Settings:
    e = os.environ if env is None else env
    d = Settings.__dataclass_fields__

    def get(name: str, key: str):
        return e.get(name, d[key].default)

    return Settings(
        proxy_api_key=e.get("PROXY_API_KEY", ""),
        user_id_header=get("USER_ID_HEADER", "user_id_header").lower(),
        trust_body_user=_bool("TRUST_BODY_USER", get("TRUST_BODY_USER", "trust_body_user")),
        extra_params_header=get("EXTRA_PARAMS_HEADER", "extra_params_header").lower(),
        llm_base_url=get("LLM_BASE_URL", "llm_base_url").rstrip("/"),
        llm_api_key=e.get("LLM_API_KEY", ""),
        enrich_model=get("ENRICH_MODEL", "enrich_model"),
        enrich_timeout_seconds=float(get("ENRICH_TIMEOUT_SECONDS", "enrich_timeout_seconds")),
        served_models=_list(e.get("SERVED_MODELS", "image-proxy-mockup")),
        image_backend_url=get("IMAGE_BACKEND_URL", "image_backend_url").rstrip("/"),
        image_api_key=e.get("IMAGE_API_KEY", ""),
        image_timeout_seconds=float(get("IMAGE_TIMEOUT_SECONDS", "image_timeout_seconds")),
        request_deadline_seconds=float(get("REQUEST_DEADLINE_SECONDS", "request_deadline_seconds")),
        blocked_terms_file=get("BLOCKED_TERMS_FILE", "blocked_terms_file"),
        max_prompt_chars=int(get("MAX_PROMPT_CHARS", "max_prompt_chars")),
        max_image_bytes=int(get("MAX_IMAGE_BYTES", "max_image_bytes")),
        max_images=int(get("MAX_IMAGES", "max_images")),
    ).check()
