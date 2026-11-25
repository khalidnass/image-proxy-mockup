"""Settings, read once from the environment. A bad setting stops the start."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ALLOWED_ALGORITHMS = ("RS256", "ES256")


class ConfigError(RuntimeError):
    """A setting is missing or unsafe. The proxy refuses to start."""


def _list(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


@dataclass(frozen=True)
class Settings:
    # Who may call the proxy: LiteLLM, with this key as Bearer.
    proxy_api_key: str
    # Where the person's token arrives, and how it is checked.
    user_token_header: str = "x-user-token"
    # The header that may carry extra parameters as one JSON object.
    extra_params_header: str = "x-image-params"
    oidc_issuer: str = ""
    oidc_jwks_url: str = ""
    oidc_audiences: tuple[str, ...] = ()
    oidc_authorized_parties: tuple[str, ...] = ()
    # The enrichment call: the gateway's OpenAI-compatible base URL.
    llm_base_url: str = ""
    enrich_model: str = ""
    enrich_timeout_seconds: float = 20.0
    # The model names this proxy answers to (as LiteLLM sends them).
    served_models: tuple[str, ...] = ("image-proxy-mockup",)
    # The image backend the shipped code in your_image_api.py calls.
    image_backend_url: str = "http://mock-backend:8081"
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
        if not self.oidc_issuer:
            raise ConfigError("OIDC_ISSUER must be set")
        if not self.llm_base_url or not self.enrich_model:
            raise ConfigError("LLM_BASE_URL and ENRICH_MODEL must be set")
        if self.enrich_model in self.served_models:
            raise ConfigError(
                f"ENRICH_MODEL {self.enrich_model!r} is a model this proxy serves: "
                "the enrichment call would loop back here")
        if self.extra_params_header == self.user_token_header:
            raise ConfigError("EXTRA_PARAMS_HEADER must not be USER_TOKEN_HEADER")
        if not self.served_models:
            raise ConfigError("SERVED_MODELS must name at least one model")
        if not 0 < self.request_deadline_seconds < 300:
            raise ConfigError("REQUEST_DEADLINE_SECONDS must be above 0 and below 300")
        if self.enrich_timeout_seconds >= self.request_deadline_seconds:
            raise ConfigError("ENRICH_TIMEOUT_SECONDS must be below REQUEST_DEADLINE_SECONDS")
        return self


def from_env(env: dict | None = None) -> Settings:
    e = os.environ if env is None else env
    d = Settings.__dataclass_fields__

    def get(name: str, key: str):
        return e.get(name, d[key].default)

    return Settings(
        proxy_api_key=e.get("PROXY_API_KEY", ""),
        user_token_header=get("USER_TOKEN_HEADER", "user_token_header").lower(),
        extra_params_header=get("EXTRA_PARAMS_HEADER", "extra_params_header").lower(),
        oidc_issuer=get("OIDC_ISSUER", "oidc_issuer"),
        oidc_jwks_url=get("OIDC_JWKS_URL", "oidc_jwks_url"),
        oidc_audiences=_list(e.get("OIDC_AUDIENCES", "")),
        oidc_authorized_parties=_list(e.get("OIDC_AUTHORIZED_PARTIES", "")),
        llm_base_url=get("LLM_BASE_URL", "llm_base_url").rstrip("/"),
        enrich_model=get("ENRICH_MODEL", "enrich_model"),
        enrich_timeout_seconds=float(get("ENRICH_TIMEOUT_SECONDS", "enrich_timeout_seconds")),
        served_models=_list(e.get("SERVED_MODELS", "image-proxy-mockup")),
        image_backend_url=get("IMAGE_BACKEND_URL", "image_backend_url").rstrip("/"),
        image_timeout_seconds=float(get("IMAGE_TIMEOUT_SECONDS", "image_timeout_seconds")),
        request_deadline_seconds=float(get("REQUEST_DEADLINE_SECONDS", "request_deadline_seconds")),
        blocked_terms_file=get("BLOCKED_TERMS_FILE", "blocked_terms_file"),
        max_prompt_chars=int(get("MAX_PROMPT_CHARS", "max_prompt_chars")),
        max_image_bytes=int(get("MAX_IMAGE_BYTES", "max_image_bytes")),
        max_images=int(get("MAX_IMAGES", "max_images")),
    ).check()
