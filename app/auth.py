"""Two checks on every image call.

1. The caller is LiteLLM: `Authorization: Bearer <PROXY_API_KEY>`.
2. The person: the user id the gateway wrote from the checked sign-in, in the
   header USER_ID_HEADER (default `x-litellm-customer-id`), which LiteLLM
   forwards to this model. No user id means no access: 403. A malformed id: 400.

The body's `user` field is not trusted: a caller can write it. Only with
TRUST_BODY_USER=true is it used, and only when the header is absent.

The proxy holds no person's token and calls no identity provider.
"""
from __future__ import annotations

import hmac
import re
from dataclasses import dataclass

from fastapi import Request

from .config import Settings
from .errors import ProxyError, unauthorized, user_refused

_USER_ID = re.compile(r"^[A-Za-z0-9._@-]{1,128}$")


@dataclass(frozen=True)
class User:
    name: str  # the user id: logged on every line, given to the image code


def check_caller(request: Request, settings: Settings) -> None:
    """The caller must be LiteLLM, holding PROXY_API_KEY. Constant-time compare."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
            value.strip().encode(), settings.proxy_api_key.encode()):
        raise unauthorized("This endpoint answers only the gateway's LiteLLM.", "invalid_proxy_key")


def user_from(request: Request, body_user, settings: Settings) -> User:
    """The person: the gateway's header first; the body's `user` only if trusted."""
    value = (request.headers.get(settings.user_id_header) or "").strip()
    if not value and settings.trust_body_user and isinstance(body_user, str):
        value = body_user.strip()
    if not value:
        raise user_refused("No user id: call this model through the gateway with your own sign-in.",
                           "missing_user_id")
    if not _USER_ID.match(value):
        raise ProxyError(400, "The user id is not a plain id.", code="invalid_user_id")
    return User(name=value)
