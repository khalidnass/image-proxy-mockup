"""Two checks on every image call.

1. The caller is LiteLLM: `Authorization: Bearer <PROXY_API_KEY>`.
2. The person: their own JWT, which the gateway copied into the header named
   by USER_TOKEN_HEADER and LiteLLM forwarded. It is verified here against
   the issuer's JWKS: signature (RS256 or ES256 only), `iss`, `exp`, and the
   optional `aud` and `azp` allow-lists. No token, or a bad one: 403
   (not 401: see USER_TOKEN_REFUSAL_STATUS in errors.py).

The token is kept only for the life of the request (the enrichment call
sends it back to the gateway). It is never logged or stored.
"""
from __future__ import annotations

import asyncio
import hmac
import time
from dataclasses import dataclass, field

import httpx
import jwt
from fastapi import Request

from .config import ALLOWED_ALGORITHMS, Settings
from .errors import unauthorized, user_refused

JWKS_TTL_SECONDS = 300
JWKS_MIN_REFRESH_SECONDS = 30


@dataclass(frozen=True)
class User:
    name: str  # preferred_username, else sub: safe to log
    token: str = field(repr=False)  # the raw JWT: never log, never store


def check_caller(request: Request, settings: Settings) -> None:
    """The caller must be LiteLLM, holding PROXY_API_KEY. Constant-time compare."""
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
            value.strip().encode(), settings.proxy_api_key.encode()):
        raise unauthorized("This endpoint answers only the gateway's LiteLLM.", "invalid_proxy_key")


class TokenVerifier:
    """Verifies the person's JWT. The JWKS is fetched once and cached; an
    unknown `kid` triggers one refresh (at most every 30 seconds)."""

    def __init__(self, settings: Settings, http: httpx.AsyncClient):
        self.s = settings
        self.http = http
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched = 0.0
        self._lock = asyncio.Lock()

    async def _jwks_url(self) -> str:
        if self.s.oidc_jwks_url:
            return self.s.oidc_jwks_url
        r = await self.http.get(f"{self.s.oidc_issuer.rstrip('/')}/.well-known/openid-configuration",
                                timeout=10)
        r.raise_for_status()
        return r.json()["jwks_uri"]

    async def _refresh(self, force: bool) -> None:
        async with self._lock:
            age = time.monotonic() - self._fetched
            if self._keys and (age < JWKS_MIN_REFRESH_SECONDS or (not force and age < JWKS_TTL_SECONDS)):
                return
            r = await self.http.get(await self._jwks_url(), timeout=10)
            r.raise_for_status()
            keys = {}
            for k in r.json().get("keys", []):
                if k.get("use", "sig") != "sig" or not k.get("kid"):
                    continue
                try:
                    keys[k["kid"]] = jwt.PyJWK(k)
                except jwt.PyJWTError:
                    continue  # a key type this proxy does not take
            self._keys = keys
            self._fetched = time.monotonic()

    async def _key(self, kid: str) -> jwt.PyJWK:
        try:
            await self._refresh(force=False)
            if kid not in self._keys:
                await self._refresh(force=True)
        except (httpx.HTTPError, ValueError, KeyError):
            raise user_refused("The identity provider's keys could not be read.", "jwks_unavailable")
        if kid not in self._keys:
            raise user_refused("The user token is not valid.", "invalid_user_token")
        return self._keys[kid]

    async def verify(self, token: str) -> User:
        bad = user_refused("The user token is not valid.", "invalid_user_token")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise bad
        if header.get("alg") not in ALLOWED_ALGORITHMS:
            raise bad  # `none`, HS256 and anything else are refused before any key is looked at
        key = await self._key(str(header.get("kid", "")))
        try:
            claims = jwt.decode(
                token, key.key, algorithms=list(ALLOWED_ALGORITHMS),
                issuer=self.s.oidc_issuer,
                audience=list(self.s.oidc_audiences) or None,
                options={"require": ["exp", "iss"], "verify_aud": bool(self.s.oidc_audiences)})
        except jwt.ExpiredSignatureError:
            raise user_refused("The user token has expired.", "expired_user_token")
        except jwt.PyJWTError:
            raise bad
        if self.s.oidc_authorized_parties and claims.get("azp") not in self.s.oidc_authorized_parties:
            raise bad
        name = claims.get("preferred_username") or claims.get("sub")
        if not name:
            raise bad
        return User(name=str(name), token=token)

    async def user_from(self, request: Request) -> User:
        token = request.headers.get(self.s.user_token_header, "").strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token:
            raise user_refused(
                f"No user token: the {self.s.user_token_header} header is missing. "
                "Call this model through the gateway with your own sign-in.", "missing_user_token")
        return await self.verify(token)
