"""Test kit: a signing key and JWKS, a fake network, and an app factory."""
from __future__ import annotations

import base64
import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from mock_backend.png import gradient

ISSUER = "https://id.example.com/realms/demo"
JWKS_URL = "https://id.example.com/jwks"
LLM = "https://llm.example.com/v1"
BACKEND = "http://mock-backend.test:8081"
PROXY_KEY = "proxy-key-for-tests-0123456789"

RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
EC_KEY = ec.generate_private_key(ec.SECP256R1())
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwk(private, kid, alg):
    pub = private.public_key()
    d = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(pub) if alg == "RS256"
                   else jwt.algorithms.ECAlgorithm.to_jwk(pub))
    d.update(kid=kid, alg=alg, use="sig")
    return d


JWKS = {"keys": [_jwk(RSA_KEY, "rsa1", "RS256"), _jwk(EC_KEY, "ec1", "ES256")]}


def make_token(user="alice", *, key=RSA_KEY, kid="rsa1", alg="RS256", **over) -> str:
    now = int(time.time())
    claims = {"iss": ISSUER, "sub": f"id-{user}", "preferred_username": user, "aud": "image-app",
              "azp": "image-app", "iat": now, "exp": now + 300}
    claims.update(over)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, key, algorithm=alg, headers={"kid": kid})


def unsigned_token(user="alice") -> str:
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    now = int(time.time())
    return f"{b64({'alg': 'none', 'kid': 'rsa1'})}.{b64({'iss': ISSUER, 'preferred_username': user, 'exp': now + 300})}."


class FakeNet:
    """Answers the proxy's outbound calls and records them."""

    def __init__(self):
        self.calls: list[httpx.Request] = []
        self.llm_status = 200
        self.llm_text = "A precise, well-lit picture"
        self.llm_raises = False
        self.backend_status = 200
        self.backend_body = None  # override the backend's JSON answer

    def names(self) -> list[str]:
        out = []
        for r in self.calls:
            u = str(r.url)
            out.append("jwks" if u == JWKS_URL else "llm" if u.startswith(LLM) else
                       "backend" if u.startswith(BACKEND) else u)
        return out

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        url = str(request.url)
        if url == JWKS_URL:
            return httpx.Response(200, json=JWKS)
        if url.startswith(LLM):
            if self.llm_raises:
                raise httpx.ConnectError("down", request=request)
            return httpx.Response(self.llm_status, json={
                "choices": [{"message": {"role": "assistant", "content": self.llm_text}}]})
        if url.startswith(BACKEND):
            if self.backend_body is not None:
                return httpx.Response(self.backend_status, json=self.backend_body)
            return httpx.Response(self.backend_status, json={"created": 1, "data": [
                {"b64_json": base64.b64encode(gradient(b"x", 16)).decode()}]})
        return httpx.Response(404)


def settings(**over) -> Settings:
    base = dict(proxy_api_key=PROXY_KEY, oidc_issuer=ISSUER, oidc_jwks_url=JWKS_URL,
                llm_base_url=LLM, enrich_model="chat-model", image_backend_url=BACKEND,
                served_models=("image-proxy-mockup",))
    base.update(over)
    return Settings(**base).check()


@pytest.fixture
def net():
    return FakeNet()


@pytest.fixture
def make_client(net):
    clients = []

    def _make(**over):
        http = httpx.AsyncClient(transport=httpx.MockTransport(net.handler))
        c = TestClient(create_app(settings(**over), http=http))
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()


def headers(token: str | None = None, key: str = PROXY_KEY) -> dict:
    h = {"Authorization": f"Bearer {key}"}
    if token is not None:
        h["x-user-token"] = token
    return h


def gen(client, token=None, prompt="a red fox", key=PROXY_KEY, **body):
    return client.post("/v1/images/generations", headers=headers(token, key),
                       json={"model": "image-proxy-mockup", "prompt": prompt, **body})
