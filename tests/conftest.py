"""Test kit: a fake network (model server and image service), and an app factory."""
from __future__ import annotations

import base64

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from mock_backend.png import gradient

LLM = "http://model-server.test:8000/v1"
BACKEND = "http://mock-backend.test:8081"
PROXY_KEY = "proxy-key-for-tests-0123456789"
ID_HEADER = "x-litellm-customer-id"


def uid(user: str = "alice") -> str:
    """The user id a gateway writes into the id header (here, as a test sends it)."""
    return user


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
            out.append("llm" if u.startswith(LLM) else "backend" if u.startswith(BACKEND) else u)
        return out

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        url = str(request.url)
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
    base = dict(proxy_api_key=PROXY_KEY, llm_base_url=LLM, enrich_model="chat-model", image_backend_url=BACKEND,
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


def headers(user: str | None = None, key: str = PROXY_KEY) -> dict:
    """LiteLLM's call: the proxy key, and the user id header the gateway wrote."""
    h = {"Authorization": f"Bearer {key}"}
    if user is not None:
        h[ID_HEADER] = user
    return h


def gen(client, who=None, prompt="a red fox", key=PROXY_KEY, **body):
    """A create as LiteLLM sends it: `who` is the id header; `user=` in body is the body field."""
    return client.post("/v1/images/generations", headers=headers(who, key),
                       json={"model": "image-proxy-mockup", "prompt": prompt, **body})
