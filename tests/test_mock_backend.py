"""The mock backend draws real, different PNGs for create and edit."""
import base64

from fastapi.testclient import TestClient

from mock_backend.main import app
from mock_backend.png import gradient

c = TestClient(app)


def _png(r):
    assert r.status_code == 200
    data = base64.b64decode(r.json()["data"][0]["b64_json"])
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return data


def test_create_and_edit_give_different_valid_pngs():
    a = _png(c.post("/v1/images/generations", json={"prompt": "a fox", "size": "64x64"}))
    b = _png(c.post("/v1/images/edits", data={"prompt": "a fox", "size": "64x64"},
                    files={"image": ("i.png", gradient(b"i", 8), "image/png")}))
    assert a != b
    # same input, same picture; other input image, other picture
    assert a == _png(c.post("/v1/images/generations", json={"prompt": "a fox", "size": "64x64"}))
    b2 = _png(c.post("/v1/images/edits", data={"prompt": "a fox", "size": "64x64"},
                     files={"image": ("i.png", gradient(b"j", 8), "image/png")}))
    assert b != b2


def test_chat_is_a_deterministic_openai_answer():
    body = {"model": "mock-llm", "messages": [{"role": "system", "content": "rewrite"},
                                              {"role": "user", "content": "a red fox."}]}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    d = r.json()
    assert d["object"] == "chat.completion"
    assert d["choices"][0]["message"]["content"] == "a red fox, clear subject, soft light, simple composition"
    assert d["usage"]["total_tokens"] == d["usage"]["prompt_tokens"] + d["usage"]["completion_tokens"] > 0
    assert c.post("/v1/chat/completions", json=body).json()["choices"] == d["choices"]


def test_chat_refuses_empty_and_streaming():
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert c.post("/v1/chat/completions", json={"stream": True,
                  "messages": [{"role": "user", "content": "x"}]}).status_code == 400


def test_chat_logs_only_whether_the_header_came(monkeypatch, capsys):
    monkeypatch.setenv("MOCK_LOG_TOKEN_HEADER", "true")
    secret_text = "a secret lighthouse 4471"
    c.post("/v1/chat/completions", headers={"x-user-token": "token-value-must-not-show"},
           json={"messages": [{"role": "user", "content": secret_text}]})
    c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": secret_text}]})
    out = capsys.readouterr().out
    assert "present=yes" in out and "present=no" in out
    assert "token-value-must-not-show" not in out and "lighthouse" not in out


def test_chat_logs_nothing_by_default(monkeypatch, capsys):
    monkeypatch.delenv("MOCK_LOG_TOKEN_HEADER", raising=False)
    c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "a fox"}]})
    assert capsys.readouterr().out == ""


def test_the_proxy_enriches_with_the_mock_llm(monkeypatch):
    """End to end in one process: proxy -> mock chat (enrich) -> mock images."""
    import httpx
    from fastapi.testclient import TestClient as TC
    from app.main import create_app
    from conftest import JWKS, JWKS_URL, gen, make_token, settings

    async def route(request: httpx.Request) -> httpx.Response:
        if str(request.url) == JWKS_URL:
            return httpx.Response(200, json=JWKS)
        inner = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock")
        async with inner:
            return await inner.request(request.method, request.url.path, headers=request.headers,
                                       content=request.content)

    http = httpx.AsyncClient(transport=httpx.MockTransport(route))
    with TC(create_app(settings(llm_base_url="http://mock/v1", image_backend_url="http://mock"),
                       http=http)) as proxy:
        r = gen(proxy, make_token(), prompt="a red fox")
    assert r.status_code == 200 and r.headers["x-prompt-enriched"] == "true"
    assert r.json()["data"][0]["revised_prompt"] == "a red fox, clear subject, soft light, simple composition"
