"""Who may call (LiteLLM, with the proxy key) and who the person is (the gateway's user id header)."""
import json

import pytest

from app import your_image_api
from app.config import ConfigError, from_env
from conftest import ID_HEADER, LLM, PROXY_KEY, gen, headers, settings, uid
from mock_backend.png import gradient


# --- the caller -------------------------------------------------------------------------------------

def test_missing_proxy_key_is_401_and_nothing_is_called(client, net):
    r = client.post("/v1/images/generations", json={"model": "image-proxy-mockup", "prompt": "x"},
                    headers={ID_HEADER: "alice"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_proxy_key"
    assert net.calls == []


def test_wrong_proxy_key_is_401(client, net):
    r = gen(client, uid(), key=PROXY_KEY + "x")
    assert r.status_code == 401 and net.calls == []


def test_models_needs_the_proxy_key(client):
    assert client.get("/v1/models").status_code == 401
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {PROXY_KEY}"})
    assert [m["id"] for m in r.json()["data"]] == ["image-proxy-mockup"]


# --- the person --------------------------------------------------------------------------------------

def test_the_header_id_is_accepted_logged_and_given_to_the_image_code(client, monkeypatch, capsys):
    got = {}

    async def create(**kw):
        got.update(kw)
        return your_image_api.ImageOutput(images=[gradient(b"x", 8)])

    monkeypatch.setattr(your_image_api, "create_image", create)
    capsys.readouterr()
    assert gen(client, uid("first.last-2@example.com")).status_code == 200
    assert got["user"] == "first.last-2@example.com"
    out = capsys.readouterr().out
    assert '"user": "first.last-2@example.com"' in out


@pytest.mark.parametrize("who", [None, "", "   "])
def test_no_user_id_is_403_and_no_llm_or_image_call(client, net, who):
    r = gen(client, who)
    assert r.status_code == 403
    err = r.json()["error"]
    assert err["code"] == "missing_user_id"
    assert err["message"] == "No user id: call this model through the gateway with your own sign-in."
    assert net.calls == []


def test_no_user_id_on_edit_is_403(client, net):
    r = client.post("/v1/images/edits", headers=headers(None),
                    data={"model": "image-proxy-mockup", "prompt": "x"},
                    files={"image": ("i.png", gradient(b"i", 8), "image/png")})
    assert r.status_code == 403 and net.calls == []


@pytest.mark.parametrize("bad", ["has space", "x" * 129, "a/b", "semi;colon", "quote\"d"])
def test_a_malformed_id_is_400_and_never_echoed(client, net, bad):
    r = gen(client, bad)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_user_id"
    assert bad not in r.text and net.calls == []


def test_the_body_user_is_ignored_by_default(client, net):
    r = gen(client, None, user="someone")
    assert r.status_code == 403 and net.calls == []


def test_a_body_user_never_wins_over_the_header(make_client, net):
    for c in (make_client(), make_client(trust_body_user=True)):
        net.calls.clear()
        assert gen(c, uid("alice"), user="mallory").status_code == 200
        llm = [x for x in net.calls if x.url.path.endswith("/chat/completions")][0]
        assert json.loads(llm.content)["user"] == "alice"
        backend = [x for x in net.calls if "mock-backend" in str(x.url)][0]
        assert json.loads(backend.content)["user"] == "alice"


def test_trust_body_user_uses_the_body_when_the_header_is_absent(make_client, net):
    c = make_client(trust_body_user=True)
    assert gen(c, None, user="bob").status_code == 200
    llm = [x for x in net.calls if x.url.path.endswith("/chat/completions")][0]
    assert json.loads(llm.content)["user"] == "bob"
    assert gen(c, None).status_code == 403
    assert gen(c, None, user="bad id").status_code == 400


def test_trust_body_user_on_edit(make_client, net):
    c = make_client(trust_body_user=True)
    r = c.post("/v1/images/edits", headers=headers(None),
               data={"model": "image-proxy-mockup", "prompt": "x", "user": "bob"},
               files={"image": ("i.png", gradient(b"i", 8), "image/png")})
    assert r.status_code == 200


def test_the_id_header_name_is_a_setting(make_client):
    c = make_client(user_id_header="x-user-id")
    r = c.post("/v1/images/generations",
               headers={"Authorization": f"Bearer {PROXY_KEY}", "x-user-id": "alice"},
               json={"model": "image-proxy-mockup", "prompt": "a fox"})
    assert r.status_code == 200
    assert gen(c, uid("alice")).status_code == 403  # the default header is not read then


# --- no token, no identity provider ------------------------------------------------------------------

def test_only_the_model_server_and_the_image_service_are_called(client, net):
    gen(client, uid())
    hosts = {x.url.host for x in net.calls}
    assert hosts == {"model-server.test", "mock-backend.test"}
    assert all(not x.url.path.endswith(("/certs", "/jwks", "/.well-known/openid-configuration"))
               for x in net.calls)


def test_a_user_token_header_is_neither_read_nor_forwarded(client, net):
    token = "a-user-token-that-must-not-travel"
    r = client.post("/v1/images/generations", headers={**headers(uid()), "x-user-token": token},
                    json={"model": "image-proxy-mockup", "prompt": "a fox"})
    assert r.status_code == 200
    for x in net.calls:
        assert token not in str(x.headers) and token.encode() not in x.content


def test_the_llm_call_carries_no_authorization_when_no_key(client, net):
    gen(client, uid())
    llm = [x for x in net.calls if str(x.url).startswith(LLM)][0]
    assert "authorization" not in llm.headers


# --- the settings ------------------------------------------------------------------------------------

def test_trust_body_user_reads_from_the_environment():
    env = {"PROXY_API_KEY": PROXY_KEY, "LLM_BASE_URL": "http://m/v1", "ENRICH_MODEL": "chat"}
    assert from_env(env).trust_body_user is False
    assert from_env({**env, "TRUST_BODY_USER": "true"}).trust_body_user is True
    with pytest.raises(ConfigError):
        from_env({**env, "TRUST_BODY_USER": "maybe"})


def test_no_oidc_setting_is_needed():
    env = {"PROXY_API_KEY": PROXY_KEY, "LLM_BASE_URL": "http://m/v1", "ENRICH_MODEL": "chat"}
    s = from_env(env)
    assert s.user_id_header == "x-litellm-customer-id" and not s.llm_api_key


def test_the_extras_header_must_differ_from_the_id_header():
    with pytest.raises(ConfigError):
        settings(extra_params_header="x-litellm-customer-id")


def test_keys_never_reach_an_answer_or_the_image_code(make_client, monkeypatch):
    got = {}

    async def create(**kw):
        got.update(kw)
        return your_image_api.ImageOutput(images=[gradient(b"x", 8)])

    monkeypatch.setattr(your_image_api, "create_image", create)
    c = make_client(llm_api_key="model-server-key-0123456789", image_api_key="image-key-0123456789")
    answers = [gen(c, uid()), gen(c, None), gen(c, "bad id"), gen(c, uid(), prompt="blocked-example")]
    passed = json.dumps({k: str(v) for k, v in got.items()})
    assert "model-server-key-0123456789" not in passed and PROXY_KEY not in passed
    assert "image-key-0123456789" in passed  # the image code needs its own service's key
    for r in answers:
        for k in ("model-server-key-0123456789", "image-key-0123456789", PROXY_KEY):
            assert k not in r.text


def test_a_refusal_while_reading_the_body_still_logs_the_user_id(client, capsys):
    capsys.readouterr()
    assert gen(client, uid("frank"), negative_prompt="a blocked-example").status_code == 400
    assert gen(client, uid("frank"), guidance=3).status_code == 400
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    refused = [x for x in lines if str(x.get("status", "")).startswith("refused")]
    assert len(refused) == 2 and all(x.get("user") == "frank" for x in refused)
