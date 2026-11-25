"""The two checks: the caller is LiteLLM, and the person's token verifies."""
import time

import pytest

from conftest import EC_KEY, OTHER_KEY, PROXY_KEY, gen, make_token, unsigned_token


def test_missing_proxy_key_is_401_and_nothing_is_called(client, net):
    r = client.post("/v1/images/generations", json={"model": "image-proxy-mockup", "prompt": "x"},
                    headers={"x-user-token": make_token()})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_proxy_key"
    assert net.calls == []


def test_wrong_proxy_key_is_401(client, net):
    r = gen(client, make_token(), key=PROXY_KEY + "x")
    assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"
    assert net.calls == []


def test_models_needs_the_proxy_key(client):
    assert client.get("/v1/models").status_code == 401
    r = client.get("/v1/models", headers={"Authorization": f"Bearer {PROXY_KEY}"})
    assert [m["id"] for m in r.json()["data"]] == ["image-proxy-mockup"]


def test_good_token_passes(client, net):
    r = gen(client, make_token("alice"))
    assert r.status_code == 200
    assert net.names() == ["jwks", "llm", "backend"]


def test_es256_token_passes(client):
    assert gen(client, make_token(key=EC_KEY, kid="ec1", alg="ES256")).status_code == 200


def test_bearer_prefix_in_the_header_is_accepted(client):
    assert gen(client, "Bearer " + make_token()).status_code == 200


@pytest.mark.parametrize("name, token, code", [
    ("missing", None, "missing_user_token"),
    ("empty", "", "missing_user_token"),
    ("expired", make_token(exp=int(time.time()) - 60), "expired_user_token"),
    ("wrong issuer", make_token(iss="https://evil.example.com"), "invalid_user_token"),
    ("alg none", unsigned_token(), "invalid_user_token"),
    ("tampered", None, "invalid_user_token"),
    ("signed by another key", make_token(key=OTHER_KEY), "invalid_user_token"),
    ("unknown kid", make_token(kid="nope"), "invalid_user_token"),
    ("no exp", make_token(exp=None), "invalid_user_token"),
    ("garbage", "not-a-jwt", "invalid_user_token"),
    ("hs256", None, "invalid_user_token"),
])
def test_bad_tokens_are_refused_and_no_llm_or_image_call(client, net, name, token, code):
    if name == "tampered":
        head, body, sig = make_token("alice").split(".")
        other = make_token("mallory").split(".")[1]
        token = ".".join([head, other, sig])
    if name == "hs256":
        import jwt
        token = jwt.encode({"iss": "https://id.example.com/realms/demo", "exp": int(time.time()) + 60,
                            "preferred_username": "x"}, "secret" * 8, algorithm="HS256",
                           headers={"kid": "rsa1"})
    r = gen(client, token)
    assert r.status_code == 403, name  # 403, not 401: LiteLLM cools a model down on 401
    assert r.json()["error"]["code"] == code
    assert "llm" not in net.names() and "backend" not in net.names()


def test_audience_allow_list(make_client, net):
    c = make_client(oidc_audiences=("image-app",))
    assert gen(c, make_token(aud="image-app")).status_code == 200
    r = gen(c, make_token(aud="another-app"))
    assert r.status_code == 403 and r.json()["error"]["code"] == "invalid_user_token"


def test_azp_allow_list(make_client):
    c = make_client(oidc_authorized_parties=("image-app",))
    assert gen(c, make_token(azp="image-app")).status_code == 200
    assert gen(c, make_token(azp="other-client")).status_code == 403


def test_header_name_is_a_setting(make_client):
    c = make_client(user_token_header="x-person-jwt")
    r = c.post("/v1/images/generations",
               headers={"Authorization": f"Bearer {PROXY_KEY}", "x-person-jwt": make_token()},
               json={"model": "image-proxy-mockup", "prompt": "a fox"})
    assert r.status_code == 200


def test_jwks_is_cached(client, net):
    for _ in range(3):
        assert gen(client, make_token()).status_code == 200
    assert net.names().count("jwks") == 1
