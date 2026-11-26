"""The example custom route, /v1/customline: the same protection as the image routes."""
import json

import pytest

from conftest import PROXY_KEY, headers


def call(client, body=None, who="alice", key=PROXY_KEY, raw=None):
    return client.post("/v1/customline", headers={**headers(who, key), "Content-Type": "application/json"},
                       content=raw if raw is not None else json.dumps(body if body is not None else {}))


def test_it_answers_with_the_user_and_the_input_names(client, net):
    r = call(client, {"text": "a lighthouse", "level": 3, "tags": ["a", "b"]})
    assert r.status_code == 200
    assert r.json() == {"object": "customline", "user": "alice", "inputs": ["level", "tags", "text"]}
    assert net.calls == []  # the shipped example calls nothing


def test_the_caller_must_be_litellm(client):
    r = call(client, {"text": "x"}, key=PROXY_KEY + "x")
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_proxy_key"


def test_no_user_id_is_403(client):
    r = call(client, {"text": "x"}, who=None)
    assert r.status_code == 403 and r.json()["error"]["code"] == "missing_user_id"


def test_a_body_user_is_not_trusted(client):
    assert call(client, {"text": "x", "user": "mallory"}, who=None).status_code == 403
    r = call(client, {"text": "x", "user": "mallory"}, who="alice")
    assert r.status_code == 200 and r.json()["user"] == "alice"


def test_a_malformed_id_is_400(client):
    assert call(client, {"text": "x"}, who="bad id").status_code == 400


@pytest.mark.parametrize("body", [{"text": "a blocked-example"}, {"deep": {"list": ["ok", "blocked-example"]}}])
def test_every_text_value_passes_the_content_rules(client, body):
    r = call(client, body)
    assert r.status_code == 400 and r.json()["error"]["code"] == "content_policy_violation"
    assert "blocked-example" not in r.text


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b'"text"'])
def test_the_body_must_be_one_json_object(client, raw):
    assert call(client, raw=raw).status_code == 400


def test_an_oversize_body_is_400(client):
    assert call(client, raw=json.dumps({"text": "x" * 70000}).encode()).status_code == 400


def test_the_log_holds_the_user_and_no_value(client, capsys):
    capsys.readouterr()
    call(client, {"text": "secret-value-5521", "n": 990011})
    call(client, {"text": "blocked-example secret-value-5521"})
    out = capsys.readouterr().out
    lines = [json.loads(x) for x in out.splitlines() if x.startswith("{")]
    custom = [x for x in lines if x.get("event") == "customline"]
    assert [x["status"] for x in custom] == ["ok", "refused_400"]
    assert all(x["user"] == "alice" for x in custom)
    assert "secret-value-5521" not in out and "990011" not in out
