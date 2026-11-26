"""The steps, their order, the enrichment call and the failure answers."""
import asyncio
import base64
import json

import pytest

from app import pipeline, your_image_api
from app.config import ConfigError
from conftest import PROXY_KEY, gen, headers, uid, settings
from mock_backend.png import gradient


def test_steps_run_in_order(client, net, capsys):
    capsys.readouterr()
    assert gen(client, uid()).status_code == 200
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    steps = [line["step"] for line in lines if line.get("event") == "step"]
    assert steps == ["guardrail_in", "enrich", "image", "guardrail_out"]
    assert [s for s, _ in pipeline.STEPS] == steps


def test_response_is_openai_shaped_with_a_png(client):
    r = gen(client, uid())
    body = r.json()
    assert isinstance(body["created"], int)
    png = base64.b64decode(body["data"][0]["b64_json"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert body["data"][0]["revised_prompt"] == "A precise, well-lit picture"
    assert r.headers["x-prompt-enriched"] == "true"


def test_enrichment_goes_to_the_model_server_with_the_user_id_and_no_auth_by_default(client, net):
    gen(client, uid("alice"))
    llm = [c for c in net.calls if str(c.url).endswith("/chat/completions")]
    assert len(llm) == 1 and str(llm[0].url).startswith("http://model-server.test:8000/v1/")
    assert "authorization" not in llm[0].headers  # LLM_API_KEY blank: no header
    body = json.loads(llm[0].content)
    assert body["model"] == "chat-model" and body["user"] == "alice"


def test_enrichment_carries_the_model_servers_key_when_set(make_client, net):
    c = make_client(llm_api_key="model-server-key-0123456789")
    gen(c, uid("alice"))
    llm = [x for x in net.calls if str(x.url).endswith("/chat/completions")][0]
    assert llm.headers["authorization"] == "Bearer model-server-key-0123456789"


def test_the_image_service_gets_the_user_id_and_its_own_key_only(make_client, net):
    c = make_client(llm_api_key="model-server-key-0123456789", image_create_api_key="image-key-0123456789")
    gen(c, uid("alice"))
    backend = [x for x in net.calls if "mock-backend" in str(x.url)][0]
    assert backend.headers["authorization"] == "Bearer image-key-0123456789"
    assert json.loads(backend.content)["user"] == "alice"
    for other in ("model-server-key-0123456789", PROXY_KEY):
        assert other not in str(backend.headers) and other.encode() not in backend.content


@pytest.mark.parametrize("how", ["status", "raises", "empty"])
def test_enrichment_failure_falls_back_to_the_prompt(client, net, how):
    if how == "status":
        net.llm_status = 500
    elif how == "raises":
        net.llm_raises = True
    else:
        net.llm_text = ""
    r = gen(client, uid(), prompt="a red fox")
    assert r.status_code == 200
    assert r.headers["x-prompt-enriched"] == "false"
    assert r.json()["data"][0]["revised_prompt"] == "a red fox"
    sent = [c for c in net.calls if "mock-backend" in str(c.url)][0]
    assert json.loads(sent.content)["prompt"] == "a red fox"


def test_enriched_text_that_breaks_the_rules_is_not_used(client, net):
    net.llm_text = "a blocked-example scene"
    r = gen(client, uid(), prompt="a red fox")
    assert r.status_code == 200 and r.headers["x-prompt-enriched"] == "false"


def test_blocked_term_is_400_and_no_backend_call(client, net):
    r = gen(client, uid(), prompt="draw a Blocked-Example now")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "content_policy_violation"
    assert "blocked" not in r.json()["error"]["message"].lower()
    assert "backend" not in net.names() and "llm" not in net.names()


def test_too_long_prompt_is_400(make_client, net):
    c = make_client(max_prompt_chars=10)
    assert gen(c, uid(), prompt="x" * 11).status_code == 400
    assert "backend" not in net.names()


@pytest.mark.parametrize("status", [500, 503, 400])
def test_backend_failure_is_502(client, net, status):
    net.backend_status = status
    r = gen(client, uid())
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "image_backend_error"


def test_backend_non_image_is_502(client, net):
    net.backend_body = {"data": [{"b64_json": base64.b64encode(b"<html>").decode()}]}
    r = gen(client, uid())
    assert r.status_code == 502 and r.json()["error"]["code"] == "not_an_image"


def test_edits_multipart(client, net):
    r = client.post("/v1/images/edits", headers=headers(uid()),
                    data={"model": "image-proxy-mockup", "prompt": "make it blue"},
                    files={"image": ("in.png", gradient(b"in", 8), "image/png"),
                           "mask": ("m.png", gradient(b"m", 8), "image/png")})
    assert r.status_code == 200
    assert base64.b64decode(r.json()["data"][0]["b64_json"]).startswith(b"\x89PNG")
    sent = [c for c in net.calls if "mock-backend" in str(c.url)][0]
    assert str(sent.url).endswith("/v1/images/edits")
    assert b'name="image"' in sent.content and b'name="mask"' in sent.content


def test_edits_without_image_is_400(client):
    r = client.post("/v1/images/edits", headers=headers(uid()),
                    data={"model": "image-proxy-mockup", "prompt": "x"},
                    files={"other": ("a.txt", b"x", "text/plain")})
    assert r.status_code == 400


def test_edits_as_json_is_400(client):
    r = client.post("/v1/images/edits", headers=headers(uid()),
                    json={"model": "image-proxy-mockup", "prompt": "x"})
    assert r.status_code == 400


def test_unknown_model_is_400(client):
    r = gen(client, uid(), model="something-else")
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_found"


def test_bad_n_and_format(client):
    assert gen(client, uid(), n=9).status_code == 400
    assert gen(client, uid(), response_format="url").status_code == 400


def test_revised_prompt_from_the_image_code_reaches_the_response(client, monkeypatch):
    async def own(**kw):
        return your_image_api.ImageOutput(images=[gradient(b"own", 8)],
                                          revised_prompt="the prompt my code used")
    monkeypatch.setattr(your_image_api, "create_image", own)
    r = gen(client, uid())
    assert r.json()["data"][0]["revised_prompt"] == "the prompt my code used"


def test_a_step_list_without_enrich_makes_no_llm_call(client, net, monkeypatch):
    monkeypatch.setattr(pipeline, "STEPS", [s for s in pipeline.STEPS if s[0] != "enrich"])
    r = gen(client, uid(), prompt="a red fox")
    assert r.status_code == 200 and "llm" not in net.names()
    assert r.json()["data"][0]["revised_prompt"] == "a red fox"


def test_deadline_is_504(make_client, monkeypatch):
    async def slow(**kw):
        await asyncio.sleep(5)
    monkeypatch.setattr(your_image_api, "create_image", slow)
    c = make_client(request_deadline_seconds=1.0, enrich_timeout_seconds=0.5)
    r = gen(c, uid())
    assert r.status_code == 504 and r.json()["error"]["code"] == "deadline_exceeded"


def test_refuses_to_start_when_enrich_model_is_served_here():
    with pytest.raises(ConfigError, match="loop"):
        settings(enrich_model="image-proxy-mockup")


@pytest.mark.parametrize("over", [
    {"proxy_api_key": ""}, {"user_id_header": "not a header"}, {"llm_base_url": ""}, {"enrich_model": ""},
    {"request_deadline_seconds": 300}, {"enrich_timeout_seconds": 300.0, "request_deadline_seconds": 240},
])
def test_refuses_to_start_on_bad_settings(over):
    with pytest.raises(ConfigError):
        settings(**over)


def test_healthz_needs_no_auth(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_no_answer_uses_a_status_that_makes_litellm_cool_the_model_down(client, net):
    # some gateways pause a model after a 401, 404, 408 or 429
    cooling = {401, 404, 408, 429}
    answers = [gen(client, None), gen(client, "garbage"), gen(client, uid(), model="x"),
               gen(client, uid(), prompt="blocked-example")]
    assert not cooling & {r.status_code for r in answers}


def test_proxy_key_never_in_an_answer(client):
    r = gen(client, None)
    assert PROXY_KEY not in r.text
