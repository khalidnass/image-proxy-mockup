"""Extra parameters: declared in the marked file, two ways in, checked, never logged, never to the LLM."""
import json

import pytest

from app import extras, your_image_api
from app.config import ConfigError
from conftest import gen, headers, uid
from mock_backend.png import gradient


@pytest.fixture
def seen(monkeypatch):
    """What the image code received."""
    got = {}

    async def create(**kw):
        got.update(kw)
        return your_image_api.ImageOutput(images=[gradient(b"x", 8)])

    async def edit(**kw):
        got.update(kw)
        return your_image_api.ImageOutput(images=[gradient(b"y", 8)])

    monkeypatch.setattr(your_image_api, "create_image", create)
    monkeypatch.setattr(your_image_api, "edit_image", edit)
    return got


def _edit(client, token, data, hdr=None):
    return client.post("/v1/images/edits", headers={**headers(token), **(hdr or {})},
                       data={"model": "image-proxy-mockup", "prompt": "make it blue", **data},
                       files={"image": ("in.png", gradient(b"in", 8), "image/png")})


def test_body_json_on_create(client, seen):
    r = gen(client, uid(), steps=20, seed=7, negative_prompt="no fog")
    assert r.status_code == 200
    assert seen["extra"] == {"steps": 20, "seed": 7, "negative_prompt": "no fog"}


def test_form_fields_on_edit_are_typed(client, seen):
    r = _edit(client, uid(), {"steps": "30", "negative_prompt": "no fog"})
    assert r.status_code == 200
    assert seen["extra"] == {"steps": 30, "negative_prompt": "no fog"}


def test_header_on_create_and_on_edit(client, seen):
    hdr = {"x-image-params": json.dumps({"seed": 5, "steps": 12})}
    r = client.post("/v1/images/generations", headers={**headers(uid()), **hdr},
                    json={"model": "image-proxy-mockup", "prompt": "a fox"})
    assert r.status_code == 200 and seen["extra"] == {"seed": 5, "steps": 12}
    assert _edit(client, uid(), {}, hdr).status_code == 200
    assert seen["extra"] == {"seed": 5, "steps": 12}


def test_the_header_wins(client, seen):
    r = client.post("/v1/images/generations",
                    headers={**headers(uid()), "x-image-params": json.dumps({"steps": 9})},
                    json={"model": "image-proxy-mockup", "prompt": "a fox", "steps": 50, "seed": 1})
    assert r.status_code == 200 and seen["extra"] == {"steps": 9, "seed": 1}


def test_no_extras_is_an_empty_dict(client, seen):
    assert gen(client, uid()).status_code == 200 and seen["extra"] == {}


@pytest.mark.parametrize("how", ["body", "header", "form"])
def test_unknown_name_is_400_and_no_image_call(client, net, seen, how):
    token = uid()
    if how == "body":
        r = gen(client, token, guidance=3)
    elif how == "header":
        r = client.post("/v1/images/generations",
                        headers={**headers(token), "x-image-params": json.dumps({"guidance": 3})},
                        json={"model": "image-proxy-mockup", "prompt": "a fox"})
    else:
        r = _edit(client, token, {"guidance": "3"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_extra_parameter"
    assert seen == {} and "llm" not in net.names()


def test_standard_openai_fields_are_accepted(client, seen):
    r = gen(client, uid(), quality="high", user="someone", output_format="png")
    assert r.status_code == 200 and seen["extra"] == {}


@pytest.mark.parametrize("value", ["twenty", 2.5, True, None, [1]])
def test_wrong_type_is_400(client, seen, value):
    r = gen(client, uid(), steps=value)
    assert r.status_code == 400 and seen == {}


def test_wrong_type_in_a_form_field_is_400(client, seen):
    assert _edit(client, uid(), {"steps": "lots"}).status_code == 400 and seen == {}


def test_a_value_is_never_echoed(client):
    secret = "hidden-value-4471"
    for r in (gen(client, uid(), steps=secret),
              gen(client, uid(), negative_prompt="x" * 5000 + secret),
              gen(client, uid(), negative_prompt="a blocked-example " + secret)):
        assert r.status_code == 400 and secret not in r.text


def test_blocked_term_in_a_string_extra_is_400(client, net, seen):
    r = gen(client, uid(), negative_prompt="a Blocked-Example here")
    assert r.status_code == 400 and r.json()["error"]["code"] == "content_policy_violation"
    assert seen == {} and "llm" not in net.names()


def test_too_long_string_is_400(make_client, seen):
    c = make_client(max_prompt_chars=20)
    assert gen(c, uid(), prompt="a fox", negative_prompt="x" * 21).status_code == 400


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '"text"', json.dumps({"steps": "x" * 5000})])
def test_bad_or_oversize_header_is_400(client, seen, raw):
    r = client.post("/v1/images/generations", headers={**headers(uid()), "x-image-params": raw},
                    json={"model": "image-proxy-mockup", "prompt": "a fox"})
    assert r.status_code == 400 and seen == {}


def test_a_non_finite_number_is_400(monkeypatch, make_client, seen):
    monkeypatch.setitem(your_image_api.EXTRA_PARAMS, "scale", float)
    c = make_client()  # the declaration is read when the app is made
    ok = _edit(c, uid(), {"scale": "1.5"})
    assert ok.status_code == 200 and seen["extra"] == {"scale": 1.5}
    seen.clear()
    for r in (_edit(c, uid(), {"scale": "nan"}),
              c.post("/v1/images/generations",
                     headers={**headers(uid()), "x-image-params": '{"scale": Infinity}'},
                     json={"model": "image-proxy-mockup", "prompt": "a fox"})):
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_extra_parameter"
    assert seen == {}


def test_extras_never_reach_the_enrichment_call(client, net):
    gen(client, uid(), steps=20, seed=7, negative_prompt="no-fog-marker")
    llm = [c for c in net.calls if str(c.url).endswith("/chat/completions")][0]
    body = llm.content.decode()
    assert "no-fog-marker" not in body and "steps" not in body and "seed" not in body


def test_extras_reach_the_mock_backend_and_change_the_picture(client, net):
    a = gen(client, uid(), prompt="a fox")
    sent = [c for c in net.calls if "mock-backend" in str(c.url)][-1]
    assert "steps" not in json.loads(sent.content)
    gen(client, uid(), prompt="a fox", steps=20)
    sent = [c for c in net.calls if "mock-backend" in str(c.url)][-1]
    assert json.loads(sent.content)["steps"] == 20


def test_mock_backend_folds_extras_into_the_picture():
    import base64
    from fastapi.testclient import TestClient
    from mock_backend.main import app
    c = TestClient(app)

    def pic(**extra):
        r = c.post("/v1/images/generations", json={"prompt": "a fox", "size": "32x32", **extra})
        return base64.b64decode(r.json()["data"][0]["b64_json"])

    assert pic() != pic(steps=20) != pic(steps=21)
    assert pic(seed=3) == pic(seed=3)
    e1 = c.post("/v1/images/edits", data={"prompt": "x", "negative_prompt": "a"},
                files={"image": ("i.png", gradient(b"i", 8), "image/png")}).json()
    e2 = c.post("/v1/images/edits", data={"prompt": "x", "negative_prompt": "b"},
                files={"image": ("i.png", gradient(b"i", 8), "image/png")}).json()
    assert e1["data"][0]["b64_json"] != e2["data"][0]["b64_json"]


def test_values_never_reach_the_logs(client, capsys):
    capsys.readouterr()
    token = uid()
    gen(client, token, steps=20, seed=777001, negative_prompt="secret-negative-7731")
    _edit(client, token, {"negative_prompt": "secret-negative-7731"},
          {"x-image-params": json.dumps({"seed": 777001})})
    gen(client, token, negative_prompt="blocked-example secret-negative-7731")
    gen(client, token, steps="secret-negative-7731")
    out = capsys.readouterr().out
    assert "guardrail_in" in out
    for value in ("secret-negative-7731", "777001"):
        assert value not in out


@pytest.mark.parametrize("bad", [{"model": str}, {"steps": list}, {"two words": int}])
def test_a_bad_declaration_refuses_to_start(bad):
    with pytest.raises(ConfigError):
        extras.check_declared(bad)


def test_the_shipped_declaration_is_valid():
    assert extras.check_declared(dict(your_image_api.EXTRA_PARAMS)) == {
        "steps": int, "seed": int, "negative_prompt": str}
