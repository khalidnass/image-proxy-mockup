"""The three servers (OpenAI format): create, edit and the LLM, each with its own address, model and key."""
import json

import pytest

from app.config import ConfigError, from_env
from conftest import PROXY_KEY, gen, headers, settings, uid
from mock_backend.png import gradient

CREATE_KEY = "create-server-key-0123456789"
EDIT_KEY = "edit-server-key-0123456789"
LLM_KEY = "llm-server-key-0123456789"


@pytest.fixture
def three(make_client):
    return make_client(image_create_url="http://create.test:8000", image_create_model="qwen-image",
                       image_create_api_key=CREATE_KEY,
                       image_edit_url="http://edit.test:8000", image_edit_model="qwen-image-edit",
                       image_edit_api_key=EDIT_KEY, llm_api_key=LLM_KEY)


def _edit(client, who="alice", **data):
    return client.post("/v1/images/edits", headers=headers(who),
                       data={"model": "image-proxy-mockup", "prompt": "make it blue", **data},
                       files={"image": ("in.png", gradient(b"in", 8), "image/png")})


def _calls(net, host):
    return [c for c in net.calls if c.url.host == host]


def test_create_goes_to_its_server_with_its_model_and_key(three, net):
    assert gen(three, uid("alice"), size="1024x1024").status_code == 200
    (call,) = _calls(net, "create.test")
    assert call.url.path == "/v1/images/generations"
    assert call.headers["authorization"] == f"Bearer {CREATE_KEY}"
    body = json.loads(call.content)
    assert body["model"] == "qwen-image"  # the configured model, not the name LiteLLM sent
    assert body["user"] == "alice" and body["response_format"] == "b64_json"
    assert body["size"] == "1024x1024" and body["n"] == 1
    assert not _calls(net, "edit.test")


def test_edit_goes_to_its_server_with_its_model_and_key(three, net):
    assert _edit(three).status_code == 200
    (call,) = _calls(net, "edit.test")
    assert call.url.path == "/v1/images/edits"
    assert call.headers["authorization"] == f"Bearer {EDIT_KEY}"
    assert call.headers["content-type"].startswith("multipart/form-data")
    for field, value in ((b"model", b"qwen-image-edit"), (b"user", b"alice"), (b"response_format", b"b64_json")):
        assert b'name="' + field + b'"\r\n\r\n' + value in call.content
    assert b'name="image"; filename=' in call.content
    assert not _calls(net, "create.test")


def test_the_llm_goes_to_its_server_with_its_key_and_the_user(three, net):
    gen(three, uid("alice"))
    llm = [c for c in net.calls if c.url.path.endswith("/chat/completions")][0]
    assert llm.headers["authorization"] == f"Bearer {LLM_KEY}"
    assert json.loads(llm.content)["user"] == "alice"


def test_no_key_means_no_authorization_header(make_client, net):
    c = make_client(image_create_url="http://create.test:8000")
    gen(c, uid())
    assert "authorization" not in _calls(net, "create.test")[0].headers


def test_each_server_sees_only_its_own_key(three, net):
    gen(three, uid())
    _edit(three)
    for call in net.calls:
        sent = str(call.headers).encode() + call.content
        own = {"create.test": CREATE_KEY, "edit.test": EDIT_KEY}.get(call.url.host, LLM_KEY)
        for key in (CREATE_KEY, EDIT_KEY, LLM_KEY, PROXY_KEY):
            if key != own:
                assert key.encode() not in sent, (call.url.host, key[:6])


def test_the_image_code_gets_only_the_key_of_its_server(three, monkeypatch):
    from app import your_image_api
    got = {}

    async def create(**kw):
        got["create"] = kw["settings"]
        return your_image_api.ImageOutput(images=[gradient(b"x", 8)])

    async def edit(**kw):
        got["edit"] = kw["settings"]
        return your_image_api.ImageOutput(images=[gradient(b"y", 8)])

    monkeypatch.setattr(your_image_api, "create_image", create)
    monkeypatch.setattr(your_image_api, "edit_image", edit)
    gen(three, uid())
    _edit(three)
    c, e = got["create"], got["edit"]
    assert (c.image_create_api_key, c.image_edit_api_key, c.llm_api_key, c.proxy_api_key) == (CREATE_KEY, "", "", "")
    assert (e.image_create_api_key, e.image_edit_api_key, e.llm_api_key, e.proxy_api_key) == ("", EDIT_KEY, "", "")


def test_qwen_inputs_are_passed_to_the_image_server_by_their_own_names(three, net):
    r = gen(three, uid(), negative_prompt="blur", true_cfg_scale=4.0, num_inference_steps=30, seed=7)
    assert r.status_code == 200
    body = json.loads(_calls(net, "create.test")[0].content)
    assert (body["negative_prompt"], body["true_cfg_scale"], body["num_inference_steps"], body["seed"]) == \
        ("blur", 4.0, 30, 7)


@pytest.mark.parametrize("name", ["guidance_scale", "steps", "height", "width", "generator", "latents"])
def test_any_other_input_is_refused(three, net, name):
    assert gen(three, uid(), **{name: 1}).status_code == 400
    assert not net.calls


def test_a_missing_prompt_is_400_with_a_plain_sentence(three, net):
    r = three.post("/v1/images/generations", headers=headers("alice"), json={"model": "image-proxy-mockup"})
    assert r.status_code == 400 and r.json()["error"]["message"] == "A prompt is required."
    r = three.post("/v1/images/edits", headers=headers("alice"), data={"model": "image-proxy-mockup"},
                   files={"image": ("in.png", gradient(b"in", 8), "image/png")})
    assert r.status_code == 400 and r.json()["error"]["message"] == "A prompt is required."
    assert not net.calls


def test_an_edit_without_an_image_is_400_with_a_plain_sentence(three, net):
    r = three.post("/v1/images/edits", headers=headers("alice"),
                   data={"model": "image-proxy-mockup", "prompt": "x"},
                   files={"mask": ("m.png", gradient(b"m", 8), "image/png")})
    assert r.status_code == 400 and r.json()["error"]["message"] == "An image file is required."
    assert not net.calls


def test_no_size_lets_the_server_choose(three, net):
    gen(three, uid())
    assert "size" not in json.loads(_calls(net, "create.test")[0].content)


ENV = {"PROXY_API_KEY": PROXY_KEY, "LLM_BASE_URL": "http://llm/v1", "ENRICH_MODEL": "chat",
       "IMAGE_CREATE_URL": "http://create:8000", "IMAGE_CREATE_MODEL": "qwen-image",
       "IMAGE_EDIT_URL": "http://edit:8000/v1/", "IMAGE_EDIT_MODEL": "qwen-image-edit"}


def test_the_server_addresses_are_taken_as_roots():
    s = from_env(ENV)
    assert s.image_create_url == "http://create:8000" and s.image_edit_url == "http://edit:8000"


@pytest.mark.parametrize("missing", ["IMAGE_CREATE_URL", "IMAGE_CREATE_MODEL", "IMAGE_EDIT_URL",
                                     "IMAGE_EDIT_MODEL", "LLM_BASE_URL", "ENRICH_MODEL"])
def test_a_missing_setting_refuses_to_start(missing):
    with pytest.raises(ConfigError, match=missing):
        from_env({k: v for k, v in ENV.items() if k != missing})


def test_the_old_backend_setting_alone_refuses_to_start_naming_the_new_ones():
    old = {k: v for k, v in ENV.items() if not k.startswith("IMAGE_")}
    with pytest.raises(ConfigError, match="IMAGE_CREATE_URL"):
        from_env({**old, "IMAGE_BACKEND_URL": "http://old:8081"})


def test_the_keys_never_reach_an_answer(three, net):
    net.backend_status = 500
    answers = [gen(three, uid()), _edit(three), gen(three, None), gen(three, "bad id")]
    for r in answers:
        for key in (CREATE_KEY, EDIT_KEY, LLM_KEY, PROXY_KEY):
            assert key not in r.text


def test_the_keys_never_reach_the_logs(three, capsys):
    capsys.readouterr()
    gen(three, uid())
    _edit(three)
    out = capsys.readouterr().out
    assert "guardrail_in" in out
    for key in (CREATE_KEY, EDIT_KEY, LLM_KEY, PROXY_KEY):
        assert key not in out
