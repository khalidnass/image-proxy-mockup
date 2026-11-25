"""No prompt, no answer, no image and no token in any log line."""
import base64
import logging

from conftest import PROXY_KEY, gen, headers, make_token
from mock_backend.png import gradient

PROMPT = "a secret lighthouse prompt 7731"


def _capture(capsys, caplog):
    out = capsys.readouterr()
    return out.out + out.err + "\n".join(
        f"{r.getMessage()} {r.__dict__}" for r in caplog.records)


def test_no_prompt_token_or_image_in_logs(client, net, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    token = make_token("alice")
    net.llm_text = "an enriched lighthouse answer 9912"
    r = gen(client, token, prompt=PROMPT)
    assert r.status_code == 200
    image_b64 = r.json()["data"][0]["b64_json"]
    # the refusal paths too
    gen(client, token, prompt="a blocked-example " + PROMPT)
    gen(client, token + "x", prompt=PROMPT)
    client.post("/v1/images/edits", headers=headers(token),
                data={"model": "image-proxy-mockup", "prompt": PROMPT},
                files={"image": ("in.png", gradient(b"in", 8), "image/png")})
    net.backend_status = 500
    gen(client, token, prompt=PROMPT)

    logs = _capture(capsys, caplog)
    assert "alice" in logs and "guardrail_in" in logs  # the log is not just empty
    for secret in (PROMPT, "lighthouse", "9912", token, token.split(".")[1], PROXY_KEY,
                   image_b64[:40], "eyJ"):
        assert secret not in logs, f"leaked into logs: {secret[:12]}"


def test_redact_filter_is_a_second_line(capsys):
    from app import logs
    logs.setup()
    logs.log.info("oops Bearer abc.def and %s", make_token())
    out = capsys.readouterr().out
    assert "eyJ" not in out and "abc.def" not in out and "[redacted]" in out


def test_validation_errors_do_not_echo_the_prompt(client):
    r = client.post("/v1/images/generations", headers=headers(make_token()),
                    content=b'{"model": 5, "prompt": "' + PROMPT.encode() + b'"',
                    )
    assert r.status_code == 400 and PROMPT not in r.text


def test_png_writer_makes_valid_pngs():
    import struct, zlib
    data = gradient(b"seed", 4)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    # IHDR crc is right
    length = struct.unpack(">I", data[8:12])[0]
    chunk = data[12:16 + length]
    assert struct.unpack(">I", data[16 + length:20 + length])[0] == zlib.crc32(chunk) & 0xFFFFFFFF
    assert base64.b64encode(data)
