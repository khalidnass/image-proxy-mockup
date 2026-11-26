# image-proxy-mockup

An image proxy that sits behind LiteLLM as a model. It answers the OpenAI image
calls (`/v1/images/generations`, `/v1/images/edits`), checks the content, makes
the prompt better with an LLM, and calls your image API. It calls the LLM and
the image service directly on the model server, and it knows the person by the
user id the gateway writes: **no user id means no access**. It is a template:
you add your own code in one marked place.

```
 app ──► gateway ──► LiteLLM ──► image proxy ──► model server (directly)
         checks the sign-in,        │              ├─ the LLM (prompt enrichment)
         writes the user id         │              └─ the image service
                                    └─ forwards the user id header to this model only
```

| Hop | `Authorization` | `x-litellm-customer-id` |
|---|---|---|
| app → gateway | `Bearer <the user's sign-in token>` | replaced by the gateway |
| gateway → LiteLLM | `Bearer <LiteLLM key>` | the user id, from the checked sign-in |
| LiteLLM → proxy | `Bearer <PROXY_API_KEY>` | the user id (this model only) |
| proxy → model server | `Bearer <LLM_API_KEY>`, or none | none; the body's `user` carries the user id |

The proxy holds no person's token. It prints the user id on every log line of a request, gives it to your
image code, and sends it to the model server as the body field `user`.

## Add your API or your code

Open `app/your_image_api.py`: `create_image(...)` and `edit_image(...)` sit under the banner
**PUT YOUR API CALL OR YOUR OWN CODE HERE**. Either point `IMAGE_BACKEND_URL` at your endpoint and adapt
the request and response mapping, or replace the function body with your own code.

- **In:** `prompt`, `model`, `n`, `size`, `extra`, `user` (the user id), `http`, `settings`; edit adds
  `image`, `mask`.
- **Out:** `ImageOutput(images=[raw bytes, ...], revised_prompt=None)`.
- **Error:** raise `ImageBackendError("short message")`; the caller gets 502.

If your code improves the prompt itself, remove `enrich` from `STEPS` in `app/pipeline.py`.

### Extra parameters

If your image service takes parameters the OpenAI call does not have, declare them in the same file:
`EXTRA_PARAMS = {"steps": int, "seed": int, "negative_prompt": str}` (types `int`, `float`, `bool`, `str`).
Your functions get them as `extra`, checked and typed; undeclared names never reach them.

- **In the body** (use this first): a top-level JSON field on create, a form field on edit.
- **In the `x-image-params` header** (the spare): one JSON object, at most 4 KB, for a client or a gateway that
  drops body fields it does not know. If a name comes both ways, the header wins.
- **Refused with 400:** an unknown name, a wrong type, a value too long or not finite, a bad header. Text values
  pass the same content rules as the prompt. Values never go to the LLM or to a log line.

## Run the proxy

| Setting | What it is |
|---|---|
| `PROXY_API_KEY` | The key LiteLLM sends (the model's `api_key`). Keep it in a Secret. |
| `LLM_BASE_URL` | The model server's OpenAI address, for example `http://vllm.example.com:8000/v1`. |
| `ENRICH_MODEL` | The chat model on that server that improves the prompt. |
| `IMAGE_BACKEND_URL` | Your image service, or the mock (`mock_backend/`). |
| `LLM_API_KEY`, `IMAGE_API_KEY` | Optional: the model server's and the image service's own keys, from a Secret. Blank: no `Authorization` header. |
| `USER_ID_HEADER` | Where the user id arrives. Default `x-litellm-customer-id`. |
| `TRUST_BODY_USER` | Default `false`. `true` uses the body's `user` when the header is absent; a caller can write that field, so it can be faked. |

Other settings: `app/config.py`. Kubernetes: `deploy/`. Locally, the mock plays the image service and the LLM:

```bash
pip install -r requirements-dev.txt && pytest
uvicorn mock_backend.main:app --port 8081 &
export PROXY_API_KEY=$(openssl rand -hex 32)
export LLM_BASE_URL=http://127.0.0.1:8081/v1 ENRICH_MODEL=mock-llm IMAGE_BACKEND_URL=http://127.0.0.1:8081
uvicorn app.main:app --factory --port 8080
```

## Integrate with LiteLLM behind a gateway

### Stage 1 — LiteLLM

Add the model and the forward setting to LiteLLM's `config.yaml`, then restart LiteLLM. The forward setting is what
carries the gateway's user id header on to the proxy; without it every call is 403.

```yaml
model_list:
  - model_name: image-proxy-mockup
    litellm_params: {model: openai/image-proxy-mockup, api_base: "http://image-proxy.<namespace>.svc:8080/v1",
                     api_key: os.environ/IMAGE_PROXY_API_KEY}   # the proxy's PROXY_API_KEY
    model_info: {mode: image_generation}
litellm_settings:
  model_group_settings:
    forward_client_headers_to_llm_api: [image-proxy-mockup]   # this model only
```

Test straight at LiteLLM (a port-forward and a LiteLLM key; you write the user id header yourself, as the gateway
will):

```bash
LITELLM=http://localhost:4000 KEY='<a LiteLLM key that holds image-proxy-mockup>'
curl -s -o create.json -w '%{http_code}\n' "$LITELLM/v1/images/generations" \
  -H "Authorization: Bearer $KEY" -H 'x-litellm-customer-id: <user id>' -H 'Content-Type: application/json' \
  -d '{"model": "image-proxy-mockup", "prompt": "a lighthouse at dusk", "size": "1024x1024"}'  # 200
jq -r '.data[0].b64_json' create.json | base64 -d > create.png && file create.png        # PNG image data
curl -s -o edit.json -w '%{http_code}\n' "$LITELLM/v1/images/edits" \
  -H "Authorization: Bearer $KEY" -H 'x-litellm-customer-id: <user id>' \
  -F model=image-proxy-mockup -F prompt='make it blue' -F image=@create.png             # 200
curl -s -o /dev/null -w '%{http_code}\n' "$LITELLM/v1/images/generations" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "image-proxy-mockup", "prompt": "a lighthouse"}'                         # 403: no user id
curl -s -o /dev/null -w '%{http_code}\n' "$LITELLM/v1/images/generations" \
  -H "Authorization: Bearer $KEY" -H 'x-litellm-customer-id: <user id>' -H 'Content-Type: application/json' \
  -d '{"model": "image-proxy-mockup", "prompt": "a lighthouse", "steps": 20, "seed": 7}'    # 200: extras in the body
curl -s -o /dev/null -w '%{http_code}\n' "$LITELLM/v1/images/generations" \
  -H "Authorization: Bearer $KEY" -H 'x-litellm-customer-id: <user id>' -H 'Content-Type: application/json' \
  -H 'x-image-params: {"negative_prompt": "fog"}' \
  -d '{"model": "image-proxy-mockup", "prompt": "a lighthouse"}'                         # 200: extras in the header
```

### Stage 2 — the gateway

The gateway in front of LiteLLM (for example Kuadrant / Red Hat Connectivity Link) must:

- check the user's sign-in, and refuse a call without a valid one;
- set `Authorization` to the LiteLLM key;
- write the user id into `x-litellm-customer-id`, replacing any value the caller sent;
- let `/v1/images/` through to LiteLLM.

No token is copied anywhere: the proxy needs only the user id.

Test through the gateway, with your own sign-in. A token from Keycloak's device flow (kept in a variable, never
echoed; use a client the gateway accepts):

```bash
KC=https://<keycloak host>/realms/<realm>
DEV=$(curl -s -d client_id=<client id> "$KC/protocol/openid-connect/auth/device")
echo "$DEV" | jq -r .verification_uri_complete    # open it, sign in, then:
USER_TOKEN=$(curl -s -d client_id=<client id> -d "device_code=$(echo "$DEV" | jq -r .device_code)" \
  -d grant_type=urn:ietf:params:oauth:grant-type:device_code \
  "$KC/protocol/openid-connect/token" | jq -r .access_token)
```

```bash
GATEWAY=https://llm.example.com
curl -s -o create.json -w '%{http_code}\n' "$GATEWAY/v1/images/generations" \
  -H "Authorization: Bearer $USER_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model": "image-proxy-mockup", "prompt": "a lighthouse at dusk", "size": "1024x1024"}'  # 200
jq -r '.data[0].b64_json' create.json | base64 -d > create.png && file create.png        # PNG image data
curl -s -o edit.json -w '%{http_code}\n' "$GATEWAY/v1/images/edits" -H "Authorization: Bearer $USER_TOKEN" \
  -F model=image-proxy-mockup -F prompt='make it blue' -F image=@create.png             # 200
curl -s -o /dev/null -w '%{http_code}\n' "$GATEWAY/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"model": "image-proxy-mockup", "prompt": "x"}'                                    # 401: no sign-in
curl -s -o /dev/null -w '%{http_code}\n' "$GATEWAY/v1/images/edits" -H "Authorization: Bearer $USER_TOKEN" \
  -F model=image-proxy-mockup -F prompt='make it blue' -F steps=20 -F image=@create.png # 200: extras as form fields
curl -s -o /dev/null -w '%{http_code}\n' "$GATEWAY/v1/images/generations" \
  -H "Authorization: Bearer $USER_TOKEN" -H 'Content-Type: application/json' \
  -H 'x-image-params: {"seed": 7}' -d '{"model": "image-proxy-mockup", "prompt": "a lighthouse"}'  # 200: header
```

## What the answer means

| Answer | Meaning |
|---|---|
| 200 | It works: `data[0].b64_json` is the image, `data[0].revised_prompt` the prompt used. |
| 401, empty body | The gateway refused the sign-in (missing, bad or expired). |
| 401, JSON body | LiteLLM refused the key. |
| 403 | No user id reached the proxy: the forward setting is missing or names another model, or the gateway did not write the header. |
| 404 | The image path is not on the route, or LiteLLM does not know the model. |
| 400 | The content rules refused the prompt, the user id is not a plain id, or the request is malformed. |
| 502 | The image service failed or did not return an image. |

## Three rules

- The gateway writes the user id and must replace any value the caller sent.
- Name only this model in `forward_client_headers_to_llm_api`.
- Protect the model server (its own key, or a network rule that lets only the proxy in): calls to it no longer pass
  LiteLLM, so LiteLLM neither checks nor counts them.

MIT licence, see [LICENSE](LICENSE).
