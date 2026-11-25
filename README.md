# image-proxy-mockup

An image proxy that sits behind LiteLLM as a model. It answers the OpenAI image
calls (`/v1/images/generations`, `/v1/images/edits`), checks the content, makes
the prompt better with an LLM called with the user's own token, and calls your
image API. It is a template: you add your own code in one marked place.

```
 app ──► gateway ──► LiteLLM ──► image proxy ──► your image API
         checks the JWT              │
                                     └──► gateway ──► LiteLLM ──► chat model
                                          (enrichment, with the user's JWT)
```

| Hop | `Authorization` | `x-user-token` |
|---|---|---|
| app → gateway | `Bearer <user JWT>` | replaced by the gateway |
| gateway → LiteLLM | `Bearer <LiteLLM key>` | the user JWT |
| LiteLLM → proxy | `Bearer <PROXY_API_KEY>` | the user JWT (this model only) |
| proxy → gateway | `Bearer <user JWT>` | set again by the gateway |

## Add your API or your code

Open `app/your_image_api.py`: `create_image(...)` and `edit_image(...)` sit under the banner
**PUT YOUR API CALL OR YOUR OWN CODE HERE**. Either point `IMAGE_BACKEND_URL` at your endpoint and adapt
the request and response mapping, or replace the function body with your own code.

- **In:** `prompt`, `model`, `n`, `size`, `user` (verified name), `http`, `settings`; edit adds `image`, `mask`.
- **Out:** `ImageOutput(images=[raw bytes, ...], revised_prompt=None)`.
- **Error:** raise `ImageBackendError("short message")`; the caller gets 502.

If your code improves the prompt itself, remove `enrich` from `STEPS` in `app/pipeline.py`.

## Run the proxy

| Setting | What it is |
|---|---|
| `PROXY_API_KEY` | The key LiteLLM sends (the model's `api_key`). Keep it in a Secret. |
| `OIDC_ISSUER` | Your identity provider's issuer, as in the token's `iss`. |
| `LLM_BASE_URL` | The gateway's OpenAI address, for example `https://llm.example.com/v1`. |
| `ENRICH_MODEL` | The chat model that improves the prompt. |
| `IMAGE_BACKEND_URL` | Your image API, or the mock (`mock_backend/`). |

Other settings: `app/config.py`. Kubernetes: `deploy/`. Locally, the mock plays the image API and the LLM:

```bash
pip install -r requirements-dev.txt && pytest
uvicorn mock_backend.main:app --port 8081 &
export PROXY_API_KEY=$(openssl rand -hex 32) OIDC_ISSUER=https://id.example.com/realms/<realm>
export LLM_BASE_URL=http://127.0.0.1:8081/v1 ENRICH_MODEL=mock-llm IMAGE_BACKEND_URL=http://127.0.0.1:8081
uvicorn app.main:app --factory --port 8080
```

## Stage 1 — with LiteLLM

Set up: add the model and the forward setting to LiteLLM's `config.yaml`, then restart LiteLLM.

```yaml
model_list:
  - model_name: image-proxy-mockup
    litellm_params: {model: openai/image-proxy-mockup, api_base: "http://image-proxy.<namespace>.svc:8080/v1",
                     api_key: os.environ/IMAGE_PROXY_API_KEY}   # the proxy's PROXY_API_KEY
    model_info: {mode: image_generation}
litellm_settings:
  model_group_settings:
    forward_client_headers_to_llm_api: [image-proxy-mockup]
```

Get a user token (Keycloak), kept in a variable and never echoed. Use a client the gateway's policy
accepts, or the gateway refuses the token:

```bash
KC=https://<keycloak host>/realms/<realm>
DEV=$(curl -s -d client_id=<client id> "$KC/protocol/openid-connect/auth/device")
echo "$DEV" | jq -r .verification_uri_complete    # open it, sign in, then:
USER_TOKEN=$(curl -s -d client_id=<client id> -d "device_code=$(echo "$DEV" | jq -r .device_code)" \
  -d grant_type=urn:ietf:params:oauth:grant-type:device_code \
  "$KC/protocol/openid-connect/token" | jq -r .access_token)
# a TEST user only, if your client allows direct access grants:
read -rs PW; USER_TOKEN=$(curl -s -d client_id=<client id> -d grant_type=password \
  -d username=<test user> --data-urlencode "password=$PW" "$KC/protocol/openid-connect/token" | jq -r .access_token)
```

Test through LiteLLM (a port-forward and a LiteLLM key; you set `x-user-token` by hand):

```bash
LITELLM=http://localhost:4000 KEY='<a LiteLLM key that holds image-proxy-mockup>'
curl -s -o create.json -w '%{http_code}\n' "$LITELLM/v1/images/generations" \
  -H "Authorization: Bearer $KEY" -H "x-user-token: $USER_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model": "image-proxy-mockup", "prompt": "a lighthouse at dusk", "size": "256x256"}'   # 200
jq -r '.data[0].b64_json' create.json | base64 -d > create.png && file create.png        # PNG image data
curl -s -o edit.json -w '%{http_code}\n' "$LITELLM/v1/images/edits" \
  -H "Authorization: Bearer $KEY" -H "x-user-token: $USER_TOKEN" \
  -F model=image-proxy-mockup -F prompt='make it blue' -F image=@create.png             # 200
curl -s -o /dev/null -w '%{http_code}\n' "$LITELLM/v1/images/generations" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model": "image-proxy-mockup", "prompt": "a lighthouse"}'                         # 403: no user token
```

## Stage 2 — behind a gateway

The gateway in front of LiteLLM (for example Kuadrant / Red Hat Connectivity Link) must:

- check the user's token, and refuse a call without a valid one;
- set `Authorization` to the LiteLLM key;
- copy the user's token into `x-user-token`, replacing any value the caller sent;
- let `/v1/images/` through to LiteLLM.

Test through the gateway, with your own token only:

```bash
GATEWAY=https://llm.example.com
curl -s -o create.json -w '%{http_code}\n' "$GATEWAY/v1/images/generations" \
  -H "Authorization: Bearer $USER_TOKEN" -H 'Content-Type: application/json' \
  -d '{"model": "image-proxy-mockup", "prompt": "a lighthouse at dusk", "size": "256x256"}'   # 200
jq -r '.data[0].b64_json' create.json | base64 -d > create.png && file create.png        # PNG image data
curl -s -o edit.json -w '%{http_code}\n' "$GATEWAY/v1/images/edits" -H "Authorization: Bearer $USER_TOKEN" \
  -F model=image-proxy-mockup -F prompt='make it blue' -F image=@create.png             # 200
curl -s -o /dev/null -w '%{http_code}\n' "$GATEWAY/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"model": "image-proxy-mockup", "prompt": "x"}'                                    # 401: no token
```

## What the answer means

| Answer | Meaning |
|---|---|
| 200 | It works: `data[0].b64_json` is the image, `data[0].revised_prompt` the prompt used. |
| 401, empty body | The gateway refused the token (missing, bad or expired). |
| 401, JSON body | LiteLLM refused the key. |
| 403 | No valid `x-user-token` reached the proxy: header not set, forward setting missing, or wrong model name in it. |
| 404 | The image path is not on the route, or LiteLLM does not know the model. |
| 400 | The content rules refused the prompt, or the request is malformed. |
| 502 | The image API failed or did not return an image. |

## Three rules

- The gateway sets `x-user-token` and must replace any value the caller sent.
- Name only this model in `forward_client_headers_to_llm_api`: every model listed there receives the token.
- Never log or store the token (the proxy logs only request id, step, status, time and user name).

MIT licence, see [LICENSE](LICENSE).
