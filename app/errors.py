"""One error type, answered in the OpenAI error shape."""
from __future__ import annotations

from fastapi.responses import JSONResponse


class ProxyError(Exception):
    """Stops the request. `message` goes to the caller: never put a prompt,
    a token or a backend's raw answer in it."""

    def __init__(self, status: int, message: str, *, type_: str = "invalid_request_error",
                 code: str | None = None, param: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.param = param

    def response(self, request_id: str = "") -> JSONResponse:
        headers = {"x-request-id": request_id} if request_id else None
        return JSONResponse(
            status_code=self.status, headers=headers,
            content={"error": {"message": self.message, "type": self.type,
                               "param": self.param, "code": self.code}})


def unauthorized(message: str, code: str) -> ProxyError:
    """The caller is not LiteLLM (wrong or missing PROXY_API_KEY)."""
    return ProxyError(401, message, type_="authentication_error", code=code)


# 403, not 401: some gateways pause a model after a 401.
USER_TOKEN_REFUSAL_STATUS = 403


def user_refused(message: str, code: str) -> ProxyError:
    """The person's token is missing or does not verify."""
    return ProxyError(USER_TOKEN_REFUSAL_STATUS, message, type_="authentication_error", code=code)
