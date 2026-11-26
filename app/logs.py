"""Logging: one JSON line per event, and only these fields.

A log line holds a request id, a step name, a status, a duration and the
user id (`user`). Never a prompt, an LLM answer, an image or a key.
The redacting filter below is a second line of defence, not the first:
the code never passes those values to a logger.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time

FIELDS = ("request_id", "step", "status", "ms", "user")
_SECRETS = re.compile(r"eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]*\.?[A-Za-z0-9_\-]*|sk-\S+|Bearer\s+\S+")

log = logging.getLogger("image_proxy")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname,
                "event": record.getMessage()}
        for name in FIELDS:
            value = getattr(record, name, None)
            if value is not None:
                line[name] = value
        return json.dumps(line)


class _Redact(logging.Filter):
    def __init__(self, exact: tuple[str, ...] = ()):
        super().__init__()
        self.exact = tuple(s for s in exact if s)

    def _clean(self, text: str) -> str:
        for secret in self.exact:  # the proxy's own keys, by their exact value
            text = text.replace(secret, "[redacted]")
        return _SECRETS.sub("[redacted]", text)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._clean(str(record.msg))
        if record.args:
            record.args = tuple(self._clean(str(a)) for a in record.args)
        return True


class _Stdout(logging.StreamHandler):
    """Writes to whatever sys.stdout is at the time of the write."""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, _value):
        pass


def setup(level: str = "INFO", secrets: list[str] | None = None) -> None:
    handler = _Stdout()
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_Redact(tuple(secrets or ())))
    log.handlers[:] = [handler]
    log.setLevel(level)
    log.propagate = False
    # httpx logs every URL at INFO; keep it quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def event(event: str, *, request_id: str, step: str | None = None, status: str | None = None,
          ms: float | None = None, user: str | None = None, level: int = logging.INFO) -> None:
    log.log(level, event, extra={"request_id": request_id, "step": step, "status": status,
                                 "ms": None if ms is None else round(ms, 1), "user": user})
