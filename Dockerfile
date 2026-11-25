# One image for both the proxy and the mock backend (they differ only in the command).
FROM docker.io/library/python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /venv && /venv/bin/pip install --no-deps -r /tmp/requirements.txt

FROM docker.io/library/python:3.12-slim
ENV PATH=/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# The image has no pip at run time: nothing can be installed into a running pod.
RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/* \
    && rm -rf /usr/local/lib/python3.12/site-packages/pip /usr/local/lib/python3.12/site-packages/pip-* \
              /usr/local/bin/pip* \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=build /venv /venv
RUN rm -rf /venv/lib/python3.12/site-packages/pip /venv/lib/python3.12/site-packages/pip-* /venv/bin/pip*
WORKDIR /srv
COPY app/ app/
COPY mock_backend/ mock_backend/
USER 10001
EXPOSE 8080
# The proxy. For the mock backend:
#   uvicorn mock_backend.main:app --host 0.0.0.0 --port 8081 --no-access-log
CMD ["uvicorn", "app.main:app", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
