# LangGraph FastAPI — standalone image (build from this directory).
#
# Build (from repo root):
#   docker build -f backend/langgraph/Dockerfile -t chatbot-langgraph:latest backend/langgraph
#
# When Docker Hub / python image pull times out, override the base image, e.g.:
#   docker build -f backend/langgraph/Dockerfile \
#     --build-arg PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.13-slim-bookworm \
#     -t chatbot-langgraph:latest backend/langgraph
#
# When apt hits 502 from deb.debian.org (CDN), use a Debian mirror hostname (no https://):
#   --build-arg APT_MIRROR_HOST=mirrors.aliyun.com
#   --build-arg APT_MIRROR_HOST=mirrors.tuna.tsinghua.edu.cn
#
# Run (secrets via env; see scripts/docker-entrypoint.sh for required vars):
#   docker run --rm -p 8001:8001 --env-file .env.production chatbot-langgraph:latest

ARG PYTHON_IMAGE=python:3.13-slim-bookworm

# --- dependencies + wheel build ---
FROM ${PYTHON_IMAGE} AS builder
ARG APT_MIRROR_HOST
WORKDIR /app

RUN set -eux; \
    if [ -n "${APT_MIRROR_HOST}" ]; then \
      find /etc/apt -type f \( -name '*.sources' -o -name 'sources.list' \) -exec sed -i \
        -e "s/deb.debian.org/${APT_MIRROR_HOST}/g" \
        -e "s/security.debian.org/${APT_MIRROR_HOST}/g" \
        {} \; ; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN chmod +x /app/scripts/docker-entrypoint.sh \
    && uv sync --frozen --no-dev

# --- runtime ---
FROM ${PYTHON_IMAGE} AS production
ARG APT_MIRROR_HOST
WORKDIR /app

ARG APP_ENV=production
ENV APP_ENV=${APP_ENV} \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PATH="/app/.venv/bin:$PATH"

RUN set -eux; \
    if [ -n "${APT_MIRROR_HOST}" ]; then \
      find /etc/apt -type f \( -name '*.sources' -o -name 'sources.list' \) -exec sed -i \
        -e "s/deb.debian.org/${APT_MIRROR_HOST}/g" \
        -e "s/security.debian.org/${APT_MIRROR_HOST}/g" \
        {} \; ; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
      libpq5 \
      curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

COPY --from=builder --chown=appuser:appuser /app /app
# WORKDIR created /app as root before COPY; ensure appuser can write (e.g. /app/logs).
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

# Align with Makefile / vite default (PORT=8001)
EXPOSE 8001

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--loop", "uvloop"]
