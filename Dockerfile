# The service image (architecture.md M4): multi-stage — deps resolved from the
# committed lockfile in a uv builder stage, runtime is a plain slim Python image
# with just the venv and the source. No uv, no compilers, no dev deps at runtime.
#
#   docker build -t turnstile .
#   docker run --rm -p 8000:8000 --env-file .env turnstile
#
# Configuration is environment-only (config.py): LLM__*, KB__*, and friends.
# There is deliberately no baked-in config and no .env in the image.
#
# Build-time provenance: `make image` passes
# --build-arg GIT_COMMIT (the immutable source fact — a git TAG can be moved,
# a sha cannot) and DOCKER_TAG (the name the image was built as, so it
# survives inside the image even after a re-tag/push). Both land as OCI
# labels (`docker inspect`) and env vars in the container. Defaults keep
# bare `docker build` working.
ARG GIT_COMMIT=unknown
ARG DOCKER_TAG=unknown

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Bytecode at build time (faster cold start), copy-mode links (the venv must
# not point back into uv's cache — it gets COPY'd out of this stage).
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Layer 1: third-party deps from the lockfile alone — this layer only rebuilds
# when pyproject.toml/uv.lock change, not on every source edit.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Layer 2: the project itself, installed non-editable so the venv stands alone.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim-bookworm

# Re-declare after FROM — Docker scopes ARGs per build stage.
ARG GIT_COMMIT
ARG DOCKER_TAG
LABEL org.opencontainers.image.revision="${GIT_COMMIT}"
LABEL org.opencontainers.image.version="${DOCKER_TAG}"
ENV GIT_COMMIT="${GIT_COMMIT}"
ENV DOCKER_TAG="${DOCKER_TAG}"

RUN useradd --system --no-create-home turnstile
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
USER turnstile
WORKDIR /app

EXPOSE 8000

# The container's liveness IS the service's /health (app.py: liveness + spec
# id, deliberately no upstream probes — a slow LLM must not flap the container).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# ONE worker by design (architecture.md §2): the conversation registry is
# worker-local; scale-out is replicas behind a sticky router, not --workers N.
CMD ["uvicorn", "--factory", "turnstile.service.app:create_app", \
     "--host", "0.0.0.0", "--port", "8000"]
