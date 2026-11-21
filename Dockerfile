# Runs on the dev laptop (amd64) and on the Pi (arm64) unchanged.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

# portaudio for sounddevice; ALSA->PulseAudio plugin so the container can use the
# host's PipeWire/Pulse socket; espeak-ng only to synthesize test fixtures.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libportaudio2 libasound2-plugins alsa-utils espeak-ng \
    && rm -rf /var/lib/apt/lists/*
COPY deploy/asound.conf /etc/asound.conf

ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 KIKI_MODELS_DIR=/app/models
WORKDIR /app

ARG UV_GROUPS="--group dev"
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project $UV_GROUPS
COPY . .
RUN uv sync $UV_GROUPS

ENTRYPOINT ["kiki"]
CMD ["--help"]
