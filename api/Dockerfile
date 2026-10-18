# --------------------------------------------------
# Stage 1: Download the Hugging Face models
# This stage is independent of the application lockfile.
# --------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS models

ENV HF_HOME=/opt/huggingface

RUN uv pip install --system huggingface-hub

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('openai/whisper-small.en'); \
    snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"


# --------------------------------------------------
# Stage 2: Build the application environment
# --------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS api

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/opt/huggingface

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Reuse the models downloaded in Stage 1.
COPY --from=models /opt/huggingface /opt/huggingface

COPY app ./app
COPY main.py ./

EXPOSE 8000

CMD ["uv", "run", "python", "main.py"]
