FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
# Where HF will store the models. Baking this in means zero downloads at runtime.
ENV HF_HOME=/opt/huggingface

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Download models into /opt/huggingface at build time.
# This layer is cached—rebuilds that only change your app code won't re-download.
RUN uv run --no-dev python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('openai/whisper-small.en'); \
    snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

COPY app ./app
COPY main.py ./

EXPOSE 8000

CMD ["uv", "run", "python", "main.py"]