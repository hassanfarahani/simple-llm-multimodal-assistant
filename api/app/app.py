import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from contextlib import asynccontextmanager
from functools import lru_cache
import asyncpg
from app.routers import messages
from app.routers import meetingminutes
from app.routers import rag
from app.config import get_settings
from app.services.rag_service import init_vectorstore

HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN, add_to_git_credential=False)


whisper_pipe = None


@lru_cache(maxsize=1)
def get_qwen():
    """Load Qwen lazily and cache it. First call is slow, rest are instant."""
    QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(QWEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=torch.float32)
    model.eval()
    return tokenizer, model

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
  # --- startup ---
  # asyncpg does not accept the SQLAlchemy "+psycopg" scheme.
    asyncpg_url = settings.database_url.replace(
        "postgresql+psycopg://",
        "postgresql://",
        1,
    )
    app.state.pool = await asyncpg.create_pool(
        asyncpg_url,
        min_size=1,
        max_size=10,
    )

    app.state.whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small.en",
        dtype=torch.float16,
        device="cpu",
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=5,
    )

    app.state.qwen = get_qwen()  # warm it up

    await init_vectorstore()

    yield

    # --- shutdown ---
    await app.state.pool.close()
    app.state.whisper_pipe = None
    app.state.qwen = None

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(messages.router)

# ***********************************************************************************************
# Option 1: Use Open Source for Transcription - Hugging Face Pipelines
# the openai/whisper-small.en model will be stored in opt/huggingface/hub inside the container
# the audio file will be stored in /tmp inside the container (/tmp/tmpj8ph40a9.mp3)
app.include_router(meetingminutes.router)

# ***********************************************************************************************
app.include_router(rag.router)


# still need to refactor using the folder structure pattern in:
# https://chat.deepseek.com/a/chat/s/131b76e5-2f86-4e5b-82f5-f1bcfabee649