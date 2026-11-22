"""LangChain-free twin of app/services/model_provider.py, built on LiteLLM.

Every function takes `paid` (True = OpenAI, False = local Ollama + MiniLM):

    response = await chat(messages, paid)                          # answers
    verdict  = await judge(messages, paid, response_format=MyModel)  # grades
    vectors  = await embed(texts, paid)                            # ingest (many texts)
    vector   = await embed_query(question, paid)                   # retrieval (one text)
    get_llm_profile(paid).table_name                               # which pgvector table

The embedding model and table_name MUST match what the table was built with
(documents = text-embedding-3-large / 3072 dims, documents_free = MiniLM / 384 dims).
"""

import asyncio
from dataclasses import dataclass
from functools import lru_cache

import litellm

from app.config import get_settings

litellm.suppress_debug_info = True

PAID_LLM = "openai/gpt-5.4-nano"
PAID_EMBEDDINGS = "openai/text-embedding-3-large"
FREE_LLM = "ollama_chat/qwen2.5:7b"   # already pulled in Ollama on the host
FREE_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"  # downloaded in the Dockerfile
EMBED_BATCH_SIZE = 100


# --------------------------------------------------------------------------
# Profiles: just names and settings, no behaviour
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMProfile:
    name: str              # shown in the API response / UI
    chat_model: str        # litellm model string that answers questions
    judge_model: str       # litellm model string that grades answers
    embedding_model: str   # model that turns text into vectors
    llm_kwargs: dict       # extra args for every LLM call (api_key / api_base, temperature, ...)
    table_name: str        # pgvector table built with THIS embedding model
    max_concurrency: int   # how many LLM calls run at once


def resolve_paid(paid: bool | None) -> bool:
    """None → use the .env default (USE_PAID_MODELS); True/False → caller's choice."""
    return get_settings().use_paid_models if paid is None else paid


@lru_cache(maxsize=2)          # build each profile once, reuse forever
def get_llm_profile(paid: bool) -> LLMProfile:
    settings = get_settings()
    if paid:
        return LLMProfile(
            name="openai",
            chat_model=PAID_LLM,
            judge_model=PAID_LLM,
            embedding_model=PAID_EMBEDDINGS,
            llm_kwargs={"api_key": settings.openai_api_key, "temperature": 0},
            table_name="documents",
            max_concurrency=8,
        )
    return LLMProfile(
        name="free (ollama + minilm)",
        chat_model=FREE_LLM,
        judge_model=FREE_LLM,
        embedding_model=FREE_EMBEDDINGS,
        llm_kwargs={"api_base": settings.ollama_base_url, "temperature": 0, "num_ctx": 4096},
        table_name="documents_free",
        max_concurrency=2,     # local Ollama can't do much in parallel
    )


# --------------------------------------------------------------------------
# LLM calls
# --------------------------------------------------------------------------
async def chat(messages: list[dict], paid: bool, **overrides):
    """Call the chat model. Extra keyword args (response_format=..., temperature=...) override the profile."""
    profile = get_llm_profile(paid)
    return await litellm.acompletion(model=profile.chat_model, messages=messages,
                                     **{**profile.llm_kwargs, **overrides})


async def judge(messages: list[dict], paid: bool, **overrides):
    """Same as chat(), but with the judge model."""
    profile = get_llm_profile(paid)
    return await litellm.acompletion(model=profile.judge_model, messages=messages,
                                     **{**profile.llm_kwargs, **overrides})


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_local_model(name: str):
    """Load the sentence-transformers model once, on first use."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


async def embed(texts: list[str], paid: bool) -> list[list[float]]:
    """Turn many texts into vectors."""
    profile = get_llm_profile(paid)

    if not paid:
        # Local model: encode() is slow and blocking, so run it in a thread
        # to keep FastAPI responsive.
        model = _load_local_model(profile.embedding_model)
        vectors = await asyncio.to_thread(model.encode, texts, batch_size=32)
        return vectors.tolist()

    # OpenAI: send the texts in batches.
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = await litellm.aembedding(model=profile.embedding_model, input=batch,
                                            api_key=profile.llm_kwargs["api_key"])
        vectors.extend(item["embedding"] for item in response.data)
    return vectors


async def embed_query(text: str, paid: bool) -> list[float]:
    """Turn one text (e.g. the user's question) into a vector."""
    return (await embed([text], paid))[0]
