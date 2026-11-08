from dataclasses import dataclass
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import get_settings

settings = get_settings()

FREE_LLM = "qwen2.5:3b"  # this model has been already pulled using ollama service on my machine
FREE_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2" # This model has been already pulled in Dockerfile so we can propagate the vector db


@dataclass(frozen=True)
class ModelProfile:
    name: str                  # shown in the API response / UI
    embeddings: Embeddings
    chat_llm: BaseChatModel    # answers the RAG question
    judge_llm: BaseChatModel   # grades the answer
    table_name: str            # pgvector table built with THIS embedding model
    max_concurrency: int       # how many tests run at once


@lru_cache(maxsize=2)          # build each profile once, reuse forever
def get_profile(paid: bool) -> ModelProfile:
    if paid:
        return ModelProfile(
            name="openai",
            embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
            chat_llm=ChatOpenAI(model="gpt-4.1-nano", temperature=0),
            judge_llm=ChatOpenAI(model="gpt-4.1-nano"),
            table_name="documents",
            max_concurrency=8,
        )
    return ModelProfile(
        name="free (ollama + minilm)",
        embeddings=HuggingFaceEmbeddings(model_name=FREE_EMBEDDINGS),
        chat_llm=ChatOllama(model=FREE_LLM, base_url=settings.ollama_base_url, temperature=0, num_ctx=4096),
        judge_llm=ChatOllama(model=FREE_LLM, base_url=settings.ollama_base_url, temperature=0, num_ctx=4096),
        table_name="documents_free",
        max_concurrency=2,     # CPU-only Ollama can't do much in parallel
    )