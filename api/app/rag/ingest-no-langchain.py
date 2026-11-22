"""
Ingest the knowledge base into Postgres/pgvector WITHOUT LangChain.

Same job as app/rag/ingest.py (load .md files -> chunk -> embed -> TRUNCATE + INSERT
into `documents` / `documents_free`), but built from plain Python + LiteLLM,
following the "PRO" ingest from llm_engineering/week5/day5:

  1. fetch_documents()  - homemade replacement for DirectoryLoader/TextLoader
  2. create_chunks()    - an LLM (via litellm) splits each document into overlapping
                          chunks and adds a headline + summary to each one
                          (or `--chunking simple` for a homemade 1000/200 character splitter,
                          the same settings ingest.py uses with RecursiveCharacterTextSplitter)
  3. create_embeddings() - embed and write to pgvector with asyncpg

Run from the api/ folder:
    uv run python -m app.rag.ingest-no-langchain --models free
    uv run python -m app.rag.ingest-no-langchain --models paid
    uv run python -m app.rag.ingest-no-langchain --models free --chunking simple
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

import asyncpg
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

from app.config import get_settings
from app.services.llm_provider import chat, embed, get_llm_profile

load_dotenv()
litellm.suppress_debug_info = True

KNOWLEDGE_BASE = Path(__file__).parent.parent.parent / "knowledge-base"
AVERAGE_CHUNK_SIZE = 500   # used only to hint the LLM how many chunks to make
CHUNK_SIZE = 1000          # --chunking simple (same as ingest.py)
CHUNK_OVERLAP = 200


# --------------------------------------------------------------------------
# Models come from app/services/llm_provider.py: chat() splits documents,
# embed() makes the vectors, get_llm_profile(paid).table_name is where they go.
# The embedding model and table MUST match what the retriever uses at query time.
# --------------------------------------------------------------------------
def ollama_base_url() -> str:
    # .env points at host.docker.internal (for the container); this script runs on the host.
    return get_settings().ollama_base_url.replace("host.docker.internal", "localhost")


# --------------------------------------------------------------------------
# Data classes (inspired by LangChain's Document, as in week5/day5)
# --------------------------------------------------------------------------
class Result(BaseModel):
    page_content: str
    metadata: dict


class Chunk(BaseModel):
    headline: str = Field(description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query")
    summary: str = Field(description="A few sentences summarizing the content of this chunk to answer common questions")
    original_text: str = Field(description="The original text of this chunk from the provided document, exactly as is, not changed in any way")

    def as_result(self, document: dict) -> Result:
        metadata = {"source": document["source"], "doc_type": document["doc_type"]}
        return Result(
            page_content=f"{self.headline}\n\n{self.summary}\n\n{self.original_text}",
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


# --------------------------------------------------------------------------
# Step 1: load documents
# --------------------------------------------------------------------------
def fetch_documents() -> list[dict]:
    """Homemade version of LangChain's DirectoryLoader + TextLoader."""
    documents = []
    for folder in sorted(p for p in KNOWLEDGE_BASE.iterdir() if p.is_dir()):
        doc_type = folder.name
        for file in sorted(folder.rglob("*.md")):
            documents.append({
                "doc_type": doc_type,
                "source": file.as_posix(),
                "text": file.read_text(encoding="utf-8"),
            })
    print(f"Loaded {len(documents)} documents")
    return documents


# --------------------------------------------------------------------------
# Step 2a: LLM chunking (week5/day5 approach)
# --------------------------------------------------------------------------
def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["doc_type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


@retry(wait=wait_exponential(multiplier=1, min=5, max=60), stop=stop_after_attempt(5), reraise=True)
async def process_document(document: dict, paid: bool) -> list[Result]:
    overrides = {}
    if not paid:
        # This script runs on the host, and whole documents need a bigger context window.
        overrides = {"api_base": ollama_base_url(), "num_ctx": 8192}
    response = await chat(
        [{"role": "user", "content": make_prompt(document)}],
        paid,
        response_format=Chunks,
        **overrides,
    )
    reply = response.choices[0].message.content
    return [chunk.as_result(document) for chunk in Chunks.model_validate_json(reply).chunks]


async def create_llm_chunks(documents: list[dict], paid: bool) -> list[Result]:
    semaphore = asyncio.Semaphore(get_llm_profile(paid).max_concurrency)

    async def worker(doc: dict) -> list[Result]:
        async with semaphore:
            return await process_document(doc, paid)

    results = await tqdm.gather(*(worker(d) for d in documents), desc="Chunking with LLM")
    return [chunk for doc_chunks in results for chunk in doc_chunks]


# --------------------------------------------------------------------------
# Step 2b: simple chunking (homemade RecursiveCharacterTextSplitter)
# --------------------------------------------------------------------------
SEPARATORS = ["\n\n", "\n", " ", ""]


def _split_recursive(text: str, separators: list[str]) -> list[str]:
    """Break text into pieces no longer than CHUNK_SIZE, preferring the coarsest separator."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    sep, *rest = separators
    parts = text.split(sep) if sep else list(text)
    pieces = []
    for part in parts:
        if len(part) <= CHUNK_SIZE:
            pieces.append(part)
        else:
            pieces.extend(_split_recursive(part, rest))
    return [p for p in pieces if p.strip()]


def split_text(text: str) -> list[str]:
    """Greedily merge pieces into ~CHUNK_SIZE chunks, carrying ~CHUNK_OVERLAP chars of overlap."""
    pieces = _split_recursive(text, SEPARATORS)
    joiner = "\n\n" if "\n\n" in text else ("\n" if "\n" in text else " ")

    chunks, current, length = [], [], 0
    for piece in pieces:
        extra = len(piece) + (len(joiner) if current else 0)
        if current and length + extra > CHUNK_SIZE:
            chunks.append(joiner.join(current).strip())
            # drop pieces from the front until only the overlap remains
            while current and (length > CHUNK_OVERLAP or length + extra > CHUNK_SIZE):
                length -= len(current[0]) + (len(joiner) if len(current) > 1 else 0)
                current.pop(0)
        current.append(piece)
        length = sum(len(p) for p in current) + len(joiner) * (len(current) - 1)
    if current:
        chunks.append(joiner.join(current).strip())
    return [c for c in chunks if c]


def create_simple_chunks(documents: list[dict]) -> list[Result]:
    return [
        Result(page_content=chunk, metadata={"source": d["source"], "doc_type": d["doc_type"]})
        for d in documents
        for chunk in split_text(d["text"])
    ]


# --------------------------------------------------------------------------
# Step 3: embed + store in pgvector
# --------------------------------------------------------------------------
async def create_embeddings(chunks: list[Result], conn: asyncpg.Connection, paid: bool) -> None:
    profile = get_llm_profile(paid)
    texts = [chunk.page_content for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = await embed(texts, paid)
    records = [(t, "[" + ",".join(map(str, v)) + "]") for t, v in zip(texts, vectors)]

    async with conn.transaction():
        await conn.execute(f"TRUNCATE {profile.table_name} RESTART IDENTITY;")
        await conn.executemany(
            f"INSERT INTO {profile.table_name} (content, embedding) VALUES ($1, $2::vector)", records
        )
    print(f"Inserted {len(records)} chunks into {profile.table_name}")


# --------------------------------------------------------------------------
async def ingest_companies(pool: asyncpg.Pool, paid: bool, chunking: str) -> None:
    profile = get_llm_profile(paid)
    documents = fetch_documents()
    if chunking == "llm":
        chunks = await create_llm_chunks(documents, paid)
    else:
        chunks = create_simple_chunks(documents)
    print(f"Created {len(chunks)} chunks ({chunking} chunking)")

    async with pool.acquire() as conn:
        await create_embeddings(chunks, conn, paid)
    print(f"Ingestion complete → {profile.table_name}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=["paid", "free"], default="free")
    parser.add_argument("--chunking", choices=["llm", "simple"], default="llm",
                        help="llm = week5/day5 LLM chunking (default); simple = 1000/200 character splitter")
    args = parser.parse_args()

    settings = get_settings()
    # asyncpg does not accept the SQLAlchemy "+psycopg" scheme.
    dsn = settings.database_url_host.replace("postgresql+psycopg://", "postgresql://", 1)
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2)
    try:
        await ingest_companies(pool, paid=args.models == "paid", chunking=args.chunking)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())


# uv run python -m app.rag.ingest-no-langchain --models free