import os
import glob
from pathlib import Path
import asyncio
import asyncpg
from dotenv import load_dotenv
from app.config import get_settings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
import argparse
from app.services.model_provider import get_profile

load_dotenv()

# MODEL = "gpt-4.1-nano"
KNOWLEDGE_BASE = str(Path(__file__).parent.parent.parent / "knowledge-base")  # /home/hassan/courses/LLM/projects/llm-chatbot/api/knowledge-base
# embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2") # 413 dimensions
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # 3072 dimensions



async def ingest_companies(pool: asyncpg.Pool, paid: bool = True) -> None:
    profile = get_profile(paid)
    async with pool.acquire() as conn:
        chunks = create_chunks(fetch_documents())
        await create_embeddings(chunks, conn, profile.embeddings, profile.table_name)
        print(f"Ingestion complete → {profile.table_name}")


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


# https://chat.deepseek.com/a/chat/s/f7663037-497b-4e63-85bf-2e0547b65798 (rewrite the function using posrgresql)
async def create_embeddingsO(chunks, conn: asyncpg.Pool, embeddings: OpenAIEmbeddings):
    # 1. Extract text content from Document objects
    texts = [chunk.page_content for chunk in chunks]

    # 2. Generate embeddings (offload sync call to thread pool)
    embeddings_vectors = await asyncio.to_thread(embeddings.embed_documents, texts)

    # 3. Prepare data for batch insertion
    # Format embedding as a string like '[0.1, 0.2, ...]' for pgvector
    records = []
    for text, vector in zip(texts, embeddings_vectors):
        vector_str = '[' + ','.join(map(str, vector)) + ']'
        records.append((text, vector_str))

    # 4. Insert into the database using a single batch operation
    # async with pool.acquire() as conn:
        # Clear old data if needed
    await conn.execute("TRUNCATE documents RESTART IDENTITY;")

    # Insert all chunks in one go
    await conn.executemany(
        "INSERT INTO documents (content, embedding) VALUES ($1, $2::vector)",
        records
    )
    print(f"Successfully inserted {len(records)} chunks into the vector store")

async def create_embeddings(chunks, conn, embeddings, table_name: str):
    texts = [chunk.page_content for chunk in chunks]
    vectors = await asyncio.to_thread(embeddings.embed_documents, texts)
    records = [(t, "[" + ",".join(map(str, v)) + "]") for t, v in zip(texts, vectors)]

    await conn.execute(f"TRUNCATE {table_name} RESTART IDENTITY;")
    await conn.executemany(
        f"INSERT INTO {table_name} (content, embedding) VALUES ($1, $2::vector)", records
    )
    print(f"Inserted {len(records)} chunks into {table_name}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=["paid", "free"], default="free")
    args = parser.parse_args()
    modes = {"paid": [True], "free": [False]}[args.models]

    settings = get_settings()
    # asyncpg does not accept the SQLAlchemy "+psycopg" scheme.
    dsn = settings.database_url_host.replace("postgresql+psycopg://", "postgresql://", 1)
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
    try:
        for paid in modes:
            await ingest_companies(pool, paid)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())

# uv run python -m app.rag.ingest --models free
