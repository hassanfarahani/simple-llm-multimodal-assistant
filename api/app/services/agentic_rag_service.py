from app.clients import openai
from app.schemas import Message, MessageCreate
from app.config import get_settings
from pathlib import Path
from app.services.llm_provider import get_llm_profile, chat, embed_query

import asyncpg
import litellm
from litellm import acompletion
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

wait = wait_exponential(multiplier=1, min=10, max=240)
RETRIEVAL_K = 20
FINAL_K = 10

# Same system prompt as process_message in rag_service.py
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

async def process_agentic_rag_message(payload: MessageCreate, paid: bool, pool: asyncpg.Pool):
    question = payload.content
    history = payload.history
    rerankedRetrievedContext = await afetch_context(question, history, paid, pool)
    return await answer_question(question, history, rerankedRetrievedContext, paid)

async def afetch_context(question, history, paid, pool):
    rewritten_question = await rewrite_query(question, history, paid)
    chunks1 = await fetch_context_unranked(question, paid, pool)
    chunks2 = await fetch_context_unranked(rewritten_question, paid, pool)
    chunks = merge_chunks(chunks1, chunks2)
    reranked = await rerank(question, chunks, paid)
    return reranked[:FINAL_K]

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout)),
    reraise=True,
)
async def answer_question(question: str, history: list[Message], chunks: list["Result"], paid: bool) -> str:
    """Answer the user's question with the reranked chunks as context (like process_message in rag_service.py)."""
    # 1. put the chunks into the system prompt
    context = "\n\n".join(chunk.page_content for chunk in chunks)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]

    # 2. add the conversation so far.
    #    The frontend already puts the current question at the end of history,
    #    so drop it here; otherwise the LLM would see the question twice.
    past = [msg.model_dump() for msg in history]
    if past and past[-1]["role"] == "user" and past[-1]["content"] == question:
        past = past[:-1]
    messages.extend(past)

    # 3. add the current question
    messages.append({"role": "user", "content": question})

    # 10 chunks + a long conversation can exceed the free profile's 4096-token window.
    overrides = {} if paid else {"num_ctx": 8192}
    response = await chat(messages, paid, **overrides)
    return response.choices[0].message.content

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout)),
    reraise=True,
)
async def rewrite_query(question, history, paid: bool):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    profile = get_llm_profile(paid)    # lru_cached, so cheap per call
    messages = buildMessagesByModelType(profile.name, question, history)
    response = await chat(messages, paid)
    return response.choices[0].message.content.strip()


def buildMessagesByModelType(modelName, question, history):
    if history is None:
        history = []

    history_text: list[dict] = [msg.model_dump() for msg in history]

    messages: list[dict]

    if modelName == "openai":
        message = f"""
You are in a conversation with a user.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history_text}

And this is the user's current question:
{question}

Since the conversation is contextual, understand the meaning of the user question
and add details based on the history.
Condense everything into a single contextually rich, VERY short and specific
question, most likely to surface relevant content.

EXAMPLE:
user: Who is the founder? -> Query: Who is the founder?
assistant: The founder is FooBar
user: What role does he cover? -> Query: What role does FooBar cover?

IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""

        messages = [
            {"role": "system", "content": message}
        ]

    else:
        system = """You rewrite the user's latest question into a short, specific
search query for a Knowledge Base.
Use the conversation history only to resolve references (e.g. "he", "it", "the company").
Since the conversation is contextual, understand the meaning of the user question
and add details based on the history.
Condense everything into a single contextually rich, VERY short and specific
question, most likely to surface relevant content.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.

Example (illustrative only, not part of this conversation):
History: user: Who is the founder? / assistant: The founder is FooBar
Question: What role does he cover?
Query: What role does FooBar cover?"""

        user = f"""Conversation history: {history_text}
Current question: {question}
Query:"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]

    return messages

class Result(BaseModel):
    page_content: str
    metadata: dict

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


async def fetch_context_unranked(question: str, paid: bool, pool: asyncpg.Pool) -> list[Result]:
    """Return the RETRIEVAL_K chunks closest to the question, closest first."""
    # 1. question -> vector (must be the same embedding model the table was built with)
    query = await embed_query(question, paid)
    table = get_llm_profile(paid).table_name   # "documents" (3072 dims) or "documents_free" (384 dims)

    query_vector = "[" + ",".join(map(str, query)) + "]"  # "[0.12,-0.45,0.78]" as The string representation is understood by pgvector

    # embedding <=> $1::vector -->	Calculates the distance between a stored embedding and the question's embedding
    rows = await pool.fetch(
        f"""
        SELECT id, content, embedding <=> $1::vector AS distance
        FROM {table}
        ORDER BY distance
        LIMIT $2
        """,
        query_vector,
        RETRIEVAL_K,
    )

    # 3. rows -> chunks (convert rows into list of Result object)
    chunks = []
    for row in rows:
        chunks.append(Result(page_content=row["content"],
                             metadata={"id": row["id"], "distance": row["distance"]}))
    return chunks

def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((litellm.RateLimitError, litellm.APIConnectionError, litellm.Timeout)),
    reraise=True,
)
async def rerank(question: str, chunks: list[Result], paid: bool) -> list[Result]:
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Up to 40 merged chunks don't fit in the free profile's default 4096-token window,
    # and Ollama would silently cut off the start of the prompt, so give it more room.
    overrides = {} if paid else {"num_ctx": 8192}
    response = await chat(messages, paid, response_format=RankOrder, **overrides)
    reply = response.choices[0].message.content  # { "order": [7, 1, 20, 10, 19, 25, ..., 27, 28, 29] }
    order = RankOrder.model_validate_json(reply).order

    # The LLM can return ids that don't exist, repeat ids, or leave some out.
    # Keep only valid ids (1..len(chunks)) once each, then add any it forgot in their original order.
    valid = list(dict.fromkeys(i for i in order if 1 <= i <= len(chunks)))
    missing = [i for i in range(1, len(chunks) + 1) if i not in valid]
    return [chunks[i - 1] for i in valid + missing]