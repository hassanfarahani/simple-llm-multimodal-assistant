from app.clients import openai
from app.schemas import Message, MessageCreate
from app.config import get_settings
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGEngine, PGVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages

MODEL = "gpt-4.1-nano"
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

settings = get_settings()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

engine = PGEngine.from_connection_string(
    url=settings.database_url
)

# Module-level holder; populated in lifespan startup
vectorstore: PGVectorStore | None = None
retriever = None

async def init_vectorstore() -> None:
    """Call once during app startup."""
    global vectorstore, retriever
    vectorstore = await PGVectorStore.create(
        engine=engine,
        table_name="documents",
        schema_name="public",
        id_column="id",
        content_column="content",
        embedding_column="embedding",
        embedding_service=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})


llm = ChatOpenAI(
    temperature=0,
    model=MODEL,
)


async def process_message(payload: MessageCreate):
    question = payload.content
    combined = combined_question(payload.history)
    
    # fetch context based on all the message sthe user has said sofar
    # this is poorer if user suddenly change the topic ---> reasons why ---> look at the end 
    # docs = fetch_context(combined) 

    # fetch context based on the most recent message
    docs = fetch_context(payload.content) 
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    # Starts the message list with the system prompt (which now contains the RAG context).
    # messages == [SystemMessage(content="You are a knowledgeable...\n\nContext:\n...")]
    messages = [SystemMessage(content=system_prompt)] 

    # payload.history: [Message(role='assistant', content='Hello! 👋 How can I help you today?'), Message(role='user', content='who is Avery?')]
    # convert to dic: history == [{"role": "assistant", "content": "Hello! 👋 How can I help you today?"}, ...]
    history: list[dict] = [msg.model_dump() for msg in payload.history] # converts each Message object into a plain dict

    # convert_to_messages from langchain_core translates role-tagged dicts into LangChain message objects.
    # messages == [
    #     SystemMessage(content="You are..."),
    #     HumanMessage(content="Tell me about Insurellm's health plan"),
    #     AIMessage(content="Insurellm's health plan covers..."),
    # ]
    messages.extend(convert_to_messages(history))

    # Appends the current user message.
    # messages == [
    #     SystemMessage(...),
    #     HumanMessage(content="Tell me about Insurellm's health plan"),
    #     AIMessage(content="Insurellm's health plan covers..."),
    #     HumanMessage(content="What about the premium version?"),  # ← new
    # ]
    messages.append(HumanMessage(content=question))

    # Sends the full conversation to ChatOpenAI (gpt-4.1-nano, temperature=0) and gets back an AIMessage
    response = llm.invoke(messages)
    return response.content

def combined_question(history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    # Convert each Message to a dict 
    # .model_dump() (Pydantic v2) or .dict() (Pydantic v1) turns it into a dict:
    prior = "\n".join(
        m["content"] for m in (msg.model_dump() for msg in history)
        if m["role"] == "user"
    )
    return prior

# https://chatgpt.com/c/6acd71dd-cc0c-83e8-84a4-e96c9914bb78 (using postgres vector db)
# def fetch_context(question: str) -> list[Document]:
def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents from PostgreSQL + pgvector.
    """
    return retriever.invoke(question)



# problems with this way of implementing RAG:
# 1. who won the prectigious IIOTY award?
# AI answer: Maxine, a Senior Data Engineer at Insurellm, was recognized as the Insurellm Innovator of the Year (IIOTY) in 2023. She received this prestigious award for her contributions and innovative work in the company.

# as you see, the response says Maxine but it does not say Maxine Thompson. It does not realize her full name. the reason is because in the chunk of related context that it found, there is no full name (knowledge-base/employees/Maxine thompson) because the fullname is at the top of the HR documents (not in the middle). this is an isseu with RAG with chunking that the chunk might answer the question, but it might not contain some crucial context from elsewhere in the document, meaning that you get back an okay but not perfect answe




# why chaging topic in a RAG app that uses all user messages so far to get the relevant context lead to a very poor response
# because we alreday restricted the number context chunks for a user message (RETRIEVAL_K=5) so when user after a few questions change the topic and ask a new question about a new topic, the retrieved context for the most recent question are e.g. 90% has to do with the users' previous questions not the most recent one so it is possible that the retrieved context might not contain the information related to the last question. 

# so RAG is quite easy to get off track.