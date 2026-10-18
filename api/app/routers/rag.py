from fastapi import APIRouter
from app.schemas import MessageCreate
from fastapi import FastAPI, HTTPException
from typing import List
from app.clients import openai
from app.services.rag_service import process_message


router = APIRouter(prefix="/rag-messages", tags=["rag-messages"])

@router.post('')
async def get_message(msg: MessageCreate):
    try:
        res = await process_message(msg)
        return {
             "response": res
            }
    except Exception as e:
        print(f'Error: {e}')
        raise HTTPException(status_code=500, detail=str(e))

def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question