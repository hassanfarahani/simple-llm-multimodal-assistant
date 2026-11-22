import asyncpg
from fastapi import APIRouter, Depends, HTTPException

from app.db import get_pool
from app.schemas import MessageCreate
from app.services.agentic_rag_service import process_agentic_rag_message
from app.services.llm_provider import resolve_paid


router = APIRouter(prefix="/agentic-rag-messages", tags=["agentic-rag-messages"])

@router.post('')
async def get_message(msg: MessageCreate, paid: bool | None = None, pool: asyncpg.Pool = Depends(get_pool)):
    """?paid=true → OpenAI, ?paid=false → Ollama, omitted → USE_PAID_MODELS from .env."""
    try:
        res = await process_agentic_rag_message(msg, resolve_paid(paid), pool)
        return {"response": res}
    except Exception as e:
        print(f'Error: {e}')
        raise HTTPException(status_code=500, detail=str(e))
