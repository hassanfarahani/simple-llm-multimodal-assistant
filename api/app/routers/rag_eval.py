import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query

from app.db import get_pool
from app.evaluation.eval import Pipeline, run_answer_evaluation, run_retrieval_evaluation
from app.schemas import AnswerEvalSummary, RetrievalEvalSummary
from app.services.llm_provider import resolve_paid

router = APIRouter(prefix="/rag-eval", tags=["rag-eval"])

# ?pipeline=simple  -> rag_service.py (default)
# ?pipeline=agentic -> agentic_rag_service.py


@router.post("/retrieval", response_model=RetrievalEvalSummary)
async def retrieval_eval(paid: bool | None = None, limit: int | None = Query(None, ge=1),
                         pipeline: Pipeline = "simple", pool: asyncpg.Pool = Depends(get_pool)):
    try:
        return await run_retrieval_evaluation(resolve_paid(paid), limit, pipeline, pool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer", response_model=AnswerEvalSummary)
async def answer_eval(paid: bool | None = None, limit: int | None = Query(None, ge=1),
                      pipeline: Pipeline = "simple", pool: asyncpg.Pool = Depends(get_pool)):
    try:
        return await run_answer_evaluation(resolve_paid(paid), limit, pipeline, pool)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
