from fastapi import APIRouter, HTTPException, Query

from app.config import get_settings
from app.evaluation.eval import run_answer_evaluation, run_retrieval_evaluation
from app.schemas import AnswerEvalSummary, RetrievalEvalSummary

router = APIRouter(prefix="/rag-eval", tags=["rag-eval"])
settings = get_settings()


def _use_paid(paid: bool | None) -> bool:
    """No query param → use USE_PAID_MODELS from .env."""
    return settings.use_paid_models if paid is None else paid


@router.post("/retrieval", response_model=RetrievalEvalSummary)
async def retrieval_eval(paid: bool | None = None, limit: int | None = Query(None, ge=1)):
    try:
        return await run_retrieval_evaluation(_use_paid(paid), limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer", response_model=AnswerEvalSummary)
async def answer_eval(paid: bool | None = None, limit: int | None = Query(None, ge=1)):
    try:
        return await run_answer_evaluation(_use_paid(paid), limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))