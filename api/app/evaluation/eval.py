import asyncio
import math
from collections import defaultdict

from pydantic import BaseModel, Field

from app.evaluation.test import TestQuestion, load_tests
from app.services.model_provider import get_profile
from app.services.rag_service import afetch_context, answer_question


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


# ---- metrics (same as week5 evaluation/eval.py) ----


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]
    dcg = calculate_dcg(relevances, k)
    idcg = calculate_dcg(sorted(relevances, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


# ---- per-test evaluation ----


async def evaluate_retrieval(test: TestQuestion, paid: bool, k: int = 10) -> RetrievalEval:
    retrieved_docs = await afetch_context(test.question, paid)
    mrr_scores = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    ndcg_scores = [calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]
    found = sum(1 for s in mrr_scores if s > 0)
    total = len(test.keywords)
    return RetrievalEval(
        mrr=_mean(mrr_scores),
        ndcg=_mean(ndcg_scores),
        keywords_found=found,
        total_keywords=total,
        keyword_coverage=(found / total * 100) if total else 0.0,
    )


JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer "
    "by comparing it to the reference answer. Only give 5/5 scores for perfect answers."
)


def build_judge_messages(test: TestQuestion, generated_answer: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]


async def evaluate_answer(test: TestQuestion, paid: bool) -> AnswerEval:
    generated_answer, _ = await answer_question(test.question, paid)
    judge = get_profile(paid).judge_llm.with_structured_output(AnswerEval)
    return await judge.ainvoke(build_judge_messages(test, generated_answer))


# ---- aggregation (replaces the loops in evaluator.py) ----


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _by_category(tests: list[TestQuestion], scores: list[float]) -> list[dict]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for test, score in zip(tests, scores):
        buckets[test.category].append(score)
    return [{"category": c, "value": _mean(v)} for c, v in buckets.items()]


async def _run_all(tests, fn, max_concurrency: int):
    sem = asyncio.Semaphore(max_concurrency)

    async def guarded(test):
        async with sem:
            try:
                return await fn(test)
            except Exception as e:  # small local models sometimes return bad JSON
                print(f"[eval] failed: {test.question!r}: {e}")
                return None

    return await asyncio.gather(*(guarded(t) for t in tests))


def _split_ok(tests, results):
    ok_tests = [t for t, r in zip(tests, results) if r is not None]
    ok = [r for r in results if r is not None]
    if not ok:
        raise RuntimeError("All tests failed – check the API logs")
    return ok_tests, ok


async def run_retrieval_evaluation(paid: bool, limit: int | None = None) -> dict:
    profile = get_profile(paid)
    tests = load_tests()[:limit]  # [:None] == all tests
    results = await _run_all(tests, lambda t: evaluate_retrieval(t, paid), profile.max_concurrency)
    ok_tests, ok = _split_ok(tests, results)
    return {
        "profile": profile.name,
        "count": len(ok),
        "failed": len(tests) - len(ok),
        "mrr": _mean([r.mrr for r in ok]),
        "ndcg": _mean([r.ndcg for r in ok]),
        "keyword_coverage": _mean([r.keyword_coverage for r in ok]),
        "by_category": _by_category(ok_tests, [r.mrr for r in ok]),
    }


async def run_answer_evaluation(paid: bool, limit: int | None = None) -> dict:
    profile = get_profile(paid)
    tests = load_tests()[:limit]
    results = await _run_all(tests, lambda t: evaluate_answer(t, paid), profile.max_concurrency)
    ok_tests, ok = _split_ok(tests, results)
    return {
        "profile": profile.name,
        "count": len(ok),
        "failed": len(tests) - len(ok),
        "accuracy": _mean([r.accuracy for r in ok]),
        "completeness": _mean([r.completeness for r in ok]),
        "relevance": _mean([r.relevance for r in ok]),
        "by_category": _by_category(ok_tests, [r.accuracy for r in ok]),
    }
