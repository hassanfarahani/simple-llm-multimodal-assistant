
from pydantic import BaseModel, Field
from typing import List

class Message(BaseModel):
    role: str
    content: str

class MessageCreate(BaseModel):
    content: str
    history: List[Message] = Field(default_factory=list) 

class CategoryScore(BaseModel):
    category: str
    value: float


class RetrievalEvalSummary(BaseModel):
    profile: str
    count: int
    failed: int
    mrr: float
    ndcg: float
    keyword_coverage: float
    by_category: list[CategoryScore]


class AnswerEvalSummary(BaseModel):
    profile: str
    count: int
    failed: int
    accuracy: float
    completeness: float
    relevance: float
    by_category: list[CategoryScore]
