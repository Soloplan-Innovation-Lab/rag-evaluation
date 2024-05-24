from typing import List
from pydantic import BaseModel
from retrieval import RetrievalType, SearchResult, PreRetrievalType, PostRetrievalType
from internal_shared.ai_models.available_models import AvailableModels

DEFAULT_MODEL = AvailableModels.GPT_4O


class RetrievalConfig(BaseModel):
    retrieval_type: RetrievalType
    pre_retrieval_type: PreRetrievalType
    post_retrieval_type: PostRetrievalType
    top_k: int = 5
    threshhold: float = 0.5


class RetrievalStep(BaseModel):
    config: RetrievalConfig
    query: str
    pre_retrieval: str
    retrieval: List[SearchResult]
    post_retrieval: List[SearchResult]


class ChatRequest(BaseModel):
    query: str
    retrieval_behaviour: List[RetrievalConfig]
    model: AvailableModels = DEFAULT_MODEL


class ChatResponse(BaseModel):
    response: str
    request: str
    documents: List[str] | None = None
    model: AvailableModels = DEFAULT_MODEL
    llm_duration: float = 0.0
    retrieval_duration: float = 0.0
    token_usage: float = 0.0
    steps: List[RetrievalStep] | None = None
