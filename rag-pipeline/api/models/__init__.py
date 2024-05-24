from typing import List
from pydantic import BaseModel
from retrieval import RetrievalType, SearchResult, PreRetrievalType, PostRetrievalType
from internal_shared.ai_models.available_models import AvailableModels

DEFAULT_MODEL = AvailableModels.GPT_4O


class TokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class RetrievalConfig(BaseModel):
    retrieval_type: RetrievalType
    pre_retrieval_type: PreRetrievalType
    post_retrieval_type: PostRetrievalType
    top_k: int = 5
    threshhold: float = 0.5


class RetrievalStep(BaseModel):
    config: RetrievalConfig
    initial_query: str
    pre_retrieval: str
    pre_retrieval_duration: float
    retrieval: List[SearchResult]
    retrieval_duration: float
    post_retrieval: List[SearchResult]
    post_retrieval_duration: float


class ChatRequest(BaseModel):
    query: str
    retrieval_behaviour: List[RetrievalConfig]
    model: AvailableModels = DEFAULT_MODEL
    # add search index?
    # add prompt template (optional) -> Optional[Dict[str, str]]


class ChatResponse(BaseModel):
    # most interesting fields
    response: str
    documents: List[str] = []
    # metadata
    request: str
    model: AvailableModels = DEFAULT_MODEL
    response_duration: float = 0.0
    token_usage: TokenUsage | None = None
    steps: List[RetrievalStep] | None = None
