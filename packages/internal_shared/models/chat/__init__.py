from enum import Enum
from bson import ObjectId
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Annotated
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import BeforeValidator
from internal_shared.models.ai import AvailableModels, AvailableEmbeddingModels

DEFAULT_MODEL = AvailableModels.GPT_4O

# ref: https://github.com/mongodb-developer/mongodb-with-fastapi/blob/master/app.py
# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]


class PreRetrievalType(str, Enum):
    DEFAULT = "default"
    QUERY_EXPANSION = "query_expansion"
    REWRITE_RETRIEVE_READ = "rewrite_retrieve_read"
    STEP_BACK_PROMPTING = "step_back_prompting"
    HYDE = "hyde"
    REPHRASE_AND_RESPOND = "rephrase_and_respond"


class RetrievalType(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"


class PostRetrievalType(str, Enum):
    DEFAULT = "default"


class SearchResult(BaseModel):
    name: str
    summary: str
    content: str
    score: float
    type: RetrievalType


class TokenUsage(BaseModel):
    """Token usage information for a response."""

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class RetrieverConfig(BaseModel):
    """Configuration for a retriever."""

    retriever_name: str
    retriever_type: RetrievalType
    index_name: str
    embedding_model: AvailableEmbeddingModels = (
        AvailableEmbeddingModels.EMBEDDING_3_LARGE
    )

    def to_dto(self):
        return RetrieverConfigDTO(**self.model_dump(by_alias=True), id=None)

    def to_dto_dict(self):
        dto = self.to_dto()
        return dto.model_dump(by_alias=True, exclude=["id"])


class RetrieverConfigDTO(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    retriever_name: str
    retriever_type: RetrievalType
    index_name: str
    embedding_model: AvailableEmbeddingModels = (
        AvailableEmbeddingModels.EMBEDDING_3_LARGE
    )

    def to_model(self):
        return RetrieverConfig(**self.model_dump(by_alias=True, exclude=["id"]))

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


class RetrievalConfig(BaseModel):
    """Configuration for a retrieval step."""

    retriever: RetrieverConfig
    context_key: str = "context"
    retrieval_type: RetrievalType
    pre_retrieval_type: PreRetrievalType
    post_retrieval_type: PostRetrievalType
    top_k: int = 5
    threshold: float = 0.5


class ResponseBehavior(BaseModel):
    # experimental feature

    # short, medium, long
    # formats like JSON, YAML, XML
    # formal, academic, casual, comercial

    pass


class RetrievalStepResult(BaseModel):
    """A single step in the retrieval pipeline."""

    config: RetrievalConfig
    initial_query: str
    pre_retrieval: str
    pre_retrieval_duration: float
    retrieval: List[SearchResult]
    retrieval_duration: float
    post_retrieval: List[SearchResult]
    post_retrieval_duration: float


class PromptTemplate(BaseModel):
    """A prompt template."""

    name: str
    template: str

    def to_dto(self):
        return PromptTemplateDTO(**self.model_dump(by_alias=True), id=None)

    def to_dto_dict(self):
        dto = self.to_dto()
        return dto.model_dump(by_alias=True, exclude=["id"])


class PromptTemplateDTO(BaseModel):
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    name: str
    template: str

    def to_model(self):
        return PromptTemplate(**self.model_dump(by_alias=True, exclude=["id"]))

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


class ChatRequest(BaseModel):
    """
    Request type for the chat API.
    """

    query: str
    retrieval_behaviour: List[RetrievalConfig]
    model: AvailableModels = DEFAULT_MODEL
    prompt_template: PromptTemplate | None = None
    history: List[Tuple[str, str]] = []


class ChatResponse(BaseModel):
    """
    Response type for the chat API.

    Contains the response, the documents retrieved, and metadata.
    """

    chat_session_id: str

    # most interesting fields
    response: str
    documents: List[str] = []
    # metadata
    request: str
    model: AvailableModels = DEFAULT_MODEL
    response_duration: float = 0.0
    token_usage: TokenUsage | None = None
    steps: List[RetrievalStepResult] | None = None

    def to_dto(self):
        return ChatResponseDTO(**self.model_dump(by_alias=True), id=None)

    def to_dto_dict(self):
        dto = self.to_dto()
        return dto.model_dump(by_alias=True, exclude=["id"])


class ChatResponseDTO(BaseModel):
    # mongodb id field
    id: Optional[PyObjectId] = Field(alias="_id", default=None)

    # most interesting fields
    response: str
    documents: List[str] = []
    # metadata
    request: str
    chat_session_id: Optional[str] = None
    model: AvailableModels = DEFAULT_MODEL
    response_duration: float = 0.0
    token_usage: TokenUsage | None = None
    steps: List[RetrievalStepResult] | None = None

    def to_model(self):
        return ChatResponse(**self.model_dump(by_alias=True, exclude=["id"]))

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str},
    )


class ChatResponseChunk(BaseModel):
    chunk: str
    metadata: Dict[str, Any] = {}


__all__ = [
    "PreRetrievalType",
    "RetrievalType",
    "PostRetrievalType",
    "SearchResult",
    "TokenUsage",
    "RetrievalConfig",
    "ResponseBehavior",
    "RetrievalStepResult",
    "PromptTemplate",
    "PromptTemplateDTO",
    "ChatRequest",
    "ChatResponse",
    "ChatResponseDTO",
    "ChatResponseChunk",
]
