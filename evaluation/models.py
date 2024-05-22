import os
from typing import List
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

# ref: https://docs.ragas.io/en/stable/howtos/customisations/azure-openai.html
# ref: https://docs.confident-ai.com/docs/metrics-introduction#azure-openai-example


# request/response models
class EvaluationPayload(BaseModel):
    input: str
    actual_output: str
    expected_output: str | None = None
    context: List[str] | None = None
    retrieval_context: List[str]


class EvaluationRequest(BaseModel):
    dataset: List[EvaluationPayload]
    # the amount of iterations for each document/payload entry
    # since some metrics are LLM based, these iterations could be benefitial
    iterations_per_entry: int
    description: str
    run_type: str  # e.g. formula, knowledge, workflow, etc.


class EvaluationResult(BaseModel):
    evaluation_batch_number: int
    evaluation_id: str
    iteration_ids: List[str]


class EvaluationResponse(BaseModel):
    run_id: str
    evaluation_ids: List[EvaluationResult]


# AI models
class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"


# load environment variables
_api_key = os.getenv("AZURE_OPEN_AI_API_KEY")
_api_version = os.getenv("AZURE_OPEN_AI_API_VERSION")
_endpoint = os.getenv("AZURE_OPEN_AI_ENDPOINT")
_primary_llm = os.getenv("AZURE_OPEN_AI_MAIN_MODEL")
_secondary_llm = os.getenv("AZURE_OPEN_AI_SECONDARY_MODEL")
_embedding_model = os.getenv("AZURE_OPEN_AI_EMBEDDING_MODEL")

azure_model = AzureChatOpenAI(
    openai_api_version=_api_version,
    azure_endpoint=_endpoint,
    azure_deployment=_secondary_llm,
    model=_secondary_llm,
    validate_base_url=False,
    api_key=_api_key,
)

azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version=_api_version,
    azure_endpoint=_endpoint,
    azure_deployment=_embedding_model,
    model=_embedding_model,
    api_key=_api_key,
)

azure_openai = AzureOpenAI(model=azure_model)
