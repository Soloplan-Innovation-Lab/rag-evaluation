import json
from typing import List
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from deepeval.models.base_model import DeepEvalBaseLLM
from pydantic import BaseModel

# ref: https://docs.ragas.io/en/stable/howtos/customisations/azure-openai.html
# ref: https://docs.confident-ai.com/docs/metrics-introduction#azure-openai-example

# parse secrets.json file to create azure_configs
with open("secrets.json") as f:
    _azure_configs = json.load(f)


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


azure_model = AzureChatOpenAI(
    openai_api_version=_azure_configs["api_version"],
    azure_endpoint=_azure_configs["base_url"],
    azure_deployment=_azure_configs["model_deployment"],
    model=_azure_configs["model_name"],
    validate_base_url=False,
    api_key=_azure_configs["api_key"],
)

critic_llm = AzureChatOpenAI(
    openai_api_version=_azure_configs["api_version"],
    azure_endpoint=_azure_configs["base_url"],
    azure_deployment=_azure_configs["critic_model_deployment"],
    model=_azure_configs["critic_model_name"],
    validate_base_url=False,
    api_key=_azure_configs["api_key"],
)

azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version=_azure_configs["api_version"],
    azure_endpoint=_azure_configs["base_url"],
    azure_deployment=_azure_configs["embedding_deployment"],
    model=_azure_configs["embedding_name"],
    api_key=_azure_configs["api_key"],
)

azure_openai = AzureOpenAI(model=azure_model)
