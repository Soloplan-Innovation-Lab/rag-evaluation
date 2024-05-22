import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from deepeval.models.base_model import DeepEvalBaseLLM

# ref: https://docs.ragas.io/en/stable/howtos/customisations/azure-openai.html
# ref: https://docs.confident-ai.com/docs/metrics-introduction#azure-openai-example


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

critic_llm = AzureChatOpenAI(
    openai_api_version=_api_version,
    azure_endpoint=_endpoint,
    azure_deployment=_primary_llm,
    model=_primary_llm,
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
