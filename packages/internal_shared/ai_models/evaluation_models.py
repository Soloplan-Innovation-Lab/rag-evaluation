try:
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from langchain_openai.chat_models.base import BaseChatOpenAI
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:
    raise ImportError(
        "Necessary evaluation dependencies are not installed. Please run `pip install langchain-openai deepeval`."
    )

from internal_shared.ai_models.available_models import GPT_4O, GPT_35_TURBO, EMBEDDING_3_LARGE

# ref: https://docs.ragas.io/en/stable/howtos/customisations/azure-openai.html
# ref: https://docs.confident-ai.com/docs/metrics-introduction#azure-openai-example

# AI models
class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(self, model: BaseChatOpenAI, name: str):
        self.model = model
        self.name = name

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self) -> str:
        return self.name


azure_model = AzureChatOpenAI(
    openai_api_version=GPT_35_TURBO.api_version,
    azure_endpoint=GPT_35_TURBO.endpoint,
    azure_deployment=GPT_35_TURBO.deployment_name,
    model=GPT_35_TURBO.model_name,
    api_key=GPT_35_TURBO.api_key,
    validate_base_url=False,
)

critic_llm = AzureChatOpenAI(
    openai_api_version=GPT_4O.api_version,
    azure_endpoint=GPT_4O.endpoint,
    azure_deployment=GPT_4O.deployment_name,
    model=GPT_4O.model_name,
    api_key=GPT_4O.api_key,
    validate_base_url=False,
)

azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version=EMBEDDING_3_LARGE.api_version,
    azure_endpoint=EMBEDDING_3_LARGE.endpoint,
    azure_deployment=EMBEDDING_3_LARGE.deployment_name,
    model=EMBEDDING_3_LARGE.model_name,
    api_key=EMBEDDING_3_LARGE.api_key,
)

azure_openai = AzureOpenAI(model=azure_model, name="Azure OpenAI GPT-35 Turbo")
