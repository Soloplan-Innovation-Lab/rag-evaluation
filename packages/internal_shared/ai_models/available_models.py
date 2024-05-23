from dataclasses import dataclass, field
from internal_shared.utils.helper_functions import get_env_variable


@dataclass(frozen=True)
class ModelMetadata:
    """Metadata for a model."""

    model_name: str
    deployment_name: str
    api_key: str
    endpoint: str
    api_version: str = field(default=get_env_variable("AZURE_OPENAI_API_VERSION"))

    @staticmethod
    def create(model_name: str, deployment_name: str, is_us_server: bool = True):
        """
        Factory method to create a ModelMetadata object.
        """
        api_key = get_env_variable(
            "AZURE_OPENAI_US_API_KEY" if is_us_server else "AZURE_OPENAI_SE_API_KEY"
        )
        endpoint = get_env_variable(
            "AZURE_OPENAI_US_ENDPOINT" if is_us_server else "AZURE_OPENAI_SE_ENDPOINT"
        )
        return ModelMetadata(
            model_name=model_name,
            deployment_name=deployment_name,
            api_key=api_key,
            endpoint=endpoint,
        )


# LLMs
GPT_35_TURBO = ModelMetadata.create(
    model_name="gpt-35-turbo", deployment_name="gpt-35-turbo"
)
GPT_4 = ModelMetadata.create(model_name="gpt-4", deployment_name="gpt-4")
GPT_4_32K = ModelMetadata.create(model_name="gpt-4-32k", deployment_name="gpt-4-32k")
GPT_4_TURBO = ModelMetadata.create(
    model_name="gpt-4", deployment_name="gpt-4-turbo", is_us_server=False
)
GPT_4O = ModelMetadata.create(model_name="gpt-4o", deployment_name="gpt-4o")

# EMBEDDING MODELS
EMBEDDING_3_LARGE = ModelMetadata.create(
    model_name="text-embedding-3-large", deployment_name="text-embedding-3-large"
)
EMBEDDING_3_SMALL = ModelMetadata.create(
    model_name="text-embedding-3-small", deployment_name="text-embedding-3-small"
)
EMBEDDING_2 = ModelMetadata.create(
    model_name="text-embedding-2", deployment_name="text-embedding-2"
)
