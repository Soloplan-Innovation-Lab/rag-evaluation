from dataclasses import dataclass, field
from enum import Enum
from internal_shared.utils.helper_functions import get_env_variable


class AvailableModels(str, Enum):
    """
    Enum for available models.
    """

    GPT_35_TURBO = "gpt-35-turbo"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    EMBEDDING_3_LARGE = "text-embedding-3-large"
    EMBEDDING_3_SMALL = "text-embedding-3-small"
    EMBEDDING_2 = "text-embedding-ada-002"

def get_embedding_models():
    """
    Get the available embedding models.
    """
    return [model for model in AvailableModels if "embedding" in model.value]


def get_chat_models():
    """
    Get the available chat models.
    """
    return [model for model in AvailableModels if "embedding" not in model.value]


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
    model_name="text-embedding-ada-002", deployment_name="text-embedding-ada-002"
)


__model_map = {
    AvailableModels.GPT_35_TURBO: GPT_35_TURBO,
    AvailableModels.GPT_4: GPT_4,
    AvailableModels.GPT_4_32K: GPT_4_32K,
    AvailableModels.GPT_4_TURBO: GPT_4_TURBO,
    AvailableModels.GPT_4O: GPT_4O,
    AvailableModels.EMBEDDING_3_LARGE: EMBEDDING_3_LARGE,
    AvailableModels.EMBEDDING_3_SMALL: EMBEDDING_3_SMALL,
    AvailableModels.EMBEDDING_2: EMBEDDING_2,
}


def available_models_to_model_metadata(model: AvailableModels) -> ModelMetadata:
    """
    Get the model metadata for a given available model.
    """
    try:
        return __model_map[model]
    except KeyError:
        raise ValueError(f"Model {model} is not supported")
