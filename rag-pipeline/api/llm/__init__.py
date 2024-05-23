from typing import Dict, List
from langchain_core.language_models import LanguageModelInput
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai import AzureOpenAI
from internal_shared.ai_models.available_models import (
    AvailableModels,
    available_models_to_model_metadata,
    EMBEDDING_3_LARGE,
)

_client_cache: Dict[AvailableModels, BaseChatOpenAI] = {}
_embedding_client = AzureOpenAI(
    api_key=EMBEDDING_3_LARGE.api_key,
    api_version=EMBEDDING_3_LARGE.api_version,
    azure_endpoint=EMBEDDING_3_LARGE.endpoint,
)


def invoke_prompt(
    prompt: LanguageModelInput, model: AvailableModels = AvailableModels.GPT_4O
):
    if model not in _client_cache:
        actual_model = available_models_to_model_metadata(model)
        _client_cache[model] = AzureChatOpenAI(
            azure_endpoint=actual_model.endpoint,
            model=actual_model.model_name,
            azure_deployment=actual_model.deployment_name,
            api_key=actual_model.api_key,
            api_version=actual_model.api_version,
        )
    return _client_cache[model].invoke(prompt)


def embed_text(text: str) -> List[float]:
    embedding_result = _embedding_client.embeddings.create(
        input=text, model=EMBEDDING_3_LARGE.model_name
    )
    return embedding_result.data[0].embedding
