from functools import lru_cache
from typing import List
from langchain_core.language_models import LanguageModelInput
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import BaseMessage
from internal_shared.models.ai import (
    AvailableModels,
    available_models_to_model_metadata,
)

__all__ = [
    "invoke_prompt",
    "invoke_prompt_async",
    "invoke_streaming_prompt_async",
    "embed_text",
    "embed_text_async",
]


@lru_cache(maxsize=10)
def _get_embedding_client(model: AvailableModels) -> AzureOpenAIEmbeddings:
    actual_model = available_models_to_model_metadata(model)
    return AzureOpenAIEmbeddings(
        azure_endpoint=actual_model.endpoint,
        model=actual_model.model_name,
        azure_deployment=actual_model.deployment_name,
        api_key=actual_model.api_key,
        api_version=actual_model.api_version,
    )


@lru_cache(maxsize=10)
def _get_client(model: AvailableModels) -> BaseChatOpenAI:
    actual_model = available_models_to_model_metadata(model)
    return AzureChatOpenAI(
        azure_endpoint=actual_model.endpoint,
        model=actual_model.model_name,
        azure_deployment=actual_model.deployment_name,
        api_key=actual_model.api_key,
        api_version=actual_model.api_version,
    )


def invoke_prompt(
    prompt: LanguageModelInput, model: AvailableModels = AvailableModels.GPT_4O
) -> BaseMessage:
    client = _get_client(model)
    return client.invoke(prompt)


async def invoke_prompt_async(
    prompt: LanguageModelInput, model: AvailableModels = AvailableModels.GPT_4O
) -> BaseMessage:
    client = _get_client(model)
    return await client.ainvoke(prompt)


async def invoke_streaming_prompt_async(
    prompt: LanguageModelInput, model: AvailableModels = AvailableModels.GPT_4O
):
    client = _get_client(model)
    async for chunk in client.astream(prompt):
        yield chunk


def embed_text(
    text: str,
    model: AvailableModels = AvailableModels.EMBEDDING_3_LARGE,
) -> List[float]:
    _embedding_client = _get_embedding_client(model)
    return _embedding_client.embed_query(text)


async def embed_text_async(
    text: str,
    model: AvailableModels = AvailableModels.EMBEDDING_3_LARGE,
) -> List[float]:
    embedding_client = _get_embedding_client(model)
    return await embedding_client.aembed_query(text)
