from functools import lru_cache
from typing import List
from langchain_core.language_models import LanguageModelInput
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import BaseMessage
from internal_shared.ai_models.available_models import (
    AvailableModels,
    available_models_to_model_metadata,
    EMBEDDING_3_LARGE,
)

_embedding_client = AzureOpenAIEmbeddings(
    api_key=EMBEDDING_3_LARGE.api_key,
    api_version=EMBEDDING_3_LARGE.api_version,
    azure_endpoint=EMBEDDING_3_LARGE.endpoint,
    azure_deployment=EMBEDDING_3_LARGE.deployment_name,
    model=EMBEDDING_3_LARGE.model_name,
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


def embed_text(text: str) -> List[float]:
    return _embedding_client.embed_query(text)


async def embed_text_async(text: str) -> List[float]:
    return await _embedding_client.aembed_query(text)

async def invoke_streaming_prompt_async(
    prompt: LanguageModelInput, model: AvailableModels = AvailableModels.GPT_4O
):
    client = _get_client(model)
    async for chunk in client.astream(prompt):
        yield chunk