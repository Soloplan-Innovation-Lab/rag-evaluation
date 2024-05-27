import tiktoken
from internal_shared.logger import get_logger
from internal_shared.models.chat import TokenUsage
from typing import Dict, List, Tuple
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

_logger = get_logger(__name__)


def calculate_token_usage(
    prompt: List[BaseMessage], response: BaseMessage, model_name: str
):
    # check, if response.response_metadata has key token_usage
    if (
        hasattr(response, "response_metadata")
        and "token_usage" in response.response_metadata
    ):
        token_usage: Dict = response.response_metadata.get("token_usage")
        if token_usage and isinstance(token_usage, dict):
            _logger.info("Token usage present in response.response_metadata")

            completion_tokens = token_usage.get("completion_tokens", 0)
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            return TokenUsage(
                completion_tokens=completion_tokens,
                prompt_tokens=prompt_tokens,
                total_tokens=completion_tokens + prompt_tokens,
            )

    # case: token_usage not present in response.response_metadata
    prompt_messages = [msg.content for msg in prompt]
    return calculate_token_str_usage(prompt_messages, response.content, model_name)

def calculate_token_str_usage(prompt: List[str], response: str, model_name: str) -> TokenUsage:
    prompt_string = "".join(prompt)

    # Encoding list:
    # https://github.com/openai/tiktoken/blob/c0ba74c238d18b4824c25f3c27fc8698055b9a76/tiktoken/model.py#L20
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        _logger.info(f"Using encoding '{encoding.name}' for model {model_name}")
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
        _logger.warning(
            f"Model '{model_name}' not found. Using default encoding '{encoding.name}'"
        )

    # use batch encoding to get token usage
    encoded_prompt, encoded_response = encoding.encode_ordinary_batch(
        [prompt_string, response]
    )

    return TokenUsage(
        completion_tokens=len(encoded_response),
        prompt_tokens=len(encoded_prompt),
        total_tokens=len(encoded_prompt + encoded_response),
    )

def format_chat_history(history: List[Tuple[str, str]]) -> List:
    """Implementation based on langchain example

    https://github.com/langchain-ai/langchain/blob/master/templates/rag-conversation/rag_conversation/chain.py#L86
    """
    buffer = []
    for human, ai in history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


def summarize_chat_history(history: List[Tuple[str, str]]):
    """Chat summarization logic based on the openai community thread

    https://community.openai.com/t/how-does-chatgpt-store-history-of-chat/319608/3
    """
    # if len(history) > 10:
    #     summary = "Summary of previous conversation: " + " ".join(
    #         [msg for role, msg in history[:-10]]
    #     )
    #     return [(role, msg) for role, msg in history[-10:]] + [("system", summary)]
    # return history

    return history[-10:]
