import os
import requests
import json
import streamlit as st
from internal_shared.models.chat import (
    ChatRequest,
    ChatResponseChunk,
    PromptTemplate,
    RetrieverConfig,
)
from internal_shared.models.evaluation import ChatEvaluationRequest

_RAG_API_BASE = os.getenv("RAG_PIPELINE_URL")
if _RAG_API_BASE is None:
    raise ValueError("RAG_PIPELINE_URL environment variable not set")

_EVAL_API_BASE = os.getenv("EVALUATION_URL")
if _EVAL_API_BASE is None:
    raise ValueError("EVALUATION_URL environment variable not set")


# HANDLE CHAT REQUESTS
def create_response(chat_request: ChatRequest):
    request_uri = f"{_RAG_API_BASE}/chat"
    if "chat_session_id" in st.session_state and st.session_state["chat_session_id"]:
        request_uri += f"?chat_id={st.session_state['chat_session_id']}"

    response = requests.post(
        request_uri,
        json=chat_request.model_dump(by_alias=True),
    )
    if response.ok:
        response_data = response.json()
        st.session_state.chat_session_id = response_data.get("chat_session_id")
        st.session_state.historical_responses.append(response_data)
        return response_data
    else:
        st.toast(f"{response.status_code}: {response.reason}", icon=":material/error:")


def create_stream_response(chat_request: ChatRequest):
    request_uri = f"{_RAG_API_BASE}/chat/stream"
    if "chat_session_id" in st.session_state and st.session_state["chat_session_id"]:
        request_uri += f"?chat_id={st.session_state['chat_session_id']}"

    response = requests.post(
        request_uri,
        json=chat_request.model_dump(by_alias=True),
        stream=True,
    )
    complete_response = ""
    metadata = None
    try:
        for chunk in response.iter_content(chunk_size=None):
            chunk_data = chunk.decode("utf-8")
            chunk_dict = json.loads(chunk_data)
            chunk_response = ChatResponseChunk.model_validate(chunk_dict)
            if chunk_response.chunk:
                complete_response += chunk_response.chunk
                yield chunk_response.chunk
            if chunk_response.metadata:
                metadata = chunk_response.metadata
    except Exception as e:
        st.toast(f"{e}", icon=":material/error:")

    if metadata:
        st.session_state.response_data = metadata
        st.session_state.historical_responses.append(metadata)
        st.session_state.chat_session_id = metadata.get("chat_session_id")
        st.session_state.messages.append(
            {"role": "assistant", "content": complete_response}
        )


# HANDLE EVALUATION
def evaluate_request(payload: ChatEvaluationRequest):
    response = requests.post(
        f"{_EVAL_API_BASE}/evaluate/chat", json=payload.model_dump(by_alias=True)
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.toast(f"{response.status_code}: {response.reason}", icon=":material/error:")


# HANDLE PROMPT TEMPLATES
def get_prompt_templates():
    response = requests.get(
        f"{_RAG_API_BASE}/prompt_template", params={"skip": 0, "limit": 100}
    )
    if response.ok:
        return response.json()


def create_prompt_template(payload: PromptTemplate):
    response = requests.post(
        f"{_RAG_API_BASE}/prompt_template",
        json=payload.model_dump(by_alias=True),
    )
    if response.ok:
        st.toast(
            f"{response.status_code}: Prompt template created", icon=":material/check:"
        )
        return response.json()
    else:
        st.toast(f"{response.status_code}: {response.reason}", icon=":material/error:")


def delete_prompt_template(template: str):
    response = requests.delete(f"{_RAG_API_BASE}/prompt_template/{template}")
    if response.ok:
        st.toast(
            f"{response.status_code}: Prompt template deleted", icon=":material/check:"
        )
        return response.json()
    else:
        st.toast(f"{response.status_code}: {response.reason}", icon=":material/error:")


def update_prompt_template(template: str, payload: PromptTemplate):
    response = requests.put(
        f"{_RAG_API_BASE}/prompt_template/{template}",
        json=payload.model_dump(by_alias=True),
    )
    if response.ok:
        st.toast(
            f"{response.status_code}: Prompt template updated", icon=":material/check:"
        )
        return response.json()
    else:
        st.toast(f"{response.status_code}: {response.reason}", icon=":material/error:")


# HANDLE RETRIEVER CONFIGURATION
def get_retriever_configs():
    response = requests.get(f"{_RAG_API_BASE}/retriever_config")
    if response.ok:
        return response.json()
    else:
        st.toast(f"{response.status_code}: {response.reason}", icon=":material/error:")


def create_retriever_config(payload: RetrieverConfig):
    response = requests.post(
        f"{_RAG_API_BASE}/retriever_config",
        json=payload.model_dump(by_alias=True),
    )
    if response.ok:
        st.toast(
            f"{response.status_code}: Retriever configuration created",
            icon=":material/check:",
        )
        return response.json()
    else:
        st.toast(f"{response.status_code}: {response.reason}", icon=":material/error:")


def delete_retriever_config(retriever_name: str):
    response = requests.delete(f"{_RAG_API_BASE}/retriever_config/{retriever_name}")
    if response.ok:
        st.toast(
            f"{response.status_code}: Retriever configuration deleted",
            icon=":material/check:",
        )
        return response.json()
    else:
        st.toast(f"{response.status_code}: {response.reason}", icon=":material/error:")


def update_retriever_config(retriever_name: str, payload: RetrieverConfig):
    response = requests.put(
        f"{_RAG_API_BASE}/retriever_config/{retriever_name}",
        json=payload.model_dump(by_alias=True),
    )
    if response.ok:
        st.toast(
            f"{response.status_code}: Retriever configuration updated",
            icon=":material/check:",
        )
        return response.json()
    else:
        st.toast(f"{response.status_code}: {response.reason}", icon=":material/error:")
