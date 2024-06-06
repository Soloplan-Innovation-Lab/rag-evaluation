import streamlit as st
from api.rag import (
    get_prompt_templates,
    get_retriever_configs,
)

def initialize_chat_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "retrieval_behaviours" not in st.session_state:
        st.session_state.retrieval_behaviours = []
    if "response_data" not in st.session_state:
        st.session_state.response_data = {}
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""
    if "historical_responses" not in st.session_state:
        st.session_state.historical_responses = []
    if "evaluation_data" not in st.session_state:
        st.session_state.evaluation_data = {}
    if "template_name" not in st.session_state:
        st.session_state.template_name = ""
    if "template_content" not in st.session_state:
        st.session_state.template_content = ""
    if "retriever_configs" not in st.session_state:
        st.session_state.retriever_configs = get_retriever_configs() or []
    if "prompt_templates" not in st.session_state:
        st.session_state.prompt_templates = get_prompt_templates() or []

def reset_chat_state():
    st.session_state.messages = []
    st.session_state.retrieval_behaviours = []
    st.session_state.response_data = {}
    st.session_state.chat_session_id = None
    st.session_state.current_prompt = ""
    st.session_state.historical_responses = []
    st.session_state.evaluation_data = {}
    st.session_state.template_name = ""
    st.session_state.template_content = ""
