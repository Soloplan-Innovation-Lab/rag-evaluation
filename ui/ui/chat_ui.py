import streamlit as st
from utils.state_management import initialize_chat_state
from ui.components import (
    render_basic_configuration,
    render_chat_area,
    render_basic_chat_charts,
)


def chat_interface():
    initialize_chat_state()
    st.title("RAG Chat Playground")

    col1, col2 = st.columns([2, 1])

    with col1:
        render_chat_area()

    with col2:
        render_basic_configuration()

    render_basic_chat_charts()
