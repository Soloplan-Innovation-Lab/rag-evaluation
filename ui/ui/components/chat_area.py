import streamlit as st
from pydantic import ValidationError
from api.rag import (
    create_response,
    create_stream_response,
    evaluate_request,
)
from utils.helper import get_prompt_template
from utils.state_management import reset_chat_state
from internal_shared.models.ai import AvailableModels
from internal_shared.models.chat import ChatRequest, RetrievalConfig
from internal_shared.models.evaluation import ChatEvaluationRequest


@st.experimental_dialog("Request preview")
def _open_preview_dialog():
    try:
        retrieval_configs = [
            RetrievalConfig(**config)
            for config in st.session_state["retrieval_behaviours"]
        ]

        prompt_template = get_prompt_template()
        if prompt_template:
            chat_request = ChatRequest(
                query=st.session_state.get("current_prompt", ""),
                retrieval_behaviour=retrieval_configs,
                model=AvailableModels[st.session_state.active_model],
                prompt_template=prompt_template,
                history=[],
            )
            st.json(chat_request.model_dump_json(by_alias=True, exclude=["id"]))
        else:
            st.toast("Prompt template not found.", icon=":material/error:")
    except ValidationError as e:
        st.toast(f"{e}", icon=":material/error:")


def render_chat_area():
    st.subheader("Chat Interface")
    sbcol1, sbcol2 = st.columns([1, 1])
    with sbcol1:
        if st.button("Reset"):
            reset_chat_state()
    with sbcol2:
        if st.button("Preview JSON"):
            _open_preview_dialog()

    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("What is up?", key="chat_input"):
        st.session_state.current_prompt = prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        def _create_chat_request():
            pt = get_prompt_template()
            return ChatRequest(
                query=prompt,
                retrieval_behaviour=[
                    RetrievalConfig(**config)
                    for config in st.session_state["retrieval_behaviours"]
                ],
                model=AvailableModels[st.session_state.get("active_model")],
                prompt_template=pt,
                history=[],
            )

        if st.session_state.get("streaming", False):
            response_generator = create_stream_response(_create_chat_request())
            assistant_response = ""
            with chat_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    response_chunks = []
                    for response_chunk in response_generator:
                        response_chunks.append(response_chunk)
                        response_placeholder.markdown("".join(response_chunks))
                    assistant_response = "".join(response_chunks)
        else:
            response_data = create_response(_create_chat_request())
            if response_data:
                response = response_data.get(
                    "response", "An error occurred. Please try again."
                )
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                st.session_state.response_data = response_data

        if st.session_state.evaluation:
            prompt_template = get_prompt_template()
            template_content = prompt_template.template if prompt_template else ""
            evaluation_request = ChatEvaluationRequest(
                description="playground_eval",
                run_type=f"rag_playground__{st.session_state.template_choice}",
                input=prompt,
                actual_output=assistant_response,
                retrieval_context=st.session_state.response_data.get("documents", []),
                system_prompt=template_content,
                chat_session_id=st.session_state.chat_session_id,
            )
            ev_res = evaluate_request(evaluation_request)
            st.session_state.evaluation_data = ev_res
