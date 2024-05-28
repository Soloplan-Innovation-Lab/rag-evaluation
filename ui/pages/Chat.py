import json
import os
import requests
import pandas as pd
import plotly.express as px
from pydantic import ValidationError
import streamlit as st
from internal_shared.ai_models import AvailableModels
from internal_shared.models.chat import (
    ChatRequest,
    ChatResponseChunk,
    PromptTemplate,
    RetrievalConfig,
    PreRetrievalType,
    RetrievalType,
    PostRetrievalType,
)

_API_BASE_URI = os.getenv("RAG_PIPELINE_URL")
if _API_BASE_URI is None:
    raise ValueError("RAG_PIPELINE_URL environment variable not set")
_MAIN_CONTAINER_HEIGHT = 725

st.title("RAG Chat Playground")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_behaviours" not in st.session_state:
    st.session_state["retrieval_behaviours"] = []
if "response_data" not in st.session_state:
    st.session_state.response_data = {}
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""
if "historical_responses" not in st.session_state:
    st.session_state["historical_responses"] = []


def create_response(chat_request: ChatRequest):
    if "chat_session_id" in st.session_state and st.session_state["chat_session_id"] is not None:
        request_uri = (
            f"{_API_BASE_URI}/chat?chat_id={st.session_state['chat_session_id']}"
        )
    else:
        request_uri = f"{_API_BASE_URI}/chat"
    response = requests.post(
        request_uri,
        json=chat_request.model_dump(by_alias=True),
    )

    if response.status_code == 200:
        response_data = response.json()
        st.session_state.chat_session_id = response_data.get("chat_session_id")
        st.session_state.historical_responses.append(response_data)

        return response_data
    else:
        st.error(f"{response.status_code}: {response.reason}")


def create_stream_response(chat_request: ChatRequest):
    if "chat_session_id" in st.session_state and st.session_state["chat_session_id"] is not None:
        request_uri = (
            f"{_API_BASE_URI}/chat/stream?chat_id={st.session_state['chat_session_id']}"
        )
    else:
        request_uri = f"{_API_BASE_URI}/chat/stream"
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
        st.error(e)

    # Update session state with the final response and metadata
    if metadata:
        st.session_state.response_data = metadata
        st.session_state.historical_responses.append(metadata)
        st.session_state.chat_session_id = metadata.get("chat_session_id")
        st.session_state.messages.append(
            {"role": "assistant", "content": complete_response}
        )


def get_prompt_templates():
    response = requests.get(
        f"{_API_BASE_URI}/prompt_template", params={"skip": 0, "limit": 100}
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"{response.status_code}: {response.reason}")


def create_prompt_template(name: str, template: str):
    payload = PromptTemplate(name=name, template=template)
    response = requests.post(
        f"{_API_BASE_URI}/prompt_template",
        json=payload.model_dump(by_alias=True),
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"{response.status_code}: {response.reason}")


def delete_prompt_template(template: str):
    response = requests.delete(f"{_API_BASE_URI}/prompt_template/{template}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"{response.status_code}: {response.reason}")


def update_prompt_template(template: str, payload: PromptTemplate):
    response = requests.put(
        f"{_API_BASE_URI}/prompt_template/{template}",
        json=payload.model_dump(by_alias=True),
    )
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"{response.status_code}: {response.reason}")


@st.experimental_dialog("Request preview")
def open_preview_dialog():
    try:
        retrieval_configs = [
            RetrievalConfig(**config)
            for config in st.session_state["retrieval_behaviours"]
        ]
        chat_request = ChatRequest(
            query=st.session_state.get("current_prompt", ""),
            retrieval_behaviour=retrieval_configs,
            model=AvailableModels[active_model],
            prompt_template=(
                PromptTemplate(name=template_name, template=template_content)
                if template_name
                else None
            ),
            history=[],
        )
        st.json(chat_request.model_dump_json(by_alias=True, exclude=["id"]))
    except ValidationError as e:
        st.error(f"Validation Error: {e}")


# Columns for layout
col1, col2, col3 = st.columns([1, 2, 1])

# Prompt Template Column
with col1:
    with st.container(border=True, height=_MAIN_CONTAINER_HEIGHT):
        st.subheader("Setup")
        prompt_templates = get_prompt_templates()
        prompt_template_choice = st.selectbox(
            "Select Prompt Template",
            options=["Create New"]
            + [template["name"] for template in prompt_templates],
        )

        p1 = st.empty()
        p2 = st.empty()

        sb1, sb2 = st.columns([1, 1])

        template_name = p1.text_input("Template Name")
        template_content = p2.text_area("Template Content")
        if prompt_template_choice == "Create New":
            if st.button("Create Template"):
                create_prompt_template(template_name, template_content)
        else:
            template_name = p1.text_input("Template Name", value=prompt_template_choice)
            template_content = p2.text_area(
                "Template Content",
                value=[
                    template["template"]
                    for template in prompt_templates
                    if template["name"] == prompt_template_choice
                ][0],
            )
            with sb1:
                if st.button("Delete Template"):
                    # renew prompt_templates?
                    delete_prompt_template(prompt_template_choice)
            with sb2:
                if st.button("Update Template"):
                    delete_prompt_template(prompt_template_choice)
                    create_prompt_template(template_name, template_content)
            selected_template = prompt_template_choice

# Model and Retrieval Behavior Column
with col3:
    with st.container(border=True, height=_MAIN_CONTAINER_HEIGHT):
        st.subheader("Configuration")
        active_model = st.selectbox(
            "Select Model", options=[model.name for model in AvailableModels]
        )

        streaming = st.checkbox("Stream Response", key="streaming")

        retrieval_types = [e.value for e in RetrievalType]
        pre_retrieval_types = [e.value for e in PreRetrievalType]
        post_retrieval_types = [e.value for e in PostRetrievalType]

        if "retrieval_behaviours" not in st.session_state:
            st.session_state["retrieval_behaviours"] = []

        def add_retrieval_step():
            st.session_state["retrieval_behaviours"].append(
                {
                    "retrieval_type": retrieval_types[0],
                    "pre_retrieval_type": pre_retrieval_types[0],
                    "post_retrieval_type": post_retrieval_types[0],
                    "top_k": 5,
                    "threshold": 0.5,
                }
            )

        def remove_retrieval_step(index):
            st.session_state["retrieval_behaviours"].pop(index)

        if st.button("Add Retrieval Step"):
            add_retrieval_step()

        for i, retrieval in enumerate(st.session_state["retrieval_behaviours"]):
            with st.expander(f"Step {i + 1}"):
                retrieval["retrieval_type"] = st.selectbox(
                    f"Retrieval Type for Step {i + 1}",
                    options=retrieval_types,
                    index=retrieval_types.index(retrieval["retrieval_type"]),
                )
                retrieval["pre_retrieval_type"] = st.selectbox(
                    f"Pre-Retrieval Type for Step {i + 1}",
                    options=pre_retrieval_types,
                    index=pre_retrieval_types.index(retrieval["pre_retrieval_type"]),
                )
                retrieval["post_retrieval_type"] = st.selectbox(
                    f"Post-Retrieval Type for Step {i + 1}",
                    options=post_retrieval_types,
                    index=post_retrieval_types.index(retrieval["post_retrieval_type"]),
                )
                retrieval["top_k"] = st.number_input(
                    f"Top K for Step {i + 1}", min_value=1, value=retrieval["top_k"]
                )
                retrieval["threshold"] = st.slider(
                    f"Threshold for Step {i + 1}",
                    min_value=0.0,
                    max_value=1.0,
                    value=retrieval["threshold"],
                )
                if st.button(f"Remove Step {i + 1}"):
                    remove_retrieval_step(i)

# Middle Column for Chat Interface
with col2:
    with st.container(border=True, height=_MAIN_CONTAINER_HEIGHT):
        st.subheader("Chat Interface")
        sbcol1, sbcol2 = st.columns([1, 1])
        with sbcol1:
            if st.button("Reset"):
                st.session_state["retrieval_behaviours"] = []
                st.session_state.messages = []
                st.session_state.response_data = {}
                st.session_state.chat_session_id = None
                st.session_state.current_prompt = ""
                st.rerun()
        with sbcol2:
            if st.button("Preview JSON"):
                open_preview_dialog()

        # Display chat messages from history on app rerun
        chat_container = st.container(height=500)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is up?", key="chat_input"):
            st.session_state.current_prompt = prompt
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Get assistant response
            chat_request = ChatRequest(
                query=prompt,
                retrieval_behaviour=[
                    RetrievalConfig(**config)
                    for config in st.session_state["retrieval_behaviours"]
                ],
                model=AvailableModels[active_model],
                prompt_template=(
                    PromptTemplate(name=template_name, template=template_content)
                    if template_name
                    else None
                ),
                history=[],
            )
            if streaming:
                response_generator = create_stream_response(chat_request)

                # Display assistant response in chat message container
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
                response_data = create_response(chat_request)
                if response_data:
                    response = response_data.get(
                        "response", "An error occurred. Please try again."
                    )

                    # Display assistant response in chat message container
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(response)

                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    st.session_state.response_data = response_data

# Output Area with Tabs for Detailed View
st.subheader("Response Details")
tabs = st.tabs(["Response", "Documents", "Metadata", "Charts"])
if "response_data" in st.session_state:
    with tabs[0]:
        st.json(st.session_state.response_data)
    with tabs[1]:
        st.json(st.session_state.response_data.get("documents", {}))
    with tabs[2]:
        st.json(st.session_state.response_data.get("steps", {}))
    with tabs[3]:
        rd = st.session_state.response_data
        # Create two columns
        col1, col2 = st.columns(2)

        # Token Usage Charts
        if rd.get("token_usage"):
            token_usage = rd["token_usage"]
            token_usage_df = pd.DataFrame(
                {
                    "Type": ["Prompt Tokens", "Completion Tokens", "Total Tokens"],
                    "Tokens": [
                        token_usage["prompt_tokens"],
                        token_usage["completion_tokens"],
                        token_usage["total_tokens"],
                    ],
                }
            )
            with col1:
                st.subheader("Token Usage")
                st.bar_chart(token_usage_df.set_index("Type"))

        # Retrieval Steps Duration Charts
        if rd.get("steps"):
            steps = rd["steps"]
            steps_data = []
            for i, step in enumerate(steps):
                steps_data.append(
                    {
                        "Step": f"Step {i+1} - Pre-Retrieval",
                        "Duration": step["pre_retrieval_duration"],
                    }
                )
                steps_data.append(
                    {
                        "Step": f"Step {i+1} - Retrieval",
                        "Duration": step["retrieval_duration"],
                    }
                )
                steps_data.append(
                    {
                        "Step": f"Step {i+1} - Post-Retrieval",
                        "Duration": step["post_retrieval_duration"],
                    }
                )
            # Add the response duration
            steps_data.append(
                {"Step": "Response Creation", "Duration": rd["response_duration"]}
            )
            steps_df = pd.DataFrame(steps_data)
            with col2:
                st.subheader("Retrieval Steps Duration")
                pie_chart = px.pie(
                    steps_df,
                    values="Duration",
                    names="Step",
                )
                st.plotly_chart(pie_chart)

        # Line Graphs for Performance and Token Usage Over Time
        historical_responses = st.session_state.historical_responses

        if historical_responses:
            performance_data = []
            token_usage_data = []

            for i, response in enumerate(historical_responses):
                if response.get("response_duration"):
                    performance_data.append(
                        {
                            "Request": i + 1,
                            "Metric": "Response creation",
                            "Duration": response["response_duration"],
                        }
                    )

                if response.get("steps"):
                    steps = response["steps"]
                    for j, step in enumerate(steps):
                        if step.get("pre_retrieval_duration"):
                            performance_data.append(
                                {
                                    "Request": i + 1,
                                    "Metric": f"Step {j + 1} - Pre-Retrieval",
                                    "Duration": step["pre_retrieval_duration"],
                                }
                            )
                        if step.get("retrieval_duration"):
                            performance_data.append(
                                {
                                    "Request": i + 1,
                                    "Metric": f"Step {j + 1} - Retrieval",
                                    "Duration": step["retrieval_duration"],
                                }
                            )
                        if step.get("post_retrieval_duration"):
                            performance_data.append(
                                {
                                    "Request": i + 1,
                                    "Metric": f"Step {j + 1} - Post-Retrieval",
                                    "Duration": step["post_retrieval_duration"],
                                }
                            )

                if response.get("token_usage"):
                    token_usage = response["token_usage"]
                    token_usage_data.append(
                        {
                            "Request": i + 1,
                            "Type": "Prompt Tokens",
                            "Tokens": token_usage["prompt_tokens"],
                        }
                    )
                    token_usage_data.append(
                        {
                            "Request": i + 1,
                            "Type": "Completion Tokens",
                            "Tokens": token_usage["completion_tokens"],
                        }
                    )
                    token_usage_data.append(
                        {
                            "Request": i + 1,
                            "Type": "Total Tokens",
                            "Tokens": token_usage["total_tokens"],
                        }
                    )

            if performance_data:
                performance_df = pd.DataFrame(performance_data)
                performance_df = performance_df.sort_values(by="Duration")
                st.subheader("Performance Over Time")
                line_chart = px.area(
                    performance_df,
                    x="Request",
                    y="Duration",
                    color="Metric",
                    title="Response Duration Over Time",
                )
                line_chart.update_traces(mode="lines+markers")  # Add data points
                st.plotly_chart(line_chart)

            if token_usage_data:
                token_usage_df = pd.DataFrame(token_usage_data)
                token_usage_df = token_usage_df.sort_values(by="Tokens")
                st.subheader("Token Usage Over Time")
                line_chart = px.area(
                    token_usage_df,
                    x="Request",
                    y="Tokens",
                    color="Type",
                    title="Token Usage Over Time",
                )
                line_chart.update_traces(mode="lines+markers")  # Add data points
                st.plotly_chart(line_chart)

else:
    with tabs[0]:
        st.write("Response will be displayed here...")
    with tabs[1]:
        st.write("Documents will be displayed here...")
    with tabs[2]:
        st.write("Metadata will be displayed here...")
    with tabs[3]:
        st.write("Charts will be displayed here...")
