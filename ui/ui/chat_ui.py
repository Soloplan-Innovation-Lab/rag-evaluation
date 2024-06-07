import pandas as pd
import plotly.express as px
from pydantic import ValidationError
import streamlit as st
from internal_shared.models.ai import AvailableModels, get_chat_models
from internal_shared.models.chat import (
    ChatRequest,
    PostRetrievalType,
    PreRetrievalType,
    RetrievalConfig,
)
from internal_shared.models.evaluation import ChatEvaluationRequest
from api.rag import (
    create_response,
    create_stream_response,
    evaluate_request,
)
from utils.helper import get_prompt_template
from utils.state_management import initialize_chat_state, reset_chat_state


def chat_interface():
    initialize_chat_state()
    st.title("RAG Chat Playground")

    col1, col2 = st.columns([2, 1])

    with col1:
        chat_area()

    with col2:
        setup_configuration()

    chat_metrics()


@st.experimental_dialog("Request preview")
def open_preview_dialog():
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


def chat_area():
    st.subheader("Chat Interface")
    sbcol1, sbcol2 = st.columns([1, 1])
    with sbcol1:
        if st.button("Reset"):
            reset_chat_state()
    with sbcol2:
        if st.button("Preview JSON"):
            open_preview_dialog()

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


def setup_configuration():
    st.subheader("Configuration")

    active_model = st.selectbox(
        "Select Model",
        options=[model.name for model in get_chat_models()],
        key="active_model",
    )
    prompt_template_choice = st.selectbox(
        "Select Prompt Template",
        options=[template["name"] for template in st.session_state.prompt_templates],
        key="template_choice",
    )

    spt = get_prompt_template()
    if spt:
        with st.expander("Selected Template Details"):
            st.markdown(f"**Name:** {spt.name}")
            h_t = spt.template.replace("{", "`{").replace("}", "}`")
            st.markdown(f"**Template:** {h_t}")

    streaming = st.checkbox("Stream Response", key="streaming")
    evaluation = st.checkbox("Evaluate Response", key="evaluation")
    pre_retrieval_types = [e.value for e in PreRetrievalType]
    post_retrieval_types = [e.value for e in PostRetrievalType]
    retriever_configs = st.session_state.retriever_configs

    if "retrieval_behaviours" not in st.session_state:
        st.session_state.retrieval_behaviours = []

    def add_retrieval_step():
        st.session_state.retrieval_behaviours.append(
            {
                "retriever": retriever_configs[0],
                "context_key": "context",
                "pre_retrieval_type": pre_retrieval_types[0],
                "post_retrieval_type": post_retrieval_types[0],
                "top_k": 5,
                "threshold": 0.5,
            }
        )

    def remove_retrieval_step(index):
        st.session_state.retrieval_behaviours.pop(index)

    if st.button("Add Retrieval Step"):
        add_retrieval_step()

    for i, retrieval in enumerate(st.session_state.retrieval_behaviours):
        with st.expander(f"Step {i + 1}"):
            retriever_index = next(
                (
                    index
                    for (index, d) in enumerate(retriever_configs)
                    if d == retrieval["retriever"]
                ),
                0,
            )

            retrieval["retriever"] = st.selectbox(
                "Select Retriever",
                format_func=lambda x: x["retriever_name"],
                options=retriever_configs,
                index=retriever_index,
                key=f"retriever_{i}",
            )
            retrieval["context_key"] = st.text_input(
                "Context Key",
                value=retrieval["context_key"],
                key=f"context_key_{i}",
            )
            retrieval["pre_retrieval_type"] = st.selectbox(
                "Pre-Retrieval Type",
                options=pre_retrieval_types,
                index=pre_retrieval_types.index(retrieval["pre_retrieval_type"]),
                key=f"pre_retrieval_type_{i}",
            )
            retrieval["post_retrieval_type"] = st.selectbox(
                "Post-Retrieval Type",
                options=post_retrieval_types,
                index=post_retrieval_types.index(retrieval["post_retrieval_type"]),
                key=f"post_retrieval_type_{i}",
            )
            retrieval["top_k"] = st.number_input(
                "Top K",
                min_value=1,
                max_value=10,
                value=retrieval["top_k"],
                key=f"top_k_{i}",
            )
            retrieval["threshold"] = st.slider(
                "Threshold",
                min_value=0.0,
                max_value=1.0,
                value=retrieval["threshold"],
                key=f"threshold_{i}",
            )
            if st.button(f"Remove Step {i + 1}"):
                remove_retrieval_step(i)


def chat_metrics():
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

            evaluation_data = st.session_state.get("evaluation_data", {})
            if evaluation_data:
                rows = []

                # Assuming evaluation_data itself is a single dictionary with necessary keys
                # Check and process deepeval metrics
                deepeval_metrics = evaluation_data.get("deepeval", {})
                for metric_name, metric in deepeval_metrics.items():
                    rows.append(
                        {
                            "Metric": metric_name,
                            "Reason": metric.get("reason", ""),
                            "Score": metric.get("score", 0),
                            "Threshold": metric.get("threshold", None),
                            "Success": metric.get("success", None),
                        }
                    )

                # Check and process ragas metrics
                ragas_metrics = evaluation_data.get("ragas", {})
                for metric_name, score in ragas_metrics.items():
                    rows.append(
                        {
                            "Metric": metric_name,
                            "Reason": "RAGAS",
                            "Score": score,
                            "Threshold": None,
                            "Success": None,
                        }
                    )

                # Create a DataFrame from the list of rows
                ev_df = pd.DataFrame(rows)

                st.subheader("Evaluation Metrics")
                # Display the DataFrame in Streamlit
                st.dataframe(ev_df, use_container_width=True)

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
