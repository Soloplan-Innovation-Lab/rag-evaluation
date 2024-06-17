import streamlit as st
from internal_shared.models.ai import get_chat_models
from internal_shared.models.chat import PostRetrievalType, PreRetrievalType
from utils.helper import get_prompt_template

def render_basic_configuration():
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
