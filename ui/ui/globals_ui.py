import streamlit as st
from internal_shared.models.ai import AvailableModels, get_embedding_models
from internal_shared.models.chat import (
    PromptTemplate,
    RetrievalType,
    RetrieverConfig,
)
from api.rag import (
    get_retriever_configs,
    create_retriever_config,
    update_retriever_config,
    delete_retriever_config,
    get_prompt_templates,
    create_prompt_template,
    update_prompt_template,
    delete_prompt_template,
)


def globals_interface():
    st.title("Globals")
    st.write(
        "This section allows you to configure global settings for the application."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        handle_retriever_config()

    with col2:
        handle_prompt_templates()


def handle_retriever_config():
    st.subheader("Retriever Configuration")
    st.write("This section allows you to configure the retriever.")

    with st.form("retriever_config_form", clear_on_submit=True):
        retriever_name = st.text_input("Retriever Name", key="retriever_name")
        retriever_type = st.selectbox(
            "Retriever Type",
            options=[model.name for model in RetrievalType],
            key="retriever_type",
        )
        index_name = st.text_input("Index Name", key="index_name")
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[model.name for model in get_embedding_models()],
            key="embedding_model",
        )

        retriever_select = st.text_input(
            "Retriever Select Fields",
            value="name,summary,content",
            key="retriever_select",
            help="Comma-separated list of fields to select from the retriever.",
        )

        with st.expander("Field Mappings"):
            mappings = {}
            fields = ["name", "summary", "content"]
            for field in fields:
                mappings[field] = st.text_input(
                    f"Mapping for {field}", key=f"mapping_{field}"
                )

        submit_button = st.form_submit_button("Create Retriever Config")

        if submit_button:
            # if retriever_type == RetrievalType.GRAPH: set index_name, retriever_select, field_mappings to None
            if retriever_type == RetrievalType.GRAPH.name:
                index_name = None
                retriever_select = None
                mappings = None

            cfg = RetrieverConfig(
                retriever_name=retriever_name,
                retriever_type=(
                    RetrievalType[retriever_type] if retriever_type else None
                ),
                index_name=index_name,
                embedding_model=(
                    AvailableModels[embedding_model] if embedding_model else None
                ),
                retriever_select=(
                    retriever_select.split(",") if retriever_select else None
                ),
                field_mappings=mappings if mappings else None,
            )
            create_retriever_config(cfg)
            st.success("Retriever configuration created successfully.")


def handle_prompt_templates():
    st.subheader("Prompt templates")
    prompt_templates = get_prompt_templates() or []
    prompt_template_choice = st.selectbox(
        "Select Prompt Template",
        options=["Create New"] + [template["name"] for template in prompt_templates],
        key="template_choice",
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
                delete_prompt_template(prompt_template_choice)
        with sb2:
            if st.button("Update Template"):
                update_prompt_template(
                    prompt_template_choice,
                    PromptTemplate(name=template_name, template=template_content),
                )

    st.session_state["template_name"] = template_name
    st.session_state["template_content"] = template_content
