import streamlit as st
from internal_shared.models.ai import AvailableModels, get_embedding_models
from internal_shared.models.chat import (
    PromptTemplate,
    RetrieverType,
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
            options=[model.name for model in RetrieverType],
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
            if retriever_type == RetrieverType.GRAPH.name:
                index_name = None
                retriever_select = None
                mappings = None

            cfg = RetrieverConfig(
                retriever_name=retriever_name,
                retriever_type=(
                    RetrieverType[retriever_type] if retriever_type else None
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

    template_name = ""
    template_content = ""
    few_shot_key = ""
    few_shot_value = ""

    def get_template_content(templates: list, choice: str):
        template_dict = next(
            (template for template in templates if template["name"] == choice),
            None,
        )
        return PromptTemplate.model_validate(template_dict) if template_dict else None

    if prompt_template_choice != "Create New":
        selected_template = get_template_content(
            prompt_templates, prompt_template_choice
        )
        if selected_template:
            template_name = selected_template.name
            template_content = selected_template.template
            few_shot_key = selected_template.few_shot_key or ""
            few_shot_value = selected_template.few_shot_value or ""

    template_name = st.text_input("Template Name", value=template_name)
    template_content = st.text_area("Template Content", value=template_content)
    with st.expander("Few Shot Configuration"):
        few_shot_key = st.text_input("Few Shot Key", value=few_shot_key)
        few_shot_value = st.text_area("Few Shot Value", value=few_shot_value)

    sb1, sb2 = st.columns([1, 1])

    if prompt_template_choice == "Create New":
        with sb1:
            if st.button("Create Template"):
                create_prompt_template(
                    PromptTemplate(
                        name=template_name,
                        template=template_content,
                        few_shot_key=few_shot_key if few_shot_key else None,
                        few_shot_value=few_shot_value if few_shot_value else None,
                    )
                )
    else:
        with sb1:
            if st.button("Delete Template"):
                delete_prompt_template(prompt_template_choice)
        with sb2:
            if st.button("Update Template"):
                update_prompt_template(
                    prompt_template_choice,
                    PromptTemplate(
                        name=template_name,
                        template=template_content,
                        few_shot_key=few_shot_key if few_shot_key else None,
                        few_shot_value=few_shot_value if few_shot_value else None,
                    ),
                )

    st.session_state["template_name"] = template_name
    st.session_state["template_content"] = template_content
