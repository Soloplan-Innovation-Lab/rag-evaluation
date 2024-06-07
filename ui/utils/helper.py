from internal_shared.models.chat import PromptTemplate
import streamlit as st


def get_prompt_template() -> PromptTemplate | None:
    if (
        "prompt_templates" not in st.session_state
        or "template_choice" not in st.session_state
    ):
        return None

    prompt_template = next(
        (
            template
            for template in st.session_state.prompt_templates
            if template["name"] == st.session_state.template_choice
        ),
        None,
    )

    return PromptTemplate(
        name=prompt_template["name"],
        template=prompt_template["template"],
        few_shot_key=prompt_template["few_shot_key"],
        few_shot_value=prompt_template["few_shot_value"],
    )
