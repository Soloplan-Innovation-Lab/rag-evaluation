from internal_shared.models.ai import AvailableModels
import streamlit as st
import tiktoken

st.title("Token Calculator")

colors = [
    "#FFCDD2",
    "#F8BBD0",
    "#E1BEE7",
    "#D1C4E9",
    "#C5CAE9",
    "#BBDEFB",
    "#B3E5FC",
    "#B2EBF2",
    "#B2DFDB",
    "#C8E6C9",
    "#DCEDC8",
    "#F0F4C3",
    "#FFECB3",
    "#FFE0B2",
    "#FFCCBC",
]

# Create two columns
col1, col2 = st.columns(2)

with col1:
    p1 = st.empty()
    text = p1.text_area("Input Text", height=350)

    selected_model = st.selectbox(
        "Select Model", options=[model.value for model in AvailableModels]
    )

    c1, c2 = st.columns(2)

    with c1:
        clear_button = st.button("Clear")

    with c2:
        calculate_button = st.button("Calculate Tokens")

with col2:
    token_count_container = st.empty()
    token_count_container.write("Total Tokens:")
    container = st.container(height=500, border=True)
    with container:
        token_display_container = st.empty()
        if calculate_button:
            try:
                encoding = tiktoken.encoding_for_model(selected_model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            encoded_text = encoding.encode(text)

            tokens_with_highlight = ""
            color_index = 0
            num_colors = len(colors)
            for token in encoded_text:
                token_str = encoding.decode([token])
                tokens_with_highlight += (
                    f'<span style="background-color: {colors[color_index]}; color: black;">'
                    f"{token_str}</span> "
                )
                color_index = (color_index + 1) % num_colors

            token_count_container.markdown(f"Total Tokens: ``{len(encoded_text)}``")
            token_display_container.markdown(
                tokens_with_highlight, unsafe_allow_html=True
            )

        if clear_button:
            p1.text_area("Input Text", value="", height=350)
            token_count_container.write("")
            token_display_container.write("")
