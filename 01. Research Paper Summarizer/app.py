import json

import streamlit as st

from langchain_core.prompts import PromptTemplate

from research_papers import PAPERS
from llm import load_llm


# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="centered"
)

# --------------------------------
# LOAD CONFIG
# --------------------------------

with open(
    "prompt_config.json",
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)

MODEL_CONFIG = config["model_config"]
PROMPT_TEMPLATE = config["prompt_template"]

# --------------------------------
# TITLE
# --------------------------------

st.title("📚 Research Paper Summarizer")

st.write(
    "Choose a research paper, summary length, and explanation style."
)

# --------------------------------
# DROPDOWN 1
# --------------------------------

selected_paper = st.selectbox(
    "Select Research Paper",
    list(PAPERS.keys())
)

# --------------------------------
# DROPDOWN 2
# --------------------------------

summary_length = st.selectbox(
    "Select Summary Length",
    [
        "1-2 paragraphs",
        "3-4 paragraphs",
        "5-6 paragraphs"
    ]
)

# --------------------------------
# DROPDOWN 3
# --------------------------------

style = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner Friendly",
        "Technical",
        "Professor Style",
        "Interview Preparation",
        "ELI5"
    ]
)

# --------------------------------
# SUBMIT BUTTON
# --------------------------------

generate_button = st.button(
    "🚀 Generate Summary",
    use_container_width=True
)

# --------------------------------
# GENERATE SUMMARY
# --------------------------------

if generate_button:

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=[
            "paper",
            "length",
            "style"
        ]
    )

    final_prompt = prompt.format(
        paper=PAPERS[selected_paper],
        length=summary_length,
        style=style
    )

    llm = load_llm(MODEL_CONFIG)

    with st.spinner(
        "Generating summary..."
    ):

        response = llm.invoke(
            final_prompt
        )

        summary = response.content

    st.markdown("---")

    st.subheader(
        "Generated Summary"
    )

    st.text_area(
        label="Output",
        value=summary,
        height=450
    )