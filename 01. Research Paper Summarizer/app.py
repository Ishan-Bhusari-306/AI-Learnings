import json

import streamlit as st

from langchain import PromptTemplate

from research_papers import PAPERS
from llm import load_llm


# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="centered"
)

# -------------------------
# LOAD JSON CONFIG
# -------------------------

with open(
    "prompts/prompt_config.json",
    "r",
    encoding="utf-8"
) as f:

    config = json.load(f)

MODEL_CONFIG = config["model_config"]
PROMPT_TEMPLATE = config["prompt_template"]


# -------------------------
# TITLE
# -------------------------

st.title("📚 Research Paper Summarizer")

st.markdown(
    """
Generate customized summaries of
famous Machine Learning and NLP papers.
"""
)

# -------------------------
# DROPDOWN 1
# -------------------------

selected_paper = st.selectbox(
    "Select Research Paper",
    list(PAPERS.keys())
)

# -------------------------
# DROPDOWN 2
# -------------------------

summary_length = st.selectbox(
    "Select Summary Length",
    [
        "1-2 paragraphs",
        "3-4 paragraphs",
        "5-6 paragraphs"
    ]
)

# -------------------------
# DROPDOWN 3
# -------------------------

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

# -------------------------
# SUBMIT BUTTON
# -------------------------

generate_btn = st.button(
    "🚀 Generate Summary",
    use_container_width=True
)

# -------------------------
# GENERATE
# -------------------------

if generate_btn:

    paper_content = PAPERS[selected_paper]

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=[
            "paper",
            "length",
            "style"
        ]
    )

    final_prompt = prompt.format(
        paper=paper_content,
        length=summary_length,
        style=style
    )

    llm = load_llm(MODEL_CONFIG)

    with st.spinner(
        "Generating summary using Llama 3.2 3B..."
    ):

        response = llm.invoke(final_prompt)

    st.markdown("---")

    st.subheader("Generated Summary")

    st.text_area(
        label="Output",
        value=response,
        height=450
    )