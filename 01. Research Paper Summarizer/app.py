import json
import streamlit as st

from langchain.prompts import PromptTemplate
from research_papers import PAPERS
from llm import load_llm

# -----------------------
# Page Config
# -----------------------

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Research Paper Summarizer")

st.write(
    "Choose a research paper, summary length, and explanation style."
)

# -----------------------
# Dropdown 1
# -----------------------

selected_paper = st.selectbox(
    "Select Research Paper",
    list(PAPERS.keys())
)

# -----------------------
# Dropdown 2
# -----------------------

summary_length = st.selectbox(
    "Select Summary Length",
    [
        "1-2 paragraphs",
        "3-4 paragraphs",
        "5-6 paragraphs"
    ]
)

# -----------------------
# Dropdown 3
# -----------------------

explanation_style = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner Friendly",
        "Technical",
        "Professor Style",
        "Interview Preparation",
        "ELI5"
    ]
)

# -----------------------
# Submit Button
# -----------------------

submit = st.button("Generate Summary")

if submit:

    with open(
        "prompts/prompt_template.json",
        "r",
        encoding="utf-8"
    ) as f:

        prompt_data = json.load(f)

    prompt_template = PromptTemplate(
        input_variables=[
            "paper",
            "length",
            "style"
        ],
        template=prompt_data["template"]
    )

    final_prompt = prompt_template.format(
        paper=PAPERS[selected_paper],
        length=summary_length,
        style=explanation_style
    )

    llm = load_llm()

    with st.spinner("Generating summary..."):

        response = llm.invoke(final_prompt)

    # Output Text Box
    st.text_area(
        label="Generated Summary",
        value=response,
        height=400
    )