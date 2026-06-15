from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline

# --------------------------------------------------
# Load TinyLlama
# --------------------------------------------------

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=512,
    temperature=0.7
)

llm = HuggingFacePipeline(pipeline=pipe)

# --------------------------------------------------
# Output Parser
# --------------------------------------------------

parser = StrOutputParser()

# --------------------------------------------------
# Dynamic Prompt 1 (Report Generator)
# --------------------------------------------------

report_prompt = PromptTemplate(
    input_variables=["topic", "audience", "depth"],
    template="""
You are an expert researcher.

Create a detailed report on the topic: {topic}

Requirements:
- Audience: {audience}
- Detail Level: {depth}
- Include introduction
- Include key concepts
- Include practical applications
- Include conclusion

Generate a professional report.
"""
)

# --------------------------------------------------
# Prompt 2 (Summarizer)
# --------------------------------------------------

summary_prompt = PromptTemplate(
    input_variables=["report"],
    template="""
You are an expert summarizer.

Summarize the following report in 5 bullet points.

Report:
{report}
"""
)

# --------------------------------------------------
# Chain 1: Report Generation
# --------------------------------------------------

report_chain = (
    report_prompt
    | llm
    | parser
)

# --------------------------------------------------
# Chain 2: Summary Generation
# --------------------------------------------------

summary_chain = (
    summary_prompt
    | llm
    | parser
)

# --------------------------------------------------
# Combined Chain
# --------------------------------------------------

full_chain = (
    report_chain
    | (lambda report: {"report": report})
    | summary_chain
)

# --------------------------------------------------
# User Input
# --------------------------------------------------

user_input = {
    "topic": "Artificial Intelligence in Healthcare",
    "audience": "MTech Students",
    "depth": "Advanced"
}

# --------------------------------------------------
# Run Chain
# --------------------------------------------------

summary = full_chain.invoke(user_input)

print("\nSUMMARY\n")
print(summary)