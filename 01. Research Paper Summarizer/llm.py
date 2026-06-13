import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")

def load_llm():

    llm = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        max_new_tokens=512,
        temperature=0.4
    )

    return llm