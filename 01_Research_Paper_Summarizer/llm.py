import os

from dotenv import load_dotenv

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace
)

load_dotenv()


def load_llm(model_config):

    endpoint = HuggingFaceEndpoint(
        repo_id=model_config["model_name"],
        huggingfacehub_api_token=os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        ),
        max_new_tokens=model_config["max_new_tokens"],
        temperature=model_config["temperature"],
        top_p=model_config["top_p"]
    )

    chat_model = ChatHuggingFace(
        llm=endpoint
    )

    return chat_model