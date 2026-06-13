import os

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()


def load_llm(model_config):

    return HuggingFaceEndpoint(
        repo_id=model_config["model_name"],
        huggingfacehub_api_token=os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        ),
        temperature=model_config["temperature"],
        max_new_tokens=model_config["max_new_tokens"],
        top_p=model_config["top_p"],
        repetition_penalty=model_config[
            "repetition_penalty"
        ]
    )