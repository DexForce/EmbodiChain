import os
from langchain_openai import AzureChatOpenAI

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------

os.environ["ALL_PROXY"] = ""
os.environ["all_proxy"] = ""
#os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
#os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["OPENAI_API_VERSION"] = "2024-10-21"
os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_ENDPOINT_HERE"
os.environ["AZURE_OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model="gpt-4o"):
    return AzureChatOpenAI(
        temperature=temperature,
        model=model,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION", "2024-10-21"),
    )


# ------------------------------------------------------------------------------
# LLM instances
# ------------------------------------------------------------------------------

task_llm = create_llm(temperature=0.0, model="gpt-4o")
code_llm = create_llm(temperature=0.0, model="gpt-4o")
validation_llm = create_llm(temperature=0.0, model="gpt-4o")
view_selection_llm = create_llm(temperature=0.0, model="gpt-4o")
