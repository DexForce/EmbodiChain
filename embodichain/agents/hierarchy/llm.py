import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------

os.environ["ALL_PROXY"] = ""
os.environ["all_proxy"] = ""
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["OPENAI_API_VERSION"] = "2024-10-21"
os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_ENDPOINT_HERE"

# ------------------------------------------------------------------------------
# LLM factory
# ------------------------------------------------------------------------------


def create_llm(*, temperature=0.0, model="gpt-4o"):
    return ChatOpenAI(
        temperature=temperature,
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


# ------------------------------------------------------------------------------
# LLM instances
# ------------------------------------------------------------------------------

task_llm = create_llm(temperature=0.0, model="gpt-4o")
code_llm = create_llm(temperature=0.0, model="gemini-2.5-flash-lite")
validation_llm = create_llm(temperature=0.0, model="gemini-3-flash-preview")
view_selection_llm = create_llm(temperature=0.0, model="gemini-2.5-flash-lite")
