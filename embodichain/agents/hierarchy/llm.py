import os
from langchain_openai import AzureChatOpenAI

# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------

# Clear proxy if not needed (optional, can be set via environment variables)

os.environ["ALL_PROXY"] = ""
os.environ["all_proxy"] = ""

# Proxy configuration (optional, uncomment if needed)
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# API version (optional, defaults to "2024-10-21" if not set)
# os.environ["OPENAI_API_VERSION"] = "2024-10-21"

# Note: AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set via environment variables
# Example in bash:
#   export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
#   export AZURE_OPENAI_API_KEY="your-api-key"

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
