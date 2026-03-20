import os

from langchain_openai import AzureChatOpenAI, ChatOpenAI


def _get_llm():
    """
    Create a single shared LangChain LLM client.

    Supported providers:
    - vllm: OpenAI-compatible endpoint (ChatOpenAI + base_url)
    - azure: Azure OpenAI (AzureChatOpenAI)
    """
    provider = os.getenv("LLM_PROVIDER", "vllm").strip().lower()
    temperature_env = float(os.getenv("LLM_TEMPERATURE", "0"))

    if provider == "azure":
        # Some Azure deployments reject temperature=0 (model-dependent).
        # If temperature==0, we omit the parameter and let Azure/model default.
        kwargs = {
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            "azure_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        }
        if temperature_env != 0.0:
            kwargs["temperature"] = temperature_env
        return AzureChatOpenAI(**kwargs)

    # Default: vLLM / OpenAI-compatible endpoint
    return ChatOpenAI(
        base_url=os.getenv("VLLM_BASE_URL"),
        api_key=os.getenv("VLLM_API_KEY"),
        model=os.getenv("VLLM_MODEL_NAME"),
        # PDF constraint: deterministic benchmark runs.
        temperature=0.0,
    )


llm = _get_llm()

