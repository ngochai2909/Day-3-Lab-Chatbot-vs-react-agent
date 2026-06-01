import os
from src.core.llm_provider import LLMProvider


def create_provider() -> LLMProvider:
    """
    Build an LLMProvider based on the DEFAULT_PROVIDER env var.
    Options: openai | google | local
    """
    provider = os.getenv("DEFAULT_PROVIDER", "openai").lower()
    model = os.getenv("DEFAULT_MODEL", "gpt-4o")

    if provider == "openai":
        from src.core.openai_provider import OpenAIProvider
        return OpenAIProvider(model_name=model)

    if provider in ("google", "gemini"):
        from src.core.gemini_provider import GeminiProvider
        gem_model = model if "gemini" in model else "gemini-1.5-flash"
        return GeminiProvider(model_name=gem_model)

    if provider == "mimo":
        from src.core.mimo_provider import MiMoProvider
        mimo_model = model if "mimo" in model else "mimo-v2.5-pro"
        return MiMoProvider(model_name=mimo_model)

    if provider in ("router", "9router"):
        from src.core.router_provider import RouterProvider
        router_model = model or "cx/gpt-5.5"
        return RouterProvider(model_name=router_model)

    if provider == "local":
        from src.core.local_provider import LocalProvider
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/Phi-3-mini-4k-instruct-q4.gguf")
        return LocalProvider(model_path=model_path)

    raise ValueError(f"Unknown provider: {provider}. Use openai | google | mimo | local.")
