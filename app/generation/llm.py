"""The swappable LLM provider: one client, two backends.

Both local Ollama and hosted providers speak the OpenAI-compatible
protocol, so the ONLY difference between them is base_url, model
name, and API key. The switch is pure configuration, no branching
logic anywhere else in the codebase.
"""

from openai import OpenAI

from app.config import settings


def get_client() -> tuple[OpenAI, str]:
    """Return (client, model_name) for the configured provider."""
    if settings.llm_provider == "local":
        client = OpenAI(
            base_url=settings.ollama_base_url,
            api_key="ollama",  # Ollama ignores keys; the client requires one
        )
        return client, settings.ollama_model

    if not settings.hosted_api_key:
        raise ValueError(
            "LLM_PROVIDER is 'hosted' but HOSTED_API_KEY is not set. Add it to your .env file."
        )
    client = OpenAI(
        base_url=settings.hosted_base_url,
        api_key=settings.hosted_api_key,
    )
    return client, settings.hosted_model


def complete(system_or_user_prompt: str) -> str:
    """Send one prompt to the configured LLM and return its text reply."""
    client, model = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": system_or_user_prompt}],
        temperature=settings.temperature,
        max_tokens=settings.max_answer_tokens,
    )
    return (response.choices[0].message.content or "").strip()
