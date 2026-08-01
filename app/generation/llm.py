"""The swappable LLM provider: one client, two backends.

Both local Ollama and hosted providers speak the OpenAI-compatible
protocol, so the ONLY difference between them is base_url, model
name, and API key. The switch is pure configuration, no branching
logic anywhere else in the codebase.
"""

from openai import OpenAI

from app.config import settings


def get_client(provider: str | None = None) -> tuple[OpenAI, str]:
    """Return (client, model_name) for the requested or configured provider."""
    provider = provider or settings.llm_provider

    if provider == "local":
        client = OpenAI(base_url=settings.ollama_base_url, api_key="ollama")
        return client, settings.ollama_model

    if not settings.hosted_api_key:
        raise ValueError(
            "Hosted mode selected but HOSTED_API_KEY is not set. Add it to your .env file."
        )
    client = OpenAI(base_url=settings.hosted_base_url, api_key=settings.hosted_api_key)
    return client, settings.hosted_model


def complete(prompt: str, provider: str | None = None) -> str:
    """Send one prompt to the selected LLM and return its text reply.

    `provider` overrides the configured default ("local" | "hosted"),
    which is how the UI switch routes a single request without mutating
    global state.
    """
    client, model = get_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.temperature,
        max_tokens=settings.max_answer_tokens,
    )
    return (response.choices[0].message.content or "").strip()
