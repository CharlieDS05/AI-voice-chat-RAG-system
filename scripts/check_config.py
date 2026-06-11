"""Sanity check: confirm configuration loads and the switch is readable.
Run from the project root with:  python -m scripts.check_config
"""

from app.config import settings


def main() -> None:
    print("✅ Config loaded successfully\n")
    print(f"LLM provider     : {settings.llm_provider}")
    print(f"Ollama base URL  : {settings.ollama_base_url}")
    print(f"Ollama model     : {settings.ollama_model}")
    print(f"Hosted base URL  : {settings.hosted_base_url}")
    print(f"Hosted model     : {settings.hosted_model}")
    # We print only WHETHER a key is set — never the key itself.
    # We print only WHETHER a key is set — never the key itself.
    key_status = "set" if settings.hosted_api_key else "not set (fine for local mode)"
    print(f"Hosted API key   : {key_status}")


if __name__ == "__main__":
    main()
