"""Run the combined API + UI server.

Usage: python -m scripts.run_api
"""

import uvicorn

from app.config import settings


def main() -> None:
    print(
        f"Effective config: provider={settings.llm_provider}, "
        f"auth={'on' if settings.api_key else 'OFF'}, "
        f"rate_limit={settings.rate_limit}"
    )
    uvicorn.run(
        "app.api.server:app",
        host=settings.server_name,
        port=settings.api_port,
    )


if __name__ == "__main__":
    main()
