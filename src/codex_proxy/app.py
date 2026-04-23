from __future__ import annotations

from fastapi import FastAPI

from codex_proxy import __version__


def create_app() -> FastAPI:
    app = FastAPI(title="codex-proxy", version=__version__)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
