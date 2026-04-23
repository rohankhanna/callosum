from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from codex_proxy import __version__
from codex_proxy.backend import Backend
from codex_proxy.selector import select


def create_app(*, backends: Sequence[Backend] = ()) -> FastAPI:
    backends_list: list[Backend] = list(backends)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            for backend in backends_list:
                await backend.aclose()

    app = FastAPI(title="codex-proxy", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/chat/completions")
    async def chat_completions(body: dict[str, Any]) -> dict[str, Any]:
        model = body.get("model")
        if not isinstance(model, str):
            raise HTTPException(status_code=400, detail="'model' must be a string")
        backend = await select(backends_list, model=model)
        if backend is None:
            raise HTTPException(status_code=503, detail=f"no viable backend for model {model!r}")
        return await backend.chat_completions(body)

    return app
