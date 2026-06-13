from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import fastapi.testclient
import httpx


class AsyncBackedClient:
    """Sync test facade backed by httpx.ASGITransport.

    Starlette's synchronous TestClient can hang in this environment. This
    facade preserves the small TestClient surface used by the test suite while
    avoiding the synchronous ASGI portal.
    """

    __test__ = False

    def __init__(self, app: Any, *, base_url: str = "http://testserver", **_: Any) -> None:
        self.app = app
        self.base_url = base_url
        self.cookies = httpx.Cookies()
        self.headers = httpx.Headers()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None
        self._lifespan_receive: asyncio.Queue[dict[str, Any]] | None = None
        self._lifespan_send: asyncio.Queue[dict[str, Any]] | None = None
        self._lifespan_task: asyncio.Task[None] | None = None

    def __enter__(self) -> AsyncBackedClient:
        self._ensure_started(lifespan=True)
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        if self._loop is not None and not self._loop.is_running():
            self.close()

    def close(self) -> None:
        if self._loop is None:
            return
        if self._client is not None:
            self._loop.run_until_complete(self._client.aclose())
            self._client = None
        if self._lifespan_task is not None:
            self._loop.run_until_complete(self._shutdown_lifespan())
        self._loop.close()
        self._loop = None

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        self._ensure_started(lifespan=False)
        assert self._loop is not None
        assert self._client is not None
        response = self._loop.run_until_complete(self._client.request(method, url, **kwargs))
        self.cookies.update(response.cookies)
        return response

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    @contextmanager
    def stream(self, method: str, url: str, **kwargs: Any) -> Iterator[httpx.Response]:
        response = self.request(method, url, **kwargs)
        try:
            yield response
        finally:
            assert self._loop is not None
            self._loop.run_until_complete(response.aclose())

    def _ensure_started(self, *, lifespan: bool) -> None:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            if lifespan:
                self._loop.run_until_complete(self._startup_lifespan())
            transport = httpx.ASGITransport(app=self.app)
            self._client = httpx.AsyncClient(transport=transport, base_url=self.base_url, cookies=self.cookies)

    async def _startup_lifespan(self) -> None:
        self._lifespan_receive = asyncio.Queue()
        self._lifespan_send = asyncio.Queue()
        await self._lifespan_receive.put({"type": "lifespan.startup"})
        self._lifespan_task = asyncio.create_task(
            self.app(
                {"type": "lifespan", "asgi": {"version": "3.0", "spec_version": "2.0"}, "state": {}},
                self._receive_lifespan,
                self._send_lifespan,
            )
        )
        message = await self._lifespan_send.get()
        if message["type"] == "lifespan.startup.failed":
            raise RuntimeError(message.get("message", "ASGI lifespan startup failed"))

    async def _shutdown_lifespan(self) -> None:
        assert self._lifespan_receive is not None
        assert self._lifespan_send is not None
        assert self._lifespan_task is not None
        await self._lifespan_receive.put({"type": "lifespan.shutdown"})
        message = await asyncio.wait_for(self._lifespan_send.get(), timeout=5.0)
        if message["type"] == "lifespan.shutdown.failed":
            raise RuntimeError(message.get("message", "ASGI lifespan shutdown failed"))
        await asyncio.wait_for(self._lifespan_task, timeout=5.0)
        self._lifespan_task = None

    async def _receive_lifespan(self) -> dict[str, Any]:
        assert self._lifespan_receive is not None
        return await self._lifespan_receive.get()

    async def _send_lifespan(self, message: dict[str, Any]) -> None:
        assert self._lifespan_send is not None
        await self._lifespan_send.put(message)


fastapi.testclient.TestClient = AsyncBackedClient
