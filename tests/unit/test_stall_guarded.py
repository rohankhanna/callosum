"""Tests for the behavioral stall guard that replaced the byte-size cap.

`stall_guarded` re-yields a streaming upstream but fails fast and `transient`
when the upstream hangs — distinguishing the cold-load/prefill wait for the
first byte from a mid-stream stall.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from callosum.backends._http import stall_guarded
from callosum.errors import BackendError


async def _from_list(items: list[bytes]) -> AsyncIterator[bytes]:
    for it in items:
        yield it


async def test_passes_through_all_items_when_no_stall() -> None:
    out = [
        chunk
        async for chunk in stall_guarded(
            _from_list([b"a", b"b", b"c"]),
            first_item_timeout_s=1.0,
            idle_timeout_s=1.0,
        )
    ]
    assert out == [b"a", b"b", b"c"]


async def test_empty_source_completes_cleanly() -> None:
    out = [
        chunk
        async for chunk in stall_guarded(
            _from_list([]),
            first_item_timeout_s=1.0,
            idle_timeout_s=1.0,
        )
    ]
    assert out == []


async def test_first_byte_timeout_raises_transient() -> None:
    async def slow_first() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.2)
        yield b"too-late"

    with pytest.raises(BackendError) as exc_info:
        async for _ in stall_guarded(
            slow_first(),
            first_item_timeout_s=0.05,
            idle_timeout_s=1.0,
            what="test-model",
        ):
            pass
    assert exc_info.value.classification == "transient"
    assert "before first byte" in exc_info.value.message
    assert "test-model" in exc_info.value.message


async def test_mid_stream_idle_timeout_raises_transient() -> None:
    async def stall_after_first() -> AsyncIterator[bytes]:
        yield b"first"
        await asyncio.sleep(0.2)
        yield b"never-arrives-in-time"

    seen: list[bytes] = []
    with pytest.raises(BackendError) as exc_info:
        async for chunk in stall_guarded(
            stall_after_first(),
            first_item_timeout_s=1.0,
            idle_timeout_s=0.05,
        ):
            seen.append(chunk)
    assert seen == [b"first"]  # the first byte streamed before the stall
    assert exc_info.value.classification == "transient"
    assert "mid-stream" in exc_info.value.message
