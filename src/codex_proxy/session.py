from __future__ import annotations


class SessionRegistry:
    """In-memory session_id -> backend_id binding used by the sticky policy.

    Safe for single-event-loop use (no locking). Bindings are process-local
    and do not survive restart; that is intentional for v1.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, str] = {}

    def get(self, session_id: str) -> str | None:
        return self._bindings.get(session_id)

    def set(self, session_id: str, backend_id: str) -> None:
        self._bindings[session_id] = backend_id

    def clear(self, session_id: str) -> None:
        self._bindings.pop(session_id, None)

    def snapshot(self) -> dict[str, str]:
        return dict(self._bindings)
