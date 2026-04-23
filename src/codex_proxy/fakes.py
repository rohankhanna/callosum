from __future__ import annotations

from codex_proxy.backend import BackendKind, HealthStatus, UsageSnapshot


class InMemoryFakeBackend:
    """Deterministic Backend implementation used by tests and local dry-runs."""

    def __init__(
        self,
        *,
        id: str,
        advertised_models: frozenset[str],
        kind: BackendKind = "openai_api_key",
        health: HealthStatus | None = None,
        usage: UsageSnapshot | None = None,
    ) -> None:
        self.id = id
        self.kind: BackendKind = kind
        self.advertised_models = advertised_models
        self._health = health if health is not None else HealthStatus(available=True, reason="ok")
        self._usage = (
            usage
            if usage is not None
            else UsageSnapshot(
                remaining_fraction=1.0,
                cooldown_until_ts=None,
                weekly_exhausted=False,
                probed_at_ts=0.0,
            )
        )

    async def health(self) -> HealthStatus:
        return self._health

    async def usage_snapshot(self) -> UsageSnapshot:
        return self._usage

    def set_health(self, health: HealthStatus) -> None:
        self._health = health

    def set_usage(self, usage: UsageSnapshot) -> None:
        self._usage = usage
