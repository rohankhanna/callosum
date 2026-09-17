# API stability

Callosum uses a rolling `main` release model.

- The HTTP surface is `/health`, `/status`, `/v1/responses`,
  `/v1/chat/completions`, and related operator endpoints.
- The Python `Backend` protocol is the supported internal extension point.
- The TOML configuration schema is versioned in `pyproject.toml`.

Compatibility promises:

- Public HTTP routes will not be removed without a deprecation period.
- The `Backend` protocol may gain optional methods, but existing required
  methods will remain stable.
- Configuration fields will not be silently repurposed.

Breaking changes should be called out in `CHANGELOG.md` before merging to
`main`.
