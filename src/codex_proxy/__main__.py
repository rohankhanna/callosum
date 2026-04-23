from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from codex_proxy.app import create_app
from codex_proxy.config import build_backends, load_config


def _default_config_path() -> Path:
    return Path("~/.config/codex-proxy/config.toml").expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(prog="codex-proxy")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"config not found: {args.config}")

    cfg = load_config(args.config)
    backends = build_backends(cfg)
    host = args.host if args.host is not None else cfg.server.host
    port = args.port if args.port is not None else cfg.server.port
    uvicorn.run(create_app(backends=backends), host=host, port=port)


if __name__ == "__main__":
    main()
