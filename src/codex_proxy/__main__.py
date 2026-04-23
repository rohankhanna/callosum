from __future__ import annotations

import argparse

import uvicorn

from codex_proxy.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="codex-proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
