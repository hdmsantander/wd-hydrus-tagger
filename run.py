#!/usr/bin/env python3
"""Entry point for WD Hydrus Tagger."""

import uvicorn

from backend.config import load_config


def main():
    config = load_config()
    uvicorn.run(
        "backend.app:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
