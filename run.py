#!/usr/bin/env python3
"""Entry point for WD Hydrus Tagger."""

import os
import sys
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parent
    from backend.logging_setup import configure_logging, parse_server_args

    args = parse_server_args(sys.argv[1:])
    # Keep env aligned with CLI so lifespan / subprocesses match the chosen level
    os.environ["LOG_LEVEL"] = args.log_level.strip()
    os.environ["WD_TAGGER_LOG_LEVEL"] = args.log_level.strip()
    log_path = configure_logging(
        args.log_level,
        log_file=args.log_file,
        repo_root=repo,
        reset=True,
    )

    import uvicorn

    import logging

    from backend.config import load_config
    from backend.listen_hints import log_startup_listen_hint, print_startup_listen_hint
    from backend.runtime_linux import uvicorn_loop_setting

    config = load_config()

    boot = logging.getLogger("wd_tagger.bootstrap")
    boot.info(
        "startup run_id=%s log_file=%s",
        os.environ.get("WD_TAGGER_RUN_ID", "—"),
        log_path,
    )
    log_startup_listen_hint(boot, config.host, config.port)
    print_startup_listen_hint(config.host, config.port, stream=sys.stderr)
    uvicorn.run(
        "backend.app:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level=args.log_level.lower(),
        loop=uvicorn_loop_setting(),
    )


if __name__ == "__main__":
    main()
