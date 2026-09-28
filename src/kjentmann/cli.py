"""Command line entry point: ``kjentmann <command>``."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the chosen step."""
    parser = argparse.ArgumentParser(prog="kjentmann", description="GPS-free visual positioning")
    parser.add_argument("command", choices=["fetch", "tiles", "map", "all"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    # Imported here so `kjentmann --help` stays fast.
    from kjentmann.config import load_config

    cfg = load_config(args.config)

    if args.command in ("fetch", "all"):
        from kjentmann.fetch import fetch

        fetch(cfg)
    if args.command in ("tiles", "all"):
        from kjentmann.tiles import build_tiles

        build_tiles(cfg)
    if args.command in ("map", "all"):
        from kjentmann.viz import build_map

        build_map(cfg)


if __name__ == "__main__":
    main()
