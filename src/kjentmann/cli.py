"""Command line entry point: ``kjentmann <command>``.

Steps, in order:
    fetch        download the reference map (Sentinel-2)
    tiles        cut the map into tiles with known positions
    map          interactive map of the tile grid
    fetch-query  download a scene from another date
    queries      cut test images with ground truth from that scene
    evaluate     index the tiles, search every test image, score the result

Shortcuts:
    all          fetch + tiles + map          (v0.1)
    v02          fetch-query + queries + evaluate
"""

from __future__ import annotations

import argparse

STEPS = ["fetch", "tiles", "map", "fetch-query", "queries", "evaluate"]
SHORTCUTS = {"all": ["fetch", "tiles", "map"], "v02": ["fetch-query", "queries", "evaluate"]}


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the chosen steps."""
    parser = argparse.ArgumentParser(
        prog="kjentmann",
        description="GPS-free visual positioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Steps, in order:")[1],
    )
    parser.add_argument("command", choices=STEPS + list(SHORTCUTS))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--embedders",
        default="pixel,dinov2",
        help="comma-separated list for 'evaluate' (default: pixel,dinov2)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="use random DINOv2 weights (testing only, no download)",
    )
    args = parser.parse_args(argv)

    # Imported here so `kjentmann --help` stays fast.
    from kjentmann.config import load_config

    cfg = load_config(args.config)
    for step in SHORTCUTS.get(args.command, [args.command]):
        run_step(step, cfg, args)


def run_step(step: str, cfg, args) -> None:
    """Run one pipeline step."""
    if step == "fetch":
        from kjentmann.fetch import fetch

        fetch(cfg)
    elif step == "tiles":
        from kjentmann.tiles import build_tiles

        build_tiles(cfg)
    elif step == "map":
        from kjentmann.viz import build_map

        build_map(cfg)
    elif step == "fetch-query":
        from kjentmann.fetch import fetch_query_scene

        fetch_query_scene(cfg)
    elif step == "queries":
        from kjentmann.queries import build_queries

        build_queries(cfg)
    elif step == "evaluate":
        from kjentmann.embed import make_embedder
        from kjentmann.evaluate import evaluate

        names = [n.strip() for n in args.embedders.split(",") if n.strip()]
        embedders = [make_embedder(n, cfg, pretrained=not args.no_pretrained) for n in names]
        evaluate(cfg, embedders)


if __name__ == "__main__":
    main()
