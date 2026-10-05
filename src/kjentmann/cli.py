"""Command line entry point: ``kjentmann <command>``.

Locate a photo (anywhere in the world):
    kjentmann locate photo.jpg --near "Gardermoen" --altitude 3000
    kjentmann locate photo.jpg --near 59.58,11.16 --altitude 600 --radius 10

Steps, in order:
    fetch        download the reference map (Sentinel-2)
    tiles        cut the map into tiles with known positions
    map          interactive map of the tile grid
    fetch-query  download a scene from another date
    queries      cut test images with ground truth from that scene
    evaluate     index the tiles, search every test image, score the result
    refine       coarse search + point matching: position in metres (v0.3)
    diagnose     coarse search on blur / rotation / scale test images, one at a time
    navigate     search inside an uncertainty circle (2, 5, 10 km), no coarse step (v0.4)
    seasons      spring / autumn / winter test images against the June map (v0.5)
    spoof        detect spoofed GPS: statistics and a flight demo with a map

Shortcuts:
    all          fetch + tiles + map          (v0.1)
    v02          fetch-query + queries + evaluate
    v03          refine

Harder test images (rotation, scale, blur, noise):
    kjentmann queries --profile realistic
    kjentmann refine  --profile realistic
"""

from __future__ import annotations

import argparse

STEPS = [
    "fetch",
    "tiles",
    "map",
    "fetch-query",
    "queries",
    "evaluate",
    "refine",
    "diagnose",
    "navigate",
    "seasons",
    "spoof",
    "locate",
]
SHORTCUTS = {
    "all": ["fetch", "tiles", "map"],
    "v02": ["fetch-query", "queries", "evaluate"],
    "v03": ["refine"],
}


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the chosen steps."""
    parser = argparse.ArgumentParser(
        prog="kjentmann",
        description="GPS-free visual positioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n", 1)[1],
    )
    parser.add_argument("command", choices=STEPS + list(SHORTCUTS))
    parser.add_argument("photo", nargs="?", help="photo for 'locate'")
    parser.add_argument("--near", help="rough position for 'locate': place name or lat,lon")
    parser.add_argument("--altitude", type=float, help="approximate altitude in metres")
    parser.add_argument("--radius", type=float, help="search radius in km (default: config)")
    parser.add_argument("--fov", type=float, help="camera field of view in degrees")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--embedders",
        default="pixel,dinov2",
        help="comma-separated list for 'evaluate' (default: pixel,dinov2)",
    )
    parser.add_argument(
        "--coarse",
        default="dinov2",
        help="embedder for coarse search in 'refine' (default: dinov2)",
    )
    parser.add_argument(
        "--matcher", default=None, help="lightglue or sift for 'refine' (default: from config)"
    )
    parser.add_argument(
        "--profile",
        default="easy",
        help="test-image profile from config.yaml, e.g. easy or realistic (default: easy)",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="refine without extra rotations/zooms (for comparison)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="use random DINOv2 weights (testing only, no download)",
    )
    args = parser.parse_args(argv)

    # Imported here so `kjentmann --help` stays fast.
    from kjentmann.config import load_config

    cfg = load_config(args.config).with_profile(args.profile)
    if args.no_tta:
        from dataclasses import replace

        cfg = replace(cfg, tta_rotations=(0.0,), tta_scales=(1.0,), fine_rotations=())
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
    elif step == "refine":
        from kjentmann.embed import make_embedder
        from kjentmann.match import make_matcher
        from kjentmann.refine import refine

        embedder = make_embedder(args.coarse, cfg, pretrained=not args.no_pretrained)
        matcher = make_matcher(args.matcher or cfg.matcher, cfg.max_keypoints, cfg.upscale)
        refine(cfg, embedder, matcher)
    elif step == "diagnose":
        from kjentmann.diagnose import diagnose
        from kjentmann.embed import make_embedder

        diagnose(cfg, make_embedder(args.coarse, cfg, pretrained=not args.no_pretrained))
    elif step == "navigate":
        from kjentmann.match import make_matcher
        from kjentmann.navigate import navigate

        navigate(cfg, make_matcher(args.matcher or cfg.matcher, cfg.max_keypoints, cfg.upscale))
    elif step == "seasons":
        from kjentmann.match import make_matcher
        from kjentmann.seasons import seasons

        seasons(cfg, make_matcher(args.matcher or cfg.matcher, cfg.max_keypoints, cfg.upscale))
    elif step == "spoof":
        from kjentmann.match import make_matcher
        from kjentmann.spoof import spoof

        spoof(cfg, make_matcher(args.matcher or cfg.matcher, cfg.max_keypoints, cfg.upscale))
    elif step == "locate":
        if not (args.photo and args.near and args.altitude):
            raise SystemExit(
                'Usage: kjentmann locate photo.jpg --near "place or lat,lon" --altitude 3000'
            )
        from kjentmann.locate import locate_cli
        from kjentmann.match import make_matcher

        matcher = make_matcher(args.matcher or cfg.matcher, cfg.max_keypoints, cfg.upscale)
        locate_cli(cfg, matcher, args.photo, args.near, args.altitude, args.radius, args.fov)


if __name__ == "__main__":
    main()
