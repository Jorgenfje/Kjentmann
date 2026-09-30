"""v0.3: from "somewhere in this tile" to a position in metres.

For each test image:

1. Coarse search (DINOv2 + FAISS) returns the 5 most similar tiles.
2. Each candidate is matched point by point against a map window around it
   (a bit larger than the tile, so images that straddle tiles still fit).
3. RANSAC keeps only matches that agree on one similarity transform. The
   candidate with most agreeing matches (inliers) wins.
4. The transform maps the image centre into the map: that is the position.
5. Too few inliers, or an implausible scale, means "unknown" instead of a guess.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass

import numpy as np
import rasterio

from kjentmann.config import Config
from kjentmann.embed import Embedder
from kjentmann.evaluate import build_index, containing_tiles, load_images, read_tiles
from kjentmann.geo import apply_transform, to_lonlat
from kjentmann.match import Matcher, Verification, verify
from kjentmann.queries import read_queries

THRESHOLDS = (8, 10, 15, 20, 30, 50)


@dataclass(frozen=True)
class Window:
    """A square crop of the map around a candidate tile."""

    x0: int
    y0: int
    pixels: np.ndarray  # H x W x 3


def window_around(image: np.ndarray, cx: float, cy: float, size: int) -> Window:
    """Crop a square of ``size`` pixels centred on (cx, cy), shifted to stay inside."""
    _, h, w = image.shape
    size = min(size, h, w)
    x0 = int(round(cx - size / 2))
    y0 = int(round(cy - size / 2))
    x0 = max(0, min(x0, w - size))
    y0 = max(0, min(y0, h - size))
    crop = np.moveaxis(image[:3, y0 : y0 + size, x0 : x0 + size], 0, -1)
    return Window(x0, y0, np.ascontiguousarray(crop))


def transform_scale(v: Verification) -> float:
    """Scale factor of a similarity transform (1.0 = same pixel size)."""
    if v.transform is None:
        return 0.0
    return float(math.hypot(v.transform[0, 0], v.transform[1, 0]))


def refine(cfg: Config, embedder: Embedder, matcher: Matcher) -> dict:
    """Run coarse search + fine matching on every test image and score it."""
    tiles = read_tiles(cfg.tiles_csv)
    queries = read_queries(cfg.queries_csv)
    with rasterio.open(cfg.map_path) as src:
        image = src.read()
        map_tf, map_crs = src.transform, src.crs
    size = cfg.tile_size_px
    win_px = int(round(size * cfg.window_scale))
    k = cfg.top_k

    print(f"Grovsøk med {embedder.name} ...")
    tile_vecs = embedder.embed(load_images(cfg.tiles_dir, [t.tile_id for t in tiles]))
    index = build_index(tile_vecs)
    q_images = load_images(cfg.queries_dir, [q.query_id for q in queries])
    _, cand = index.search(embedder.embed(q_images), k)

    # Map pixel coordinates of each tile centre, for cutting windows.
    inv = ~map_tf
    tile_px = [apply_transform(inv, t.center_x, t.center_y) for t in tiles]
    windows: dict[int, Window] = {}

    print(f"Finmatcher {len(queries)} testbilder mot topp {k} med {matcher.name} ...")
    rows = []
    t0 = time.perf_counter()
    for qi, q in enumerate(queries):
        img = q_images[qi]
        qh, qw = img.shape[:2]
        correct = containing_tiles(q, tiles, map_tf, size)
        ranked = [int(i) for i in cand[qi]]

        best = None  # (inliers, tile index, verification, window)
        second = 0
        for ti in ranked:
            if ti not in windows:
                windows[ti] = window_around(image, *tile_px[ti], win_px)
            win = windows[ti]
            pa, pb = matcher.match(img, win.pixels, b_key=ti, a_key=qi)
            v = verify(pa, pb, cfg.ransac_px)
            s = transform_scale(v)
            inl = v.inliers if cfg.min_scale <= s <= cfg.max_scale else 0
            if best is None or inl > best[0]:
                second = best[0] if best else 0
                best = (inl, ti, v, win)
            elif inl > second:
                second = inl

        inliers, ti, v, win = best
        coarse = tiles[ranked[0]]
        row = {
            "query_id": q.query_id,
            "rotation_deg": q.rotation_deg,
            "scale": q.scale,
            "true_lat": q.center_lat,
            "true_lon": q.center_lon,
            "coarse_rank_of_correct": next(
                (r + 1 for r, i in enumerate(ranked) if i in correct), ""
            ),
            "coarse_error_m": round(
                float(math.hypot(coarse.center_x - q.center_x, coarse.center_y - q.center_y)), 1
            ),
            "chosen_tile": tiles[ti].tile_id,
            "chosen_is_correct": ti in correct,
            "inliers": inliers,
            "second_inliers": second,
            "est_scale": round(transform_scale(v), 3),
            "est_lat": "",
            "est_lon": "",
            "error_m": "",
        }
        pt = v.map_point(qw / 2, qh / 2)
        if pt is not None and inliers > 0:
            mx, my = apply_transform(map_tf, win.x0 + pt[0], win.y0 + pt[1])
            lon, lat = to_lonlat(map_crs, mx, my)
            row.update(
                est_lat=lat,
                est_lon=lon,
                error_m=round(float(math.hypot(mx - q.center_x, my - q.center_y)), 1),
            )
        rows.append(row)
    ms_per_query = (time.perf_counter() - t0) / max(len(queries), 1) * 1000

    summary = summarize(rows, cfg.min_inliers, k)
    summary.update(
        profile=cfg.profile,
        coarse=embedder.name,
        matcher=matcher.name,
        min_inliers=cfg.min_inliers,
        ms_per_query_matching=ms_per_query,
        sweep=[summarize(rows, t, k) for t in THRESHOLDS],
    )
    write_outputs(cfg, matcher.name, rows, summary)
    return summary


def summarize(rows: list[dict], min_inliers: int, k: int) -> dict:
    """Scores for one confidence threshold."""
    n = max(len(rows), 1)
    answered = [r for r in rows if r["inliers"] >= min_inliers and r["error_m"] != ""]
    errs = np.array([r["error_m"] for r in answered], dtype=float)

    def share(limit: float, of: int) -> float:
        return float((errs <= limit).sum() / of) if of else 0.0

    coarse_errs = np.array([r["coarse_error_m"] for r in rows], dtype=float)
    return {
        "threshold": min_inliers,
        "queries": len(rows),
        "coarse_hit@1": sum(r["coarse_rank_of_correct"] == 1 for r in rows) / n,
        f"coarse_hit@{k}": sum(r["coarse_rank_of_correct"] != "" for r in rows) / n,
        "coarse_median_error_m": float(np.median(coarse_errs)) if len(rows) else 0.0,
        "answered": len(answered) / n,
        "median_error_m": float(np.median(errs)) if len(errs) else float("nan"),
        "within_50m_of_all": share(50, n),
        "within_100m_of_all": share(100, n),
        "within_100m_of_answered": share(100, len(answered)),
        "wrong_answers_over_500m": float((errs > 500).sum() / n),
    }


def results_table(s: dict, k: int) -> str:
    """Markdown summary for the README."""
    return "\n".join(
        [
            "| Step | Result |",
            "|---|---|",
            f"| Coarse search ({s['coarse']}): right tile first | {s['coarse_hit@1']:.0%} |",
            f"| Coarse search: right tile in top {k} | {s[f'coarse_hit@{k}']:.0%} |",
            f"| Fine matching ({s['matcher']}): answered (≥ {s['min_inliers']} inliers) "
            f"| {s['answered']:.0%} |",
            f"| Median error when answered | {s['median_error_m']:.0f} m |",
            f"| Within 100 m, of all images | {s['within_100m_of_all']:.0%} |",
            f"| Within 100 m, of answered images | {s['within_100m_of_answered']:.0%} |",
            f"| Wrong answers (> 500 m), of all images | {s['wrong_answers_over_500m']:.0%} |",
            f"| Matching time per image | {s['ms_per_query_matching']:.0f} ms |",
        ]
    )


def sweep_table(s: dict) -> str:
    """How the confidence threshold trades coverage against correctness."""
    lines = [
        "| Min. inliers | Answered | Within 100 m (of answered) | Wrong > 500 m (of all) |",
        "|---|---|---|---|",
    ]
    for t in s["sweep"]:
        lines.append(
            f"| {t['threshold']} | {t['answered']:.0%} | {t['within_100m_of_answered']:.0%} "
            f"| {t['wrong_answers_over_500m']:.0%} |"
        )
    return "\n".join(lines)


def write_outputs(cfg: Config, name: str, rows: list[dict], summary: dict) -> None:
    """CSV per test image, JSON and Markdown summary, and a results map."""
    from kjentmann.viz import build_refine_map

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.results_dir / f"refine_{name}_queries.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (cfg.results_dir / f"refine_{name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    md = results_table(summary, cfg.top_k) + "\n\n" + sweep_table(summary) + "\n"
    (cfg.results_dir / f"refine_{name}.md").write_text(md, encoding="utf-8")
    build_refine_map(cfg, name, rows)
    print()
    print(md)
