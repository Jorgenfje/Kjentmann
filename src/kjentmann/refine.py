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
from kjentmann.queries import read_queries, render_view

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


def augment(img: np.ndarray, angle: float, zoom: float) -> np.ndarray:
    """The same view rotated about its centre and zoomed (zoom < 1 = closer)."""
    h, w = img.shape[:2]
    if angle == 0 and zoom == 1:
        return img
    return render_view(img, w / 2, h / 2, w, angle, zoom)


def coarse_candidates(
    embedder: Embedder,
    index,
    images: list[np.ndarray],
    k: int,
    rotations: tuple[float, ...] = (0.0,),
    zooms: tuple[float, ...] = (1.0,),
) -> np.ndarray:
    """Top-k tiles per image, trying several rotations and zooms of each image.

    DINOv2 fingerprints change when an image is rotated or seen from another
    height. Each variant is searched, and each tile keeps its best similarity.
    """
    best: list[dict[int, float]] = [{} for _ in images]
    for angle in rotations:
        for zoom in zooms:
            vecs = embedder.embed([augment(im, angle, zoom) for im in images])
            scores, idx = index.search(vecs, k * 2)
            for qi in range(len(images)):
                seen = best[qi]
                for sc, ti in zip(scores[qi], idx[qi], strict=True):
                    ti = int(ti)
                    if ti >= 0 and sc > seen.get(ti, -1.0):
                        seen[ti] = float(sc)
    out = np.full((len(images), k), -1, dtype=int)
    for qi, seen in enumerate(best):
        ranked = sorted(seen, key=seen.get, reverse=True)[:k]
        out[qi, : len(ranked)] = ranked
    return out


def match_candidates(
    cfg: Config, matcher: Matcher, img: np.ndarray, qi: int, ranked: list[int], get_window
) -> tuple:
    """Match one image against its candidates; retry rotated if nothing fits.

    Returns (inliers, tile index, verification, window, second best inliers).
    Rotating about the centre keeps the centre on the same ground point, so the
    position is read off the same way for every rotation.
    """
    best = None
    second = 0
    for angle in (0.0, *cfg.fine_rotations):
        if angle != 0 and best is not None and best[0] >= cfg.min_inliers:
            break
        view = augment(img, angle, 1.0)
        for ti in ranked:
            if ti < 0:
                continue
            win = get_window(ti)
            pa, pb = matcher.match(view, win.pixels, b_key=ti, a_key=(qi, angle))
            v = verify(pa, pb, cfg.ransac_px)
            s = transform_scale(v)
            inl = v.inliers if cfg.min_scale <= s <= cfg.max_scale else 0
            if best is None or inl > best[0]:
                second = best[0] if best else 0
                best = (inl, ti, v, win)
            elif inl > second:
                second = inl
    return (*best, second)


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
    cand = coarse_candidates(embedder, index, q_images, k, cfg.tta_rotations, cfg.tta_scales)

    # Map pixel coordinates of each tile centre, for cutting windows.
    inv = ~map_tf
    tile_px = [apply_transform(inv, t.center_x, t.center_y) for t in tiles]
    windows: dict[int, Window] = {}

    def get_window(ti: int) -> Window:
        if ti not in windows:
            windows[ti] = window_around(image, *tile_px[ti], win_px)
        return windows[ti]

    print(f"Finmatcher {len(queries)} testbilder mot topp {k} med {matcher.name} ...")
    rows = []
    t0 = time.perf_counter()
    for qi, q in enumerate(queries):
        img = q_images[qi]
        qh, qw = img.shape[:2]
        correct = containing_tiles(q, tiles, map_tf, size)
        ranked = [int(i) for i in cand[qi]]

        inliers, ti, v, win, second = match_candidates(cfg, matcher, img, qi, ranked, get_window)
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
        done = qi + 1
        if done % 10 == 0 or done == len(queries):
            spent = time.perf_counter() - t0
            left = spent / done * (len(queries) - done)
            print(f"  {done}/{len(queries)} bilder, ca. {left / 60:.1f} min igjen", flush=True)
    ms_per_query = (time.perf_counter() - t0) / max(len(queries), 1) * 1000

    summary = summarize(rows, cfg.min_inliers, k)
    summary.update(
        profile=cfg.profile,
        tta_rotations=list(cfg.tta_rotations),
        tta_zooms=list(cfg.tta_scales),
        fine_rotations=list(cfg.fine_rotations),
        breakdown=breakdown(rows, cfg.min_inliers, k),
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


def breakdown(rows: list[dict], min_inliers: int, k: int) -> list[dict]:
    """Scores split by how much each test image was rotated and scaled."""
    groups = [
        ("rotation < 5°", lambda r: abs(float(r["rotation_deg"])) < 5),
        ("rotation 5–10°", lambda r: 5 <= abs(float(r["rotation_deg"])) < 10),
        ("rotation ≥ 10°", lambda r: abs(float(r["rotation_deg"])) >= 10),
        ("scale < 0.9", lambda r: float(r["scale"]) < 0.9),
        ("scale 0.9–1.1", lambda r: 0.9 <= float(r["scale"]) <= 1.1),
        ("scale > 1.1", lambda r: float(r["scale"]) > 1.1),
    ]
    out = []
    for name, keep in groups:
        sub = [r for r in rows if keep(r)]
        if sub:
            s = summarize(sub, min_inliers, k)
            out.append(
                {
                    "group": name,
                    "images": len(sub),
                    f"coarse_hit@{k}": s[f"coarse_hit@{k}"],
                    "answered": s["answered"],
                    "within_100m_of_all": s["within_100m_of_all"],
                }
            )
    return out


def breakdown_table(s: dict, k: int) -> str:
    """Markdown for the breakdown, or empty when all images are plain crops."""
    groups = s.get("breakdown", [])
    if len(groups) <= 2:  # everything in one rotation and one scale group
        return ""
    lines = [
        f"| Test images | Count | Right tile in coarse top {k} | Answered | Within 100 m |",
        "|---|---|---|---|---|",
    ]
    for g in groups:
        lines.append(
            f"| {g['group']} | {g['images']} | {g[f'coarse_hit@{k}']:.0%} "
            f"| {g['answered']:.0%} | {g['within_100m_of_all']:.0%} |"
        )
    return "\n".join(lines)


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
    if len(cfg.tta_rotations) <= 1 and len(cfg.tta_scales) <= 1 and not cfg.fine_rotations:
        name = f"{name}_notta"
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
    parts = [results_table(summary, cfg.top_k), sweep_table(summary)]
    extra = breakdown_table(summary, cfg.top_k)
    if extra:
        parts.append(extra)
    md = "\n\n".join(parts) + "\n"
    (cfg.results_dir / f"refine_{name}.md").write_text(md, encoding="utf-8")
    build_refine_map(cfg, name, rows)
    print()
    print(md)
