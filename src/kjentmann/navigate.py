"""v0.4: search only where the aircraft could be.

Without GPS an aircraft still has an estimate of its position from inertial
navigation (dead reckoning). The estimate drifts, so the true position lies
somewhere within an uncertainty circle of radius R around it.

Instead of a global coarse search, Kjentmann matches the camera image point by
point against map windows inside that circle, nearest first, and stops as soon
as one match is confident. The coarse DINOv2 search is not used: under blur and
noise it misses the right area too often (see ``kjentmann diagnose``), while
point matching with geometric verification stays reliable.

For evaluation, each test image gets a simulated inertial estimate: the true
position shifted by up to R in a random direction.
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
from kjentmann.evaluate import load_images, read_tiles
from kjentmann.geo import apply_transform, to_lonlat
from kjentmann.match import Matcher, verify
from kjentmann.queries import read_queries
from kjentmann.refine import augment, transform_scale, window_around


@dataclass(frozen=True)
class Candidate:
    """A map window that can be searched, and where its centre lies."""

    key: tuple
    cx_px: float
    cy_px: float
    cx_m: float
    cy_m: float


def search_windows(cfg: Config, tiles, map_tf) -> list[Candidate]:
    """Every other tile centre in each direction (stride = one tile).

    With windows of ``nav_window_scale`` x tile (2 x by default) and a stride of
    one tile, any test image of tile size fits fully inside at least one window.
    """
    inv = ~map_tf
    out = []
    for t in tiles:
        row, col = int(t.tile_id[1:4]), int(t.tile_id[6:9])
        if row % 2 or col % 2:
            continue
        px, py = apply_transform(inv, t.center_x, t.center_y)
        out.append(Candidate(("nav", t.tile_id), px, py, t.center_x, t.center_y))
    return out


def simulated_priors(n: int, seed: int) -> np.ndarray:
    """Unit offsets, uniform over a disc, scaled by R later.

    The same directions are reused for every radius, so results across radii
    differ only in how far off the estimate is.
    """
    rng = np.random.default_rng(seed + 7)
    angle = rng.uniform(0, 2 * math.pi, n)
    dist = np.sqrt(rng.uniform(0, 1, n))  # sqrt -> uniform over the disc area
    return np.stack([np.cos(angle) * dist, np.sin(angle) * dist], axis=1)


def locate(
    cfg: Config,
    matcher: Matcher,
    img: np.ndarray,
    qi: int,
    prior_xy: tuple[float, float],
    radius_m: float,
    candidates: list[Candidate],
    image: np.ndarray,
    win_px: int,
    pixel_m: float,
    windows: dict,
):
    """Search windows inside the circle, nearest first, until one is confident.

    Returns (best inliers, best verification, best window, windows tried).
    """
    reach = radius_m + win_px * pixel_m / math.sqrt(2)  # window may overlap the circle
    px, py = prior_xy
    near = sorted(
        (c for c in candidates if math.hypot(c.cx_m - px, c.cy_m - py) <= reach),
        key=lambda c: math.hypot(c.cx_m - px, c.cy_m - py),
    )
    best = (0, None, None)
    tried = 0
    for angle in (0.0, *cfg.fine_rotations):
        if angle != 0 and best[0] >= cfg.min_inliers:
            break
        view = augment(img, angle, 1.0)
        for c in near:
            if c.key not in windows:
                windows[c.key] = window_around(image, c.cx_px, c.cy_px, win_px)
            win = windows[c.key]
            pa, pb = matcher.match(view, win.pixels, b_key=c.key, a_key=(qi, angle))
            v = verify(pa, pb, cfg.ransac_px)
            s = transform_scale(v)
            inl = v.inliers if cfg.min_scale <= s <= cfg.max_scale else 0
            tried += 1
            if inl > best[0]:
                best = (inl, v, win)
            if inl >= cfg.early_stop_inliers:
                return (*best, tried)
    return (*best, tried)


def navigate(cfg: Config, matcher: Matcher, prefix: str = "navigate") -> list[dict]:
    """Evaluate the uncertainty-circle search for each configured radius."""
    tiles = read_tiles(cfg.tiles_csv)
    queries = read_queries(cfg.queries_csv)
    with rasterio.open(cfg.map_path) as src:
        image = src.read()
        map_tf, map_crs = src.transform, src.crs
    pixel_m = abs(map_tf.a)
    win_px = int(round(cfg.tile_size_px * cfg.nav_window_scale))
    candidates = search_windows(cfg, tiles, map_tf)
    q_images = load_images(cfg.queries_dir, [q.query_id for q in queries])
    unit = simulated_priors(len(queries), cfg.query_seed)
    windows: dict = {}

    summaries = []
    for radius_km in cfg.nav_radii_km:
        r_m = radius_km * 1000
        print(f"\nRadius {radius_km:g} km: {len(queries)} testbilder ...")
        rows = []
        t0 = time.perf_counter()
        for qi, q in enumerate(queries):
            prior = (q.center_x + unit[qi, 0] * r_m, q.center_y + unit[qi, 1] * r_m)
            inliers, v, win, tried = locate(
                cfg,
                matcher,
                q_images[qi],
                qi,
                prior,
                r_m,
                candidates,
                image,
                win_px,
                pixel_m,
                windows,
            )
            row = {
                "query_id": q.query_id,
                "rotation_deg": q.rotation_deg,
                "scale": q.scale,
                "true_lat": q.center_lat,
                "true_lon": q.center_lon,
                "prior_error_m": round(float(math.hypot(*(unit[qi] * r_m))), 1),
                "windows_tried": tried,
                "inliers": inliers,
                "est_lat": "",
                "est_lon": "",
                "error_m": "",
            }
            qh, qw = q_images[qi].shape[:2]
            pt = v.map_point(qw / 2, qh / 2) if v is not None else None
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
            if done % 20 == 0 or done == len(queries):
                spent = time.perf_counter() - t0
                left = spent / done * (len(queries) - done)
                print(f"  {done}/{len(queries)} bilder, ca. {left / 60:.1f} min igjen", flush=True)
        seconds = (time.perf_counter() - t0) / max(len(queries), 1)
        s = summarize_radius(rows, cfg.min_inliers, radius_km, seconds)
        summaries.append(s)
        write_rows(cfg, f"{prefix}_{matcher.name}", radius_km, rows)
        if radius_km == cfg.nav_map_radius_km:
            from kjentmann.viz import build_refine_map

            build_refine_map(
                cfg,
                matcher.name,
                rows,
                filename=f"{prefix}_{matcher.name}_{radius_km:g}km_map.html",
            )

    write_summary(cfg, f"{prefix}_{matcher.name}", summaries)
    return summaries


def summarize_radius(rows: list[dict], min_inliers: int, radius_km: float, seconds: float) -> dict:
    """Scores for one radius."""
    n = max(len(rows), 1)
    answered = [r for r in rows if r["inliers"] >= min_inliers and r["error_m"] != ""]
    errs = np.array([r["error_m"] for r in answered], dtype=float)
    return {
        "radius_km": radius_km,
        "queries": len(rows),
        "answered": len(answered) / n,
        "median_error_m": float(np.median(errs)) if len(errs) else float("nan"),
        "within_100m_of_all": float((errs <= 100).sum() / n),
        "within_100m_of_answered": float((errs <= 100).mean()) if len(errs) else 0.0,
        "wrong_answers_over_500m": float((errs > 500).sum() / n),
        "mean_windows_tried": float(np.mean([r["windows_tried"] for r in rows])) if rows else 0.0,
        "seconds_per_image": seconds,
    }


def results_table(summaries: list[dict], matcher: str, min_inliers: int) -> str:
    """Markdown table: one row per radius."""
    lines = [
        f"Uncertainty-circle search, {matcher}, answer when ≥ {min_inliers} agreeing matches.",
        "",
        "| Radius | Answered | Median error | Within 100 m (of all) | Wrong > 500 m "
        "| Windows tried | Time per image |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['radius_km']:g} km | {s['answered']:.0%} | {s['median_error_m']:.0f} m "
            f"| {s['within_100m_of_all']:.0%} | {s['wrong_answers_over_500m']:.0%} "
            f"| {s['mean_windows_tried']:.1f} | {s['seconds_per_image']:.2f} s |"
        )
    return "\n".join(lines)


def write_rows(cfg: Config, name: str, radius_km: float, rows: list[dict]) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.results_dir / f"{name}_{radius_km:g}km_queries.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(cfg: Config, name: str, summaries: list[dict]) -> None:
    md = results_table(summaries, name.split("_")[-1], cfg.min_inliers) + "\n"
    (cfg.results_dir / f"{name}.md").write_text(md, encoding="utf-8")
    (cfg.results_dir / f"{name}.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print()
    print(md)
