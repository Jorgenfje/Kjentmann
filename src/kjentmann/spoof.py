"""Detect spoofed GPS by comparing it with the camera's position.

GPS says "you are here", the camera says "you are here". If the camera is
confident and the two disagree by more than ``threshold_m``, the GPS is treated
as spoofed and an alarm is raised. If the camera is unsure, no decision is made.

Two evaluations:

* ``spoof_stats``: every test image gets an honest GPS reading (a few metres of
  noise) and spoofed readings shifted 50 m to 2 km. Reports how often spoofing
  is detected and how often honest GPS raises a false alarm.
* ``spoof_flight``: a 15 km flight over the area with an image every 250 m.
  Partway, a spoofer starts dragging the GPS position away a little per image,
  the way real spoofing avoids sudden jumps. The camera tracks the aircraft and
  raises the alarm; the result is a map of the true route, the spoofed GPS
  route and where the alarm went off.
"""

from __future__ import annotations

import csv
import json
import math

import folium
import numpy as np
import rasterio
from pyproj import Transformer

from kjentmann.config import Config
from kjentmann.evaluate import load_images
from kjentmann.geo import apply_transform, to_lonlat
from kjentmann.match import Matcher
from kjentmann.navigate import VisualFixer, simulated_priors
from kjentmann.queries import degrade, read_queries, render_view


def gps_reading(true_xy, offset_m: float, rng, noise_m: float) -> tuple[float, float]:
    """A GPS position: the truth, shifted ``offset_m`` in a random direction, plus noise."""
    a = rng.uniform(0, 2 * math.pi)
    return (
        true_xy[0] + offset_m * math.cos(a) + rng.normal(0, noise_m),
        true_xy[1] + offset_m * math.sin(a) + rng.normal(0, noise_m),
    )


def decide(fix_xy, gps_xy, threshold_m: float) -> str:
    """'alarm', 'ok' or 'unknown' (camera not confident)."""
    if fix_xy[0] is None:
        return "unknown"
    gap = math.hypot(fix_xy[0] - gps_xy[0], fix_xy[1] - gps_xy[1])
    return "alarm" if gap > threshold_m else "ok"


def spoof_stats(cfg: Config, matcher: Matcher, fixer: VisualFixer | None = None) -> list[dict]:
    """Detection and false-alarm rates for a range of spoofing offsets."""
    fixer = fixer or VisualFixer(cfg, matcher)
    queries = read_queries(cfg.queries_csv)
    images = load_images(cfg.queries_dir, [q.query_id for q in queries])
    r_m = cfg.spoof_radius_km * 1000
    unit = simulated_priors(len(queries), cfg.query_seed)

    print(f"Locating {len(queries)} test images with the camera ...")
    fixes = []
    for qi, q in enumerate(queries):
        prior = (q.center_x + unit[qi, 0] * r_m, q.center_y + unit[qi, 1] * r_m)
        x, y, _ = fixer.fix(images[qi], ("spoof", qi), prior, r_m)
        fixes.append((x, y))
        if (qi + 1) % 50 == 0:
            print(f"  {qi + 1}/{len(queries)}", flush=True)

    rng = np.random.default_rng(cfg.query_seed + 11)
    rows = []
    for offset in (0.0, *cfg.spoof_offsets_m):
        counts = {"alarm": 0, "ok": 0, "unknown": 0}
        for qi, q in enumerate(queries):
            gps = gps_reading((q.center_x, q.center_y), offset, rng, cfg.gps_noise_m)
            counts[decide(fixes[qi], gps, cfg.spoof_threshold_m)] += 1
        n = len(queries)
        rows.append(
            {
                "offset_m": offset,
                "alarm": counts["alarm"] / n,
                "ok": counts["ok"] / n,
                "unknown": counts["unknown"] / n,
            }
        )
    return rows


def stats_table(rows: list[dict], threshold_m: float) -> str:
    """Markdown: honest GPS first (false alarms), then each spoofing offset."""
    lines = [
        f"Alarm when camera and GPS disagree by more than {threshold_m:g} m.",
        "",
        "| GPS | Alarm | No alarm | Camera unsure |",
        "|---|---|---|---|",
    ]
    for r in rows:
        name = "Honest" if r["offset_m"] == 0 else f"Spoofed {r['offset_m']:g} m"
        lines.append(f"| {name} | {r['alarm']:.0%} | {r['ok']:.0%} | {r['unknown']:.0%} |")
    return "\n".join(lines)


def flight_route(cfg: Config, map_tf, width: int, height: int):
    """Straight route through the middle of the map, ``flight_steps`` points apart."""
    left, top = apply_transform(map_tf, 0, 0)
    right, bottom = apply_transform(map_tf, width, height)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    angle = math.radians(-35)  # heading towards south-east
    dx, dy = math.cos(angle), math.sin(angle)
    total = (cfg.flight_steps - 1) * cfg.flight_step_m
    start = (cx - dx * total / 2, cy - dy * total / 2)
    pts = [
        (start[0] + dx * i * cfg.flight_step_m, start[1] + dy * i * cfg.flight_step_m)
        for i in range(cfg.flight_steps)
    ]
    return pts, (-dy, dx)  # route points, and the direction the spoofer drags towards


def spoof_flight(cfg: Config, matcher: Matcher, fixer: VisualFixer | None = None) -> dict:
    """Simulate a flight with gradual spoofing and report when the alarm goes off."""
    fixer = fixer or VisualFixer(cfg, matcher)
    with rasterio.open(cfg.query_scene_path) as src:
        scene = np.ascontiguousarray(np.moveaxis(src.read()[:3], 0, -1))
        scene_tf, scene_crs = src.transform, src.crs
    to_scene = Transformer.from_crs(fixer.map_crs, scene_crs, always_xy=True)
    route, drag = flight_route(cfg, fixer.map_tf, fixer.image.shape[2], fixer.image.shape[1])

    rng = np.random.default_rng(cfg.query_seed + 23)
    inv_scene = ~scene_tf
    prior = route[0]
    streak = 0
    alarm_step = None
    rows = []
    print(f"Flying {len(route)} images, spoofing starts at image {cfg.spoof_start_step} ...")
    for i, true_xy in enumerate(route):
        sx, sy = to_scene.transform(*true_xy)
        px, py = apply_transform(inv_scene, sx, sy)
        angle = float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
        scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
        img = render_view(scene, px, py, cfg.tile_size_px, angle, scale)
        img = degrade(img, cfg.blur_sigma, cfg.noise_std, rng)

        offset = max(0, i - cfg.spoof_start_step) * cfg.drift_m_per_step
        gps = (
            true_xy[0] + drag[0] * offset + rng.normal(0, cfg.gps_noise_m),
            true_xy[1] + drag[1] * offset + rng.normal(0, cfg.gps_noise_m),
        )
        x, y, inliers = fixer.fix(img, ("flight", i), prior, cfg.track_radius_km * 1000)
        verdict = decide((x, y), gps, cfg.spoof_threshold_m)
        streak = streak + 1 if verdict == "alarm" else (streak if verdict == "unknown" else 0)
        if alarm_step is None and streak >= 2:
            alarm_step = i

        # Next prior: last camera fix, moved one step along the route (dead reckoning).
        step = (
            route[min(i + 1, len(route) - 1)][0] - true_xy[0],
            route[min(i + 1, len(route) - 1)][1] - true_xy[1],
        )
        base = (x, y) if x is not None else prior
        prior = (base[0] + step[0], base[1] + step[1])

        gap = None if x is None else math.hypot(x - gps[0], y - gps[1])
        rows.append(
            {
                "step": i,
                "spoof_offset_m": round(offset, 1),
                "true": to_lonlat(fixer.map_crs, *true_xy),
                "gps": to_lonlat(fixer.map_crs, *gps),
                "camera": None if x is None else to_lonlat(fixer.map_crs, x, y),
                "camera_error_m": None
                if x is None
                else round(math.hypot(x - true_xy[0], y - true_xy[1]), 1),
                "gap_m": None if gap is None else round(gap, 1),
                "inliers": inliers,
                "verdict": verdict,
            }
        )

    result = {
        "steps": len(route),
        "step_m": cfg.flight_step_m,
        "spoof_start_step": cfg.spoof_start_step,
        "drift_m_per_step": cfg.drift_m_per_step,
        "alarm_step": alarm_step,
        "offset_at_alarm_m": None
        if alarm_step is None
        else max(0, alarm_step - cfg.spoof_start_step) * cfg.drift_m_per_step,
        "false_alarm_before_spoofing": alarm_step is not None and alarm_step < cfg.spoof_start_step,
        "camera_fixes": sum(r["camera"] is not None for r in rows) / len(rows),
        "median_camera_error_m": float(
            np.median([r["camera_error_m"] for r in rows if r["camera_error_m"] is not None])
        )
        if any(r["camera_error_m"] is not None for r in rows)
        else None,
    }
    build_flight_map(cfg, rows, result)
    return {"summary": result, "rows": rows}


def flight_text(s: dict) -> str:
    """One short paragraph for the terminal and the README."""
    if s["alarm_step"] is None:
        alarm = "No alarm was raised."
    elif s["false_alarm_before_spoofing"]:
        alarm = f"False alarm at image {s['alarm_step']}, before spoofing started."
    else:
        km = s["alarm_step"] * s["step_m"] / 1000
        alarm = (
            f"Alarm at image {s['alarm_step']} ({km:.2f} km into the flight), when the GPS "
            f"had been dragged {s['offset_at_alarm_m']:g} m off."
        )
    return (
        f"Flight: {s['steps']} images, {s['step_m']} m apart. Spoofing starts at image "
        f"{s['spoof_start_step']} and drags the GPS {s['drift_m_per_step']} m per image. "
        f"{alarm} Camera position found for {s['camera_fixes']:.0%} of images, "
        f"median error {s['median_camera_error_m']:.0f} m."
    )


def build_flight_map(cfg: Config, rows: list[dict], s: dict) -> None:
    """True route, spoofed GPS route, camera fixes and the alarm point."""
    from kjentmann.viz import _base_map

    m = _base_map(cfg)
    latlon = lambda p: [p[1], p[0]]  # noqa: E731 (lon, lat) -> [lat, lon]
    folium.PolyLine(
        [latlon(r["true"]) for r in rows], color="#222222", weight=3, tooltip="True route"
    ).add_to(m)
    folium.PolyLine(
        [latlon(r["gps"]) for r in rows],
        color="#d73027",
        weight=3,
        dash_array="6 6",
        tooltip=f"GPS (spoofed from image {s['spoof_start_step']})",
    ).add_to(m)
    for r in rows:
        if r["camera"] is not None:
            folium.CircleMarker(
                latlon(r["camera"]),
                radius=3,
                color="#1a9850",
                fill=True,
                fill_opacity=0.9,
                tooltip=f"Image {r['step']}: camera {r['camera_error_m']:.0f} m off, "
                f"GPS gap {r['gap_m']:.0f} m",
            ).add_to(m)
    start = rows[s["spoof_start_step"]]
    folium.Marker(
        latlon(start["true"]), tooltip="Spoofing starts", icon=folium.Icon(color="orange")
    ).add_to(m)
    if s["alarm_step"] is not None:
        a = rows[s["alarm_step"]]
        folium.Marker(
            latlon(a["true"]),
            tooltip=f"ALARM: GPS {s['offset_at_alarm_m']:g} m off",
            icon=folium.Icon(color="red", icon="exclamation-sign"),
        ).add_to(m)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.results_dir / "spoof_flight_map.html"
    m.save(str(out))
    print(f"Saved flight map: {out}")


def spoof(cfg: Config, matcher: Matcher) -> dict:
    """Run both evaluations and write one report."""
    fixer = VisualFixer(cfg, matcher)
    stats = spoof_stats(cfg, matcher, fixer)
    flight = spoof_flight(cfg, matcher, fixer)
    md = stats_table(stats, cfg.spoof_threshold_m) + "\n\n" + flight_text(flight["summary"]) + "\n"
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "spoof.md").write_text(md, encoding="utf-8")
    (cfg.results_dir / "spoof.json").write_text(
        json.dumps({"stats": stats, "flight": flight["summary"]}, indent=2), encoding="utf-8"
    )
    with (cfg.results_dir / "spoof_flight.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flight["rows"][0].keys()))
        writer.writeheader()
        writer.writerows(flight["rows"])
    print()
    print(md)
    return {"stats": stats, "flight": flight["summary"]}
