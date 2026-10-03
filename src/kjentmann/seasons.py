"""v0.5: does the search hold up when the season differs from the map?

The map is a June scene. Test images come from a spring, an autumn and a winter
scene, with the same simulated camera conditions as the 'realistic' profile.
Each set is searched inside an uncertainty circle (``season_radius_km``).

The winter scene is chosen as the snowiest clear scene in its date range, and
the snow share inside the area is reported, so a mild winter cannot pass as a
snow test by accident.
"""

from __future__ import annotations

import json
from dataclasses import replace

from kjentmann.config import Config
from kjentmann.fetch import fetch_query_scene
from kjentmann.match import Matcher
from kjentmann.navigate import navigate
from kjentmann.queries import build_queries

SEASONS = (("realistic", "Spring"), ("autumn", "Autumn"), ("winter", "Winter"))


def seasons(base: Config, matcher: Matcher) -> list[dict]:
    """Fetch, build and search each season's test images; write one table."""
    r = base.season_radius_km
    rows = []
    for profile, label in SEASONS:
        if profile not in base.profiles:
            continue
        cfg = replace(base.with_profile(profile), nav_radii_km=(r,), nav_map_radius_km=r)
        print(f"\n=== {label} ({profile}) ===")
        if not cfg.query_scene_path.exists():
            fetch_query_scene(cfg)
        if not cfg.queries_csv.exists():
            build_queries(cfg)
        meta = json.loads(cfg.query_scene_meta_path.read_text(encoding="utf-8"))
        s = navigate(cfg, matcher, prefix="seasons")[0]
        rows.append(
            {
                "season": label,
                "profile": profile,
                "date": (meta.get("datetime") or "")[:10],
                "snow": meta.get("area_snow_fraction"),
                **s,
            }
        )

    table = seasons_table(rows, r, base)
    out = base.data_dir / "results" / f"{base.area_name}_seasons.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table + "\n", encoding="utf-8")
    print(f"\n{table}\n\nLagret: {out}")
    return rows


def seasons_table(rows: list[dict], radius_km: float, cfg: Config) -> str:
    """Markdown: one row per season."""
    map_date = ""
    if cfg.map_meta_path.exists():
        map_date = json.loads(cfg.map_meta_path.read_text(encoding="utf-8"))["datetime"][:10]
    lines = [
        f"Map: {map_date}. Uncertainty radius {radius_km:g} km, realistic camera conditions.",
        "",
        "| Test images | Date | Snow in area | Answered | Median error | Within 100 m "
        "| Wrong > 500 m | Time per image |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for x in rows:
        snow = "n/a" if x["snow"] is None else f"{x['snow']:.0%}"
        med = "–" if x["median_error_m"] != x["median_error_m"] else f"{x['median_error_m']:.0f} m"
        lines.append(
            f"| {x['season']} | {x['date']} | {snow} | {x['answered']:.0%} | {med} "
            f"| {x['within_100m_of_all']:.0%} | {x['wrong_answers_over_500m']:.0%} "
            f"| {x['seconds_per_image']:.2f} s |"
        )
    return "\n".join(lines)
