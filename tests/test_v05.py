"""Offline tests for v0.5: scene selection by snow and cloud, and the season run."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin

import kjentmann.fetch as fetch_mod
from kjentmann.geo import square_bounds, utm_crs_for
from kjentmann.match import SiftMatcher
from kjentmann.seasons import seasons

LAT, LON = 59.583, 11.162


def _write(path, data, crs, transform):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=data.shape[0],
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


def _fake_item(tmp_path, name, day, scl_value_fractions):
    """A catalogue item with a visual COG and an SCL layer on local disk."""
    crs = utm_crs_for(LAT, LON)
    min_x, _, _, max_y = square_bounds(LAT, LON, 8, crs)
    visual = np.full((3, 800, 800), 120, np.uint8)
    _write(tmp_path / f"{name}_vis.tif", visual, crs, from_origin(min_x, max_y, 10, 10))
    scl = np.full((1, 400, 400), 4, np.uint8)  # 4 = vegetation
    flat = scl.reshape(-1)
    start = 0
    for value, frac in scl_value_fractions:
        n = int(frac * flat.size)
        flat[start : start + n] = value
        start += n
    _write(tmp_path / f"{name}_scl.tif", scl, crs, from_origin(min_x, max_y, 20, 20))
    return SimpleNamespace(
        id=name,
        properties={"datetime": f"{day}T10:50:00Z", "eo:cloud_cover": 5.0},
        assets={
            "visual": SimpleNamespace(href=str(tmp_path / f"{name}_vis.tif")),
            "scl": SimpleNamespace(href=str(tmp_path / f"{name}_scl.tif")),
        },
    )


def test_prefer_snow_picks_snowiest_clear_scene(prepared, tmp_path, monkeypatch):
    items = [
        _fake_item(tmp_path, "clear_nosnow", "2025-02-05", []),
        _fake_item(tmp_path, "cloudy_snow", "2025-02-10", [(11, 0.6), (9, 0.3)]),
        _fake_item(tmp_path, "clear_snow", "2025-03-01", [(11, 0.7)]),
    ]
    monkeypatch.setattr(
        fetch_mod, "search_scenes", lambda *a, **k: [fetch_mod.Scene([it]) for it in items]
    )
    cfg = replace(prepared, size_km=5.0)
    out, meta_path = tmp_path / "w.tif", tmp_path / "w.json"

    meta = fetch_mod.fetch_scene(cfg, "a", "b", out, meta_path, prefer_snow=True)
    assert meta["scene_id"] == "clear_snow"  # cloudy one skipped despite snow
    assert meta["area_snow_fraction"] > 0.5

    meta = fetch_mod.fetch_scene(cfg, "a", "b", out, meta_path, prefer_snow=False)
    assert meta["scene_id"] == "clear_nosnow"  # first valid scene when not preferring snow


def test_seasons_runs_every_season(prepared):
    from kjentmann.config import Config

    base: Config = replace(prepared, season_radius_km=2.0)
    # Offline: reuse the default synthetic test scene for every season.
    for profile in ("autumn", "winter"):
        cfg = base.with_profile(profile)
        shutil.copy(base.query_scene_path, cfg.query_scene_path)
        cfg.query_scene_meta_path.write_text(
            json.dumps({"datetime": "2025-03-01T10:50:00Z", "area_snow_fraction": 0.42})
        )
    base.with_profile("realistic").query_scene_meta_path.write_text(
        json.dumps({"datetime": "2025-05-19T10:50:00Z", "area_snow_fraction": 0.0})
    )

    rows = seasons(base, SiftMatcher())
    assert [r["season"] for r in rows] == ["Spring", "Autumn", "Winter"]
    assert all(r["answered"] > 0.5 for r in rows)
    table = (base.data_dir / "results" / "askim_seasons.md").read_text()
    assert "| Winter | 2025-03-01 | 42% |" in table


def test_scattered_dark_pixels_do_not_cluster_test_images():
    from kjentmann.queries import sample_offsets

    rng = np.random.default_rng(1)
    valid = rng.random((1000, 1000)) > 0.001  # 0.1 % isolated zero pixels
    valid[:, :100] = False  # one real no-data strip
    offs = sample_offsets(1000, 1000, 300, 50, seed=3, min_valid=valid)
    assert len(offs) == 50
    cols = [c for _, c in offs]
    assert min(cols) >= 97  # the real no-data strip is still avoided (1 % tolerance)
    assert max(cols) - min(cols) > 400  # spread out, not clustered


def test_joins_two_scenes_when_the_area_lies_on_their_edge(prepared, tmp_path, monkeypatch):
    """Two images in different UTM zones, each covering half the area."""
    from pyproj import CRS, Transformer
    from rasterio.warp import Resampling, reproject
    from shapely.geometry import box, mapping
    from shapely.ops import transform as shp_transform

    cfg = replace(prepared, size_km=5.0)
    crs = utm_crs_for(LAT, LON)
    min_x, min_y, max_x, max_y = square_bounds(LAT, LON, 6, crs)  # a bit larger than the area
    n = 600
    tf = from_origin(min_x, max_y, 10, 10)
    rng = np.random.default_rng(1)
    world = rng.integers(1, 255, (3, n, n), dtype=np.uint8)
    world = np.repeat(np.repeat(world[:, ::4, ::4], 4, 1), 4, 2)  # 40 m blocks survive warping

    def item(name, cols, dst_crs):
        part = np.zeros_like(world)
        part[:, :, cols] = world[:, :, cols]
        if dst_crs == crs:
            data, dtf = part, tf
        else:
            data = np.zeros((3, n + 200, n + 200), np.uint8)
            x0, y1 = Transformer.from_crs(crs, dst_crs, always_xy=True).transform(min_x, max_y)
            dtf = from_origin(x0 - 1000, y1 + 1000, 10, 10)
            reproject(
                part,
                data,
                src_transform=tf,
                src_crs=crs,
                dst_transform=dtf,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0,
            )
        _write(tmp_path / f"{name}.tif", data, dst_crs, dtf)
        scl = np.full((1, data.shape[1] // 2, data.shape[2] // 2), 4, np.uint8)
        _write(tmp_path / f"{name}_scl.tif", scl, dst_crs, from_origin(dtf.c, dtf.f, 20, 20))
        x = min_x + cols.start * 10, min_x + cols.stop * 10
        foot = shp_transform(
            Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform,
            box(x[0], min_y, x[1], max_y),
        )
        return SimpleNamespace(
            id=name,
            geometry=mapping(foot),
            properties={"datetime": "2025-07-01T10:50:00Z", "eo:cloud_cover": 1.0},
            assets={
                "visual": SimpleNamespace(href=str(tmp_path / f"{name}.tif")),
                "scl": SimpleNamespace(href=str(tmp_path / f"{name}_scl.tif")),
            },
        )

    west = item("west", slice(0, 320), crs)
    east = item("east", slice(280, n), CRS.from_epsg(32633))
    area = box(*fetch_mod.lonlat_bbox(LAT, LON, cfg.size_km))
    scenes = fetch_mod.group_scenes([west, east], area)
    assert [sc.id for sc in scenes] == ["west+east"]

    monkeypatch.setattr(fetch_mod, "search_scenes", lambda *a, **k: scenes)
    out = tmp_path / "joined.tif"
    meta = fetch_mod.fetch_scene(cfg, "a", "b", out, tmp_path / "joined.json")
    assert meta["scene_id"] == "west+east"

    with rasterio.open(out) as src:
        got = src.read()
        a_min_x, _, _, a_max_y = square_bounds(LAT, LON, cfg.size_km, src.crs)
        assert src.crs == crs
    assert (got.max(axis=0) > 0).mean() > 0.999  # no holes along the seam
    r0, c0 = round((max_y - a_max_y) / 10), round((a_min_x - min_x) / 10)
    truth = world[:, r0 : r0 + got.shape[1], c0 : c0 + got.shape[2]]
    assert (np.abs(got.astype(int) - truth).max(axis=0) == 0).mean() > 0.95
