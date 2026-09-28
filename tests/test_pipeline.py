"""Offline tests: a synthetic GeoTIFF stands in for the Sentinel-2 image."""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from kjentmann.config import load_config
from kjentmann.fetch import read_area, valid_fraction
from kjentmann.geo import square_bounds, to_lonlat, utm_crs_for
from kjentmann.tiles import build_tiles, make_tiles, tile_origins
from kjentmann.viz import build_map

CENTER_LAT, CENTER_LON = 59.583, 11.162


@pytest.fixture()
def cfg(tmp_path):
    base = load_config("config.yaml")
    return replace(base, data_dir=tmp_path, size_km=5.0)


@pytest.fixture()
def big_scene(tmp_path):
    """A 12 x 12 km fake scene in UTM 32N with 10 m pixels around Askim."""
    crs = utm_crs_for(CENTER_LAT, CENTER_LON)
    min_x, min_y, max_x, max_y = square_bounds(CENTER_LAT, CENTER_LON, 12, crs)
    size = 1200
    rng = np.random.default_rng(0)
    data = rng.integers(1, 255, size=(3, size, size), dtype=np.uint8)
    path = tmp_path / "scene.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=from_origin(min_x, max_y, 10, 10),
        tiled=True,
    ) as dst:
        dst.write(data)
    return path


def test_utm_zone_for_askim():
    assert utm_crs_for(CENTER_LAT, CENTER_LON).to_epsg() == 32632


def test_square_bounds_is_square_and_centered():
    crs = utm_crs_for(CENTER_LAT, CENTER_LON)
    min_x, min_y, max_x, max_y = square_bounds(CENTER_LAT, CENTER_LON, 20, crs)
    assert max_x - min_x == pytest.approx(20_000)
    assert max_y - min_y == pytest.approx(20_000)
    lon, lat = to_lonlat(crs, (min_x + max_x) / 2, (min_y + max_y) / 2)
    assert lat == pytest.approx(CENTER_LAT, abs=1e-6)
    assert lon == pytest.approx(CENTER_LON, abs=1e-6)


def test_tile_origins_cover_edge():
    assert tile_origins(1000, 256, 128) == [0, 128, 256, 384, 512, 640, 744]
    assert tile_origins(100, 256, 128) == []


def test_make_tiles_positions():
    crs = utm_crs_for(CENTER_LAT, CENTER_LON)
    transform = from_origin(600_000, 6_610_000, 10, 10)
    image = np.zeros((3, 512, 512), dtype=np.uint8)
    tiles = make_tiles(image, transform, crs, size=256, overlap=0.5)
    assert len(tiles) == 9  # 3 x 3
    first, pixels = tiles[0]
    assert pixels.shape == (3, 256, 256)
    assert first.center_x == pytest.approx(600_000 + 128 * 10)
    assert first.center_y == pytest.approx(6_610_000 - 128 * 10)


def test_read_area_crops_to_config(cfg, big_scene):
    data, profile = read_area(cfg, str(big_scene))
    assert data.shape == (3, 500, 500)  # 5 km / 10 m
    assert valid_fraction(data) == 1.0
    assert profile["crs"].to_epsg() == 32632


def test_tiles_and_map_end_to_end(cfg, big_scene):
    data, profile = read_area(cfg, str(big_scene))
    cfg.map_path.parent.mkdir(parents=True)
    with rasterio.open(cfg.map_path, "w", **profile) as dst:
        dst.write(data)

    tiles = build_tiles(cfg)
    assert len(tiles) == 9  # 500 px, 256 px tiles, stride 128 -> 0, 128, 244
    assert len(list(cfg.tiles_dir.glob("*.png"))) == 9
    with cfg.tiles_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["tile_id"] == "r000_c000"

    build_map(cfg)
    html = (cfg.data_dir / f"{cfg.area_name}_map.html").read_text()
    assert "r001_c001" in html
