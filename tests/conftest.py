"""Shared fixtures: synthetic terrain standing in for Sentinel-2."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from kjentmann.config import load_config
from kjentmann.fetch import read_area
from kjentmann.geo import square_bounds, utm_crs_for
from kjentmann.queries import build_queries
from kjentmann.tiles import build_tiles

CENTER_LAT, CENTER_LON = 59.583, 11.162


def landscape(size: int, seed: int) -> np.ndarray:
    """Smooth random RGB terrain, so neighbouring pixels are related."""
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(3):
        # Large shapes (fields, forest) plus finer detail, so keypoints exist.
        coarse = rng.integers(0, 255, size=(size // 20, size // 20), dtype=np.uint8)
        fine = rng.integers(0, 255, size=(size // 4, size // 4), dtype=np.uint8)
        big = np.asarray(Image.fromarray(coarse).resize((size, size), Image.BICUBIC), float)
        small = np.asarray(Image.fromarray(fine).resize((size, size), Image.BICUBIC), float)
        layers.append(0.7 * big + 0.3 * small)
    return np.clip(np.stack(layers), 1, 255).astype(np.uint8)


def write_scene(path, data, crs, transform):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


@pytest.fixture()
def prepared(tmp_path):
    """Map, tiles and a query scene from 'another date', all on disk."""
    cfg = replace(load_config("config.yaml"), data_dir=tmp_path, size_km=10.0, query_count=40)
    crs = utm_crs_for(CENTER_LAT, CENTER_LON)
    min_x, _, _, max_y = square_bounds(CENTER_LAT, CENTER_LON, 13, crs)
    transform = from_origin(min_x, max_y, 10, 10)
    terrain = landscape(1300, seed=1)

    big = tmp_path / "big.tif"
    write_scene(big, terrain, crs, transform)
    data, profile = read_area(cfg, str(big))
    cfg.map_path.parent.mkdir(parents=True)
    with rasterio.open(cfg.map_path, "w", **profile) as dst:
        dst.write(data)
    cfg.map_meta_path.write_text(json.dumps({"datetime": "2025-07-01T10:00:00Z"}))

    rng = np.random.default_rng(2)
    other = terrain.astype(np.float32) * 0.8 + 20 + rng.normal(0, 8, terrain.shape)
    other_path = tmp_path / "other.tif"
    write_scene(other_path, np.clip(other, 1, 255).astype(np.uint8), crs, transform)
    qdata, qprofile = read_area(cfg, str(other_path))
    with rasterio.open(cfg.query_scene_path, "w", **qprofile) as dst:
        dst.write(qdata)

    build_tiles(cfg)
    build_queries(cfg)
    return cfg
