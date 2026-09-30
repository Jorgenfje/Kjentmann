"""Offline tests for v0.2: test images, search and scoring.

A smooth synthetic landscape stands in for Sentinel-2. The "other date" scene
is the same landscape with changed brightness and added noise, like a second
satellite pass.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from kjentmann.config import load_config
from kjentmann.embed import DinoEmbedder, PixelEmbedder
from kjentmann.evaluate import containing_tiles, evaluate, random_recall, read_tiles
from kjentmann.fetch import read_area
from kjentmann.geo import square_bounds, utm_crs_for
from kjentmann.queries import build_queries, read_queries, sample_offsets
from kjentmann.tiles import build_tiles

CENTER_LAT, CENTER_LON = 59.583, 11.162


def landscape(size: int, seed: int) -> np.ndarray:
    """Smooth random RGB terrain, so neighbouring pixels are related."""
    rng = np.random.default_rng(seed)
    layers = []
    for _ in range(3):
        coarse = rng.integers(0, 255, size=(size // 20, size // 20), dtype=np.uint8)
        layers.append(np.asarray(Image.fromarray(coarse).resize((size, size), Image.BICUBIC)))
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


def test_random_recall():
    assert random_recall(100, 1, 1) == pytest.approx(0.01)
    assert random_recall(100, 4, 5) == pytest.approx(
        1 - (96 * 95 * 94 * 93 * 92) / (100 * 99 * 98 * 97 * 96)
    )
    assert random_recall(10, 0, 5) == 0.0
    assert random_recall(3, 1, 5) == 1.0


def test_sample_offsets_reproducible_and_inside():
    a = sample_offsets(500, 400, 256, 20, seed=7)
    b = sample_offsets(500, 400, 256, 20, seed=7)
    assert a == b and len(a) == 20
    assert all(0 <= r <= 244 and 0 <= c <= 144 for r, c in a)
    mask = np.ones((500, 400), bool)
    mask[:, :200] = False
    assert all(c >= 200 for _, c in sample_offsets(500, 400, 100, 10, 1, mask))


def test_query_ground_truth_lies_in_some_tile(prepared):
    cfg = prepared
    tiles = read_tiles(cfg.tiles_csv)
    with rasterio.open(cfg.map_path) as src:
        transform = src.transform
    queries = read_queries(cfg.queries_csv)
    assert len(queries) == 40
    for q in queries:
        # Edge tiles are aligned to the border, so up to 3 x 3 tiles can overlap.
        assert 1 <= len(containing_tiles(q, tiles, transform, cfg.tile_size_px)) <= 9


def test_pixel_baseline_beats_random(prepared):
    summary = evaluate(prepared, [PixelEmbedder()])[0]
    k = prepared.top_k
    # Shifted crops are hard for raw pixels, but it must still beat chance clearly.
    assert summary["recall@1"] > 2 * summary["random_recall@1"]
    assert summary[f"recall@{k}"] >= summary["recall@1"]
    assert (prepared.results_dir / "results.md").exists()
    assert (prepared.results_dir / "pixel_map.html").exists()


def test_dinov2_pipeline_runs_without_download(prepared):
    emb = DinoEmbedder(pretrained=False, device="cpu", batch_size=8)
    vecs = emb.embed([np.zeros((256, 256, 3), np.uint8)] * 3)
    assert vecs.shape == (3, 768)  # class token + mean patch token, 384 each
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-5)
    summaries = evaluate(prepared, [PixelEmbedder(), emb])
    assert [s["embedder"] for s in summaries] == ["pixel", "dinov2"]
    assert (prepared.results_dir / "dinov2_map.html").exists()


def test_index_finds_each_tile_itself(prepared):
    from kjentmann.evaluate import build_index, load_images

    tiles = read_tiles(prepared.tiles_csv)
    vecs = PixelEmbedder().embed(load_images(prepared.tiles_dir, [t.tile_id for t in tiles]))
    _, idx = build_index(vecs).search(vecs, 1)
    assert (idx[:, 0] == np.arange(len(tiles))).all()
