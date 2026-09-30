"""Offline tests for v0.2: test images, search and scoring.

A smooth synthetic landscape stands in for Sentinel-2. The "other date" scene
is the same landscape with changed brightness and added noise, like a second
satellite pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from kjentmann.embed import DinoEmbedder, PixelEmbedder
from kjentmann.evaluate import containing_tiles, evaluate, random_recall, read_tiles
from kjentmann.queries import read_queries, sample_offsets


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
