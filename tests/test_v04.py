"""Offline tests for v0.4: search inside an uncertainty circle."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import rasterio

from kjentmann.evaluate import read_tiles
from kjentmann.match import SiftMatcher
from kjentmann.navigate import navigate, search_windows, simulated_priors


def test_priors_fill_the_unit_disc_and_are_reproducible():
    a = simulated_priors(2000, seed=42)
    assert np.allclose(a, simulated_priors(2000, seed=42))
    r = np.hypot(a[:, 0], a[:, 1])
    assert r.max() <= 1.0
    # Uniform over the area: about a quarter of points within half the radius.
    assert 0.2 < (r < 0.5).mean() < 0.3


def test_search_windows_use_a_stride_of_one_tile(prepared):
    tiles = read_tiles(prepared.tiles_csv)
    with rasterio.open(prepared.map_path) as src:
        cands = search_windows(prepared, tiles, src.transform)
    assert 0 < len(cands) < len(tiles)
    xs = sorted({round(c.cx_px) for c in cands})
    steps = np.diff(xs)
    assert (steps <= prepared.tile_size_px).all()  # no gaps wider than a tile


def test_navigate_end_to_end(prepared):
    from kjentmann.queries import build_queries

    cfg = replace(
        prepared.with_profile("realistic"), nav_radii_km=(2.0, 5.0), nav_map_radius_km=2.0
    )
    build_queries(cfg)
    summaries = navigate(cfg, SiftMatcher())

    assert [s["radius_km"] for s in summaries] == [2.0, 5.0]
    for s in summaries:
        assert s["answered"] > 0.5
        assert s["within_100m_of_answered"] > 0.9
        assert s["wrong_answers_over_500m"] <= 0.05
    # A wider circle never needs fewer windows on average.
    assert summaries[1]["mean_windows_tried"] >= summaries[0]["mean_windows_tried"]

    saved = json.loads((cfg.results_dir / "navigate_sift.json").read_text())
    assert len(saved) == 2
    assert (cfg.results_dir / "navigate_sift_2km_map.html").exists()
    assert (cfg.results_dir / "navigate_sift.md").read_text().startswith("Uncertainty-circle")
