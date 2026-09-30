"""Offline tests for v0.3: point matching, verification and position in metres.

SIFT stands in for LightGlue here, so no weights need downloading. The same
code path runs with LightGlue on a machine with internet access.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from kjentmann.embed import PixelEmbedder
from kjentmann.match import SiftMatcher, verify
from kjentmann.refine import refine, summarize, window_around


def test_verify_recovers_known_shift_and_rejects_noise():
    rng = np.random.default_rng(0)
    pa = rng.uniform(0, 256, size=(60, 2)).astype(np.float32)
    pb = pa + np.array([40.0, -25.0], np.float32)
    pb[:15] = rng.uniform(0, 256, size=(15, 2))  # 25 % outliers
    v = verify(pa, pb)
    assert v.inliers >= 44
    x, y = v.map_point(128, 128)
    assert (x, y) == pytest.approx((168, 103), abs=0.5)

    assert verify(pa[:3], pb[:3]).transform is None  # too few points


def test_window_stays_inside_map():
    img = np.zeros((3, 500, 400), np.uint8)
    w = window_around(img, cx=390, cy=10, size=384)
    assert (w.x0, w.y0) == (16, 0)
    assert w.pixels.shape == (384, 384, 3)


def test_sift_finds_crop_in_window(prepared):
    import rasterio

    with rasterio.open(prepared.map_path) as src:
        image = src.read()
    win = window_around(image, 500, 500, 384)
    crop = np.ascontiguousarray(win.pixels[100:356, 60:316])
    crop = cv2.convertScaleAbs(crop, alpha=0.85, beta=15)  # "another date"
    v = verify(*SiftMatcher().match(crop, win.pixels))
    assert v.inliers > 30
    assert v.map_point(128, 128) == pytest.approx((60 + 128, 100 + 128), abs=2)


def test_refine_end_to_end(prepared):
    cfg = prepared
    summary = refine(cfg, PixelEmbedder(), SiftMatcher())

    # Windows overlap neighbouring tiles, so fine matching can place an image
    # even when the coarse search ranked a neighbour of the right tile first.
    assert summary["within_100m_of_all"] >= summary["coarse_hit@1"]
    # Answered images are placed to within a few pixels (10 m each).
    assert summary["answered"] > 0.3
    assert summary["median_error_m"] < 50
    assert summary["within_100m_of_answered"] > 0.9
    # A stricter threshold never answers more often.
    answered = [t["answered"] for t in summary["sweep"]]
    assert answered == sorted(answered, reverse=True)

    saved = json.loads((cfg.results_dir / "refine_sift.json").read_text())
    assert saved["matcher"] == "sift"
    assert (cfg.results_dir / "refine_sift_map.html").exists()
    assert (cfg.results_dir / "refine_sift.md").exists()


def test_summarize_treats_low_confidence_as_unknown():
    rows = [
        {
            "inliers": 40,
            "error_m": 20.0,
            "coarse_rank_of_correct": 1,
            "coarse_error_m": 900.0,
            "chosen_is_correct": True,
        },
        {
            "inliers": 5,
            "error_m": 3000.0,
            "coarse_rank_of_correct": "",
            "coarse_error_m": 5000.0,
            "chosen_is_correct": False,
        },
    ]
    s = summarize(rows, min_inliers=15, k=5)
    assert s["answered"] == 0.5
    assert s["median_error_m"] == 20.0
    assert s["wrong_answers_over_500m"] == 0.0
    assert summarize(rows, min_inliers=1, k=5)["wrong_answers_over_500m"] == 0.5


def test_lightglue_code_path_without_download(monkeypatch):
    """Run the real LightGlue matcher with random weights (no network)."""
    import kornia.feature as KF
    import torch

    from kjentmann.match import LightGlueMatcher

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *a, **k: {})
    monkeypatch.setattr(
        KF.DISK, "from_pretrained", classmethod(lambda cls, checkpoint="depth", device=None: cls())
    )
    m = LightGlueMatcher(max_keypoints=128, device="cpu")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
    b = rng.integers(0, 255, (384, 384, 3), dtype=np.uint8)
    pa, pb = m.match(a, b, b_key=7)
    assert pa.shape[1] == 2 and pb.shape == pa.shape
    assert 7 in m._cache  # window features are reused across test images


def test_render_view_matches_crop_and_rotation():
    from kjentmann.queries import render_view

    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    plain = render_view(rgb, 150, 150, 64, 0, 1)
    assert np.array_equal(plain, rgb[118:182, 118:182])
    # 90 degrees about the centre is an exact rotation of the plain crop.
    turned = render_view(rgb, 150, 150, 64, 90, 1)
    assert np.abs(turned.astype(int) - np.rot90(plain, k=1).astype(int)).mean() < 2
    # Scale 2: the view covers twice the ground, so it matches a shrunk crop.
    wide = render_view(rgb, 150, 150, 64, 0, 2)
    ref = cv2.resize(rgb[86:214, 86:214], (64, 64), interpolation=cv2.INTER_AREA)
    assert np.abs(wide.astype(int) - ref.astype(int)).mean() < 3


def test_realistic_profile_end_to_end(prepared):
    from kjentmann.queries import build_queries, read_queries

    cfg = prepared.with_profile("realistic")
    assert cfg.queries_dir != prepared.queries_dir  # separate folders
    queries = build_queries(cfg)
    assert any(abs(q.rotation_deg) > 1 for q in queries)
    assert all(0.8 <= q.scale <= 1.25 for q in read_queries(cfg.queries_csv))

    summary = refine(cfg, PixelEmbedder(), SiftMatcher())
    assert summary["profile"] == "realistic"
    # Rotation and scale are handled by the similarity fit: answered images
    # still land close, and a wrong answer stays rare.
    assert summary["answered"] > 0.2
    assert summary["within_100m_of_answered"] > 0.8
    assert (cfg.results_dir / "refine_sift.md").exists()


def test_coarse_candidates_without_augmentation_equals_plain_search(prepared):
    from kjentmann.evaluate import build_index, load_images, read_tiles
    from kjentmann.refine import coarse_candidates

    tiles = read_tiles(prepared.tiles_csv)
    imgs = load_images(prepared.tiles_dir, [t.tile_id for t in tiles])
    emb = PixelEmbedder()
    index = build_index(emb.embed(imgs))
    plain = index.search(emb.embed(imgs[:6]), 5)[1]
    got = coarse_candidates(emb, index, imgs[:6], 5)
    assert (got[:, 0] == plain[:, 0]).all()
    # With extra rotations each image still finds itself first.
    got_tta = coarse_candidates(emb, index, imgs[:6], 5, rotations=(-15, 0, 15))
    assert (got_tta[:, 0] == np.arange(6)).all()
