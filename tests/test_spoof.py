"""Offline tests for spoofing detection."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from kjentmann.match import SiftMatcher
from kjentmann.spoof import decide, gps_reading, spoof


def test_decide():
    assert decide((None, None), (0, 0), 100) == "unknown"
    assert decide((0, 0), (30, 40), 100) == "ok"  # 50 m apart
    assert decide((0, 0), (300, 400), 100) == "alarm"  # 500 m apart


def test_gps_reading_offset_and_noise():
    rng = np.random.default_rng(0)
    d = [np.hypot(*np.subtract(gps_reading((0, 0), 1000, rng, 5), (0, 0))) for _ in range(200)]
    assert 980 < np.mean(d) < 1020


def test_spoof_end_to_end(prepared):
    from kjentmann.queries import build_queries

    cfg = replace(
        prepared.with_profile("realistic"),
        spoof_radius_km=2.0,
        flight_steps=20,
        flight_step_m=250,
        spoof_start_step=8,
        drift_m_per_step=50,
        track_radius_km=1.0,
    )
    build_queries(cfg)
    result = spoof(cfg, SiftMatcher())

    stats = {r["offset_m"]: r for r in result["stats"]}
    assert stats[0.0]["alarm"] <= 0.05  # honest GPS: hardly any false alarms
    assert stats[2000.0]["alarm"] >= stats[0.0]["alarm"] + 0.5  # big spoofs are caught
    f = result["flight"]
    assert f["alarm_step"] is not None and not f["false_alarm_before_spoofing"]
    assert f["offset_at_alarm_m"] <= 300
    saved = json.loads((cfg.results_dir / "spoof.json").read_text())
    assert saved["flight"]["alarm_step"] == f["alarm_step"]
    assert (cfg.results_dir / "spoof_flight_map.html").exists()
