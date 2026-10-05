"""Offline test for 'kjentmann locate': a synthetic phone photo of known position."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
import rasterio
from conftest import landscape, write_scene
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin

from kjentmann.config import load_config
from kjentmann.fetch import read_area
from kjentmann.geo import square_bounds, utm_crs_for
from kjentmann.locate import area_config, exif_position, locate_photo, make_view, parse_near
from kjentmann.match import SiftMatcher
from kjentmann.queries import render_view

LAT, LON = 59.583, 11.162


def _gps_exif(lat: float, lon: float) -> Image.Exif:
    def dms(v: float):
        d = int(v)
        m = int((v - d) * 60)
        s = round(((v - d) * 60 - m) * 60, 4)
        return (d, m, s)

    exif = Image.Exif()
    exif.get_ifd(0x8825).update({1: "N", 2: dms(lat), 3: "E", 4: dms(lon)})
    return exif


def test_parse_near_coordinates():
    assert parse_near("59.58, 11.16")[:2] == (59.58, 11.16)


def test_make_view_keeps_outside_black():
    sq = np.full((64, 64, 3), 200, np.uint8)
    v = make_view(sq, 45, 1.0)
    assert v[0, 0].sum() == 0  # rotated corner is empty, not mirrored content
    assert v[32, 32].sum() > 0


def test_locate_synthetic_phone_photo(tmp_path):
    base = replace(load_config("config.yaml"), data_dir=tmp_path)
    crs = utm_crs_for(LAT, LON)
    min_x, _, _, max_y = square_bounds(LAT, LON, 13, crs)
    world = landscape(1300, seed=5)
    big = tmp_path / "world.tif"
    write_scene(big, world, crs, from_origin(min_x, max_y, 10, 10))

    # The photo: 3 km of ground, 1.2 km north-east of the rough position, heading 120°.
    to_m = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = to_m.transform(LON, LAT)
    tx, ty = cx + 850, cy + 850
    true_lon, true_lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(tx, ty)
    col, row = (tx - min_x) / 10, (max_y - ty) / 10
    rgb = np.ascontiguousarray(np.moveaxis(world, 0, -1))
    patch = render_view(rgb, col, row, 300, 120, 1.0)  # 300 px x 10 m = 3 km
    photo = Image.fromarray(patch).resize((1200, 1200), Image.BICUBIC).crop((0, 150, 1200, 1050))
    path = tmp_path / "photo.jpg"
    photo.save(path, exif=_gps_exif(true_lat, true_lon), quality=92)
    assert exif_position(path) == pytest.approx((true_lat, true_lon), abs=1e-5)

    # Altitude that gives 3 km across with a 70° field of view, guessed 15 % too low.
    altitude = 3000 / (2 * math.tan(math.radians(35))) * 0.85
    cfg = area_config(base, LAT, LON, 2.0, 300)
    data, profile = read_area(cfg, str(big))
    cfg.map_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(cfg.map_path, "w", **profile) as dst:
        dst.write(data)

    matcher = SiftMatcher()
    fix = locate_photo(base, matcher, path, f"{LAT},{LON}", altitude, radius_km=2.0)
    assert fix.found
    assert fix.error_m < 60
    assert fix.accuracy_m <= 60
    assert (
        abs(((fix.heading_deg - 240) + 180) % 360 - 180) < 15
        or abs(((fix.heading_deg - 120) + 180) % 360 - 180) < 15
    )
    assert 0.8 * altitude / 0.85 < fix.altitude_m < 1.2 * altitude / 0.85

    # A second, different photo with the same matcher (as in the web app) must not
    # reuse the first photo's cached features.
    tx2, ty2 = cx - 900, cy - 600
    lon2, lat2 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(tx2, ty2)
    patch2 = render_view(rgb, (tx2 - min_x) / 10, (max_y - ty2) / 10, 300, 0, 1.0)
    path2 = tmp_path / "photo2.jpg"
    Image.fromarray(patch2).resize((1200, 1200), Image.BICUBIC).save(
        path2, exif=_gps_exif(lat2, lon2), quality=92
    )
    fix2 = locate_photo(base, matcher, path2, f"{LAT},{LON}", altitude, radius_km=2.0)
    assert fix2.found
    assert fix2.error_m < 60


def test_ensure_map_relaxes_search_until_a_scene_is_found(prepared, tmp_path, monkeypatch):
    from test_v05 import LAT as V5_LAT
    from test_v05 import LON as V5_LON
    from test_v05 import _fake_item

    import kjentmann.fetch as fetch_mod
    from kjentmann.locate import ensure_map

    item = _fake_item(tmp_path, "clear", "2024-07-01", [])
    calls = []

    def search(cfg, date_from, date_to, max_cloud=None):
        calls.append((date_from, max_cloud))
        return [item] if date_from.startswith("2024") else []  # only last year has one

    monkeypatch.setattr(fetch_mod, "search_scenes", search)
    cfg = replace(area_config(prepared, V5_LAT, V5_LON, 1.0, 100), size_km=5.0)
    ensure_map(cfg)
    assert cfg.map_path.exists()
    assert [c[1] for c in calls] == [cfg.max_cloud_cover, 60.0, 60.0]
    assert calls[-1][0].startswith("2024")
