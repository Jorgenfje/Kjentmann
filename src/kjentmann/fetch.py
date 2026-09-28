"""Download a cloud-free Sentinel-2 image of the area.

Uses the public Earth Search STAC catalogue. No account or API key is needed:
the images are Cloud Optimized GeoTIFFs, so only the pixels inside the area
are downloaded, not the whole 110 x 110 km scene.
"""

from __future__ import annotations

import json
import os

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.windows import from_bounds
from shapely.geometry import box, shape

from kjentmann.config import Config
from kjentmann.geo import lonlat_bbox, square_bounds

# Anonymous access to the public bucket, and faster remote reads.
GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
}


def search_scenes(cfg: Config) -> list:
    """Find scenes that fully cover the area, least cloudy first."""
    bbox = lonlat_bbox(cfg.center_lat, cfg.center_lon, cfg.size_km)
    client = Client.open(cfg.stac_url)
    search = client.search(
        collections=[cfg.collection],
        bbox=bbox,
        datetime=f"{cfg.date_from}/{cfg.date_to}",
        query={"eo:cloud_cover": {"lt": cfg.max_cloud_cover}},
        max_items=200,
    )
    area = box(*bbox)
    items = [it for it in search.items() if shape(it.geometry).contains(area)]
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    return items


def read_area(cfg: Config, href: str) -> tuple[np.ndarray, dict]:
    """Read only the pixels inside the area from a remote COG.

    Returns:
        The RGB array (bands, rows, cols) and a rasterio profile for writing it.
    """
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        bounds = square_bounds(cfg.center_lat, cfg.center_lon, cfg.size_km, src.crs)
        window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
        data = src.read(window=window)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            transform=src.window_transform(window),
            compress="deflate",
            tiled=True,
        )
    return data, profile


def valid_fraction(data: np.ndarray) -> float:
    """Share of pixels that are not no-data (all bands zero)."""
    return float((data.max(axis=0) > 0).mean())


def fetch(cfg: Config) -> None:
    """Download the best scene for the area and save it as a GeoTIFF."""
    for key, value in GDAL_ENV.items():
        os.environ.setdefault(key, value)

    items = search_scenes(cfg)
    if not items:
        raise SystemExit(
            "Fant ingen skyfrie bilder. Prøv et lengre datointervall eller høyere max_cloud_cover."
        )
    print(f"Fant {len(items)} kandidater. Prøver den minst skyete først.")

    for item in items:
        href = item.assets["visual"].href
        cloud = item.properties.get("eo:cloud_cover")
        print(f"  {item.id}  skydekke {cloud:.1f} %")
        data, profile = read_area(cfg, href)
        frac = valid_fraction(data)
        if frac < cfg.min_valid_fraction:
            print(f"    hopper over: bare {frac:.1%} gyldige piksler i området")
            continue

        cfg.map_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(cfg.map_path, "w", **profile) as dst:
            dst.write(data)
        meta = {
            "scene_id": item.id,
            "datetime": item.properties.get("datetime"),
            "cloud_cover": cloud,
            "crs": str(profile["crs"]),
            "width": profile["width"],
            "height": profile["height"],
            "pixel_size_m": abs(profile["transform"].a),
            "source": href,
        }
        cfg.map_meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Lagret {cfg.map_path} ({profile['width']} x {profile['height']} px)")
        return

    raise SystemExit("Ingen av kandidatene dekket hele området. Prøv et annet datointervall.")
