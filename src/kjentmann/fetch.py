"""Download cloud-free Sentinel-2 images of the area.

Uses the public Earth Search STAC catalogue. No account or API key is needed:
the images are Cloud Optimized GeoTIFFs, so only the pixels inside the area
are downloaded, not the whole 110 x 110 km scene.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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


def search_scenes(cfg: Config, date_from: str, date_to: str) -> list:
    """Find scenes that fully cover the area, least cloudy first."""
    bbox = lonlat_bbox(cfg.center_lat, cfg.center_lon, cfg.size_km)
    client = Client.open(cfg.stac_url)
    search = client.search(
        collections=[cfg.collection],
        bbox=bbox,
        datetime=f"{date_from}/{date_to}",
        query={"eo:cloud_cover": {"lt": cfg.max_cloud_cover}},
        max_items=300,
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
        # Do not inherit the source's block layout: it may not fit the crop.
        for key in ("blockxsize", "blockysize", "tiled", "interleave"):
            profile.pop(key, None)
        height, width = data.shape[1], data.shape[2]
        profile.update(
            driver="GTiff",
            height=height,
            width=width,
            transform=src.window_transform(window),
            compress="deflate",
        )
        if height >= 256 and width >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
    return data, profile


def valid_fraction(data: np.ndarray) -> float:
    """Share of pixels that are not no-data (all bands zero)."""
    return float((data.max(axis=0) > 0).mean())


def fetch_scene(
    cfg: Config,
    date_from: str,
    date_to: str,
    out_tif: Path,
    out_meta: Path,
    exclude_dates: set[str] | None = None,
) -> dict:
    """Download the least cloudy valid scene in a date range.

    Args:
        cfg: Project configuration.
        date_from: First date (YYYY-MM-DD).
        date_to: Last date (YYYY-MM-DD).
        out_tif: Where to write the GeoTIFF.
        out_meta: Where to write metadata as JSON.
        exclude_dates: Acquisition dates (YYYY-MM-DD) to skip.

    Returns:
        The metadata that was written.
    """
    for key, value in GDAL_ENV.items():
        os.environ.setdefault(key, value)
    exclude_dates = exclude_dates or set()

    items = [
        it
        for it in search_scenes(cfg, date_from, date_to)
        if it.properties.get("datetime", "")[:10] not in exclude_dates
    ]
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

        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst:
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
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Lagret {out_tif} ({profile['width']} x {profile['height']} px)")
        return meta

    raise SystemExit("Ingen av kandidatene dekket hele området. Prøv et annet datointervall.")


def fetch(cfg: Config) -> dict:
    """Download the reference map."""
    return fetch_scene(cfg, cfg.date_from, cfg.date_to, cfg.map_path, cfg.map_meta_path)


def fetch_query_scene(cfg: Config) -> dict:
    """Download a scene from another date than the map, for test images."""
    if not cfg.map_meta_path.exists():
        raise SystemExit("Kjør 'kjentmann fetch' først, så vi vet hvilken dato kartet er fra.")
    map_date = json.loads(cfg.map_meta_path.read_text(encoding="utf-8"))["datetime"][:10]
    print(f"Kartet er fra {map_date}. Henter testbilde fra en annen dato.")
    return fetch_scene(
        cfg,
        cfg.query_date_from,
        cfg.query_date_to,
        cfg.query_scene_path,
        cfg.query_scene_meta_path,
        exclude_dates={map_date},
    )
