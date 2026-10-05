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
from pystac_client.stac_api_io import StacApiIO
from rasterio.windows import from_bounds
from shapely.geometry import box, shape

from kjentmann.config import Config
from kjentmann.geo import lonlat_bbox, square_bounds

# Anonymous access to the public bucket, and faster remote reads.
GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    # Never wait forever on a slow server: time out and retry a few times.
    "GDAL_HTTP_CONNECTTIMEOUT": "15",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}
STAC_TIMEOUT = (15, 45)  # seconds: connect, read
STAC_RETRIES = 2


def search_scenes(
    cfg: Config, date_from: str, date_to: str, max_cloud: float | None = None
) -> list:
    """Find scenes that fully cover the area, least cloudy first."""
    bbox = lonlat_bbox(cfg.center_lat, cfg.center_lon, cfg.size_km)
    client = Client.open(
        cfg.stac_url,
        stac_io=StacApiIO(max_retries=STAC_RETRIES),
        timeout=STAC_TIMEOUT,  # must be given here: open() overwrites the stac_io timeout
    )
    search = client.search(
        collections=[cfg.collection],
        bbox=bbox,
        datetime=f"{date_from}/{date_to}",
        query={"eo:cloud_cover": {"lt": cfg.max_cloud_cover if max_cloud is None else max_cloud}},
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


# Sentinel-2 scene classification (SCL) codes.
SCL_CLOUD = (3, 8, 9, 10)  # cloud shadow, cloud medium/high probability, thin cirrus
SCL_SNOW = (11,)


def scl_fractions(cfg: Config, item) -> tuple[float, float] | None:
    """Share of cloud and snow inside the area, from the SCL layer (20 m).

    Returns None if the scene has no SCL asset.
    """
    asset = item.assets.get("scl")
    if asset is None:
        return None
    scl, _ = read_area(cfg, asset.href)
    scl = scl[0]
    valid = scl > 0
    n = max(int(valid.sum()), 1)
    cloud = float(np.isin(scl, SCL_CLOUD).sum() / n)
    snow = float(np.isin(scl, SCL_SNOW).sum() / n)
    return cloud, snow


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
    max_cloud: float | None = None,
    prefer_snow: bool = False,
    max_area_cloud: float = 0.05,
    max_candidates: int = 8,
) -> dict:
    """Download the best valid scene in a date range.

    A scene is valid when it covers the whole area and the SCL layer shows at
    most ``max_area_cloud`` cloud inside the area. Normally the least cloudy
    valid scene is used. With ``prefer_snow``, up to ``max_candidates`` valid
    scenes are compared and the one with most snow wins.

    Args:
        cfg: Project configuration.
        date_from: First date (YYYY-MM-DD).
        date_to: Last date (YYYY-MM-DD).
        out_tif: Where to write the GeoTIFF.
        out_meta: Where to write metadata as JSON.
        exclude_dates: Acquisition dates (YYYY-MM-DD) to skip.
        max_cloud: Scene-level cloud limit for the catalogue search (percent).
        prefer_snow: Pick the snowiest valid scene instead of the clearest.
        max_area_cloud: Largest accepted cloud share inside the area (0 to 1).
        max_candidates: How many valid scenes to compare when preferring snow.

    Returns:
        The metadata that was written.
    """
    for key, value in GDAL_ENV.items():
        os.environ.setdefault(key, value)
    exclude_dates = exclude_dates or set()

    items = [
        it
        for it in search_scenes(cfg, date_from, date_to, max_cloud)
        if it.properties.get("datetime", "")[:10] not in exclude_dates
    ]
    if not items:
        raise SystemExit(
            "No cloud-free scenes found. Try a longer date range or a higher max_cloud_cover."
        )
    order = "most snow" if prefer_snow else "least cloud"
    print(f"{len(items)} candidates. Choosing the valid scene with {order}.")

    chosen = None  # (snow, item, data, profile, cloud_area)
    checked = 0
    for item in items:
        cloud = item.properties.get("eo:cloud_cover")
        day = item.properties.get("datetime", "")[:10]
        data, profile = read_area(cfg, item.assets["visual"].href)
        frac = valid_fraction(data)
        if frac < cfg.min_valid_fraction:
            print(f"  {day}: skipped, only {frac:.1%} valid pixels")
            continue
        fr = scl_fractions(cfg, item)
        cloud_area, snow = fr if fr is not None else (0.0, float("nan"))
        if cloud_area > max_area_cloud:
            print(f"  {day}: skipped, {cloud_area:.0%} cloud over the area")
            continue
        print(f"  {day}: cloud {cloud_area:.0%}, snow {snow:.0%} in the area (scene {cloud:.1f}%)")
        checked += 1
        if chosen is None or (prefer_snow and snow > chosen[0]):
            chosen = (snow, item, data, profile, cloud_area)
        if not prefer_snow or checked >= max_candidates:
            break

    if chosen is None:
        raise SystemExit("No valid scenes in the date range. Try another range.")

    snow, item, data, profile, cloud_area = chosen
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data)
    meta = {
        "scene_id": item.id,
        "datetime": item.properties.get("datetime"),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
        "area_cloud_fraction": round(cloud_area, 3),
        "area_snow_fraction": None if snow != snow else round(snow, 3),
        "crs": str(profile["crs"]),
        "width": profile["width"],
        "height": profile["height"],
        "pixel_size_m": abs(profile["transform"].a),
        "source": item.assets["visual"].href,
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {out_tif}: {meta['datetime'][:10]}, snow {snow:.0%} in the area")
    return meta


def fetch(cfg: Config) -> dict:
    """Download the reference map."""
    return fetch_scene(cfg, cfg.date_from, cfg.date_to, cfg.map_path, cfg.map_meta_path)


def fetch_query_scene(cfg: Config) -> dict:
    """Download a scene from another date than the map, for test images."""
    if not cfg.map_meta_path.exists():
        raise SystemExit("Run 'kjentmann fetch' first, so the map date is known.")
    map_date = json.loads(cfg.map_meta_path.read_text(encoding="utf-8"))["datetime"][:10]
    date_from, date_to, max_cloud = cfg.scene_dates
    print(f"Map is from {map_date}. Fetching test scene ({cfg.scene}) {date_from} to {date_to}.")
    return fetch_scene(
        cfg,
        date_from,
        date_to,
        cfg.query_scene_path,
        cfg.query_scene_meta_path,
        exclude_dates={map_date},
        max_cloud=max_cloud,
        prefer_snow=cfg.scene_prefers_snow,
    )
