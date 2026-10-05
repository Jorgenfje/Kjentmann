"""Interactive maps: check the tile grid, and show where each search landed."""

from __future__ import annotations

import base64
import csv
import io

import folium
import numpy as np
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

from kjentmann.config import Config
from kjentmann.geo import apply_transform, to_lonlat

KARTVERKET_URL = (
    "https://cache.kartverket.no/v1/wmts/1.0.0/{layer}/default/webmercator/{{z}}/{{y}}/{{x}}.png"
)


def _base_map(cfg: Config) -> folium.Map:
    """Folium map with Kartverket base layers.

    OpenStreetMap blocks tile requests from local files (no Referer header),
    and CartoDB needs an API key, so Kartverket's open maps are used.
    """
    m = folium.Map(location=[cfg.center_lat, cfg.center_lon], zoom_start=12, tiles=None)
    for layer, name in (("topo", "Kartverket topo"), ("topograatone", "Kartverket greyscale")):
        folium.TileLayer(
            tiles=KARTVERKET_URL.format(layer=layer),
            attr='&copy; <a href="https://www.kartverket.no/">Kartverket</a>',
            name=name,
            max_zoom=18,
        ).add_to(m)
    return m


def _overlay_png(cfg: Config) -> tuple[str, list[list[float]]]:
    """Reproject the map to lon/lat and return it as a PNG data URL plus its bounds.

    Pixels outside the image after reprojection are made transparent, so no
    black frame is drawn around it.
    """
    with rasterio.open(cfg.map_path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        out = np.zeros((3, height, width), dtype=np.uint8)
        for band in range(3):
            reproject(
                source=rasterio.band(src, band + 1),
                destination=out[band],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
    alpha = np.where(out.max(axis=0) > 0, 255, 0).astype(np.uint8)
    rgba = np.dstack([np.moveaxis(out, 0, -1), alpha])
    west, north = apply_transform(transform, 0, 0)
    east, south = apply_transform(transform, width, height)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return url, [[south, west], [north, east]]


def _add_satellite(m: folium.Map, cfg: Config, show: bool = True) -> None:
    url, bounds = _overlay_png(cfg)
    folium.raster_layers.ImageOverlay(
        image=url, bounds=bounds, opacity=0.85, name="Sentinel-2", show=show
    ).add_to(m)


def build_map(cfg: Config) -> None:
    """Write data/<area>_map.html with the satellite image and the tile grid."""
    with rasterio.open(cfg.map_path) as src:
        crs, transform = src.crs, src.transform
    size = cfg.tile_size_px

    m = _base_map(cfg)
    _add_satellite(m, cfg)

    grid = folium.FeatureGroup(name="Tiles", show=True)
    with cfg.tiles_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x0, y0 = int(row["px_x"]), int(row["px_y"])
            corners = [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)]
            latlon = [to_lonlat(crs, *apply_transform(transform, *c))[::-1] for c in corners]
            folium.Polygon(
                latlon,
                color="#ffcc00",
                weight=1,
                fill=False,
                tooltip=f"{row['tile_id']}  ({float(row['center_lat']):.4f}, "
                f"{float(row['center_lon']):.4f})",
            ).add_to(grid)
    grid.add_to(m)

    folium.LayerControl().add_to(m)
    out = cfg.data_dir / f"{cfg.area_name}_map.html"
    m.save(str(out))
    print(f"Saved map: {out}  (open in a browser)")


def build_results_map(cfg: Config, embedder_name: str, rows: list[dict]) -> None:
    """Map of all test images: green = found, red = missed, line to the guess."""
    m = _base_map(cfg)
    _add_satellite(m, cfg, show=False)

    hits = folium.FeatureGroup(name="Found first")
    near = folium.FeatureGroup(name=f"In top {cfg.top_k}")
    misses = folium.FeatureGroup(name="Missed")
    for r in rows:
        rank = r["rank_of_correct"]
        if rank == 1:
            color, group = "#1a9850", hits
        elif rank:
            color, group = "#fdae61", near
        else:
            color, group = "#d73027", misses
        true_pt = [r["true_lat"], r["true_lon"]]
        guess_pt = [r["top1_lat"], r["top1_lon"]]
        tip = (
            f"{r['query_id']}: rank {rank or 'not found'}, "
            f"error {r['top1_error_m'] / 1000:.2f} km, score {r['top1_score']:.2f}"
        )
        folium.CircleMarker(
            true_pt, radius=5, color=color, fill=True, fill_opacity=0.9, tooltip=tip
        ).add_to(group)
        if rank != 1:
            folium.PolyLine([true_pt, guess_pt], color=color, weight=1, opacity=0.6).add_to(group)
    for g in (hits, near, misses):
        g.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out = cfg.results_dir / f"{embedder_name}_map.html"
    m.save(str(out))
    print(f"Saved results map: {out}")


def build_refine_map(
    cfg: Config, matcher_name: str, rows: list[dict], filename: str | None = None
) -> None:
    """Map after fine matching: colour by error in metres, grey for "unknown"."""
    m = _base_map(cfg)
    _add_satellite(m, cfg, show=False)

    groups = {
        "good": folium.FeatureGroup(name="Error under 100 m"),
        "ok": folium.FeatureGroup(name="Error 100-500 m"),
        "bad": folium.FeatureGroup(name="Error over 500 m"),
        "unknown": folium.FeatureGroup(name="Unknown"),
    }
    colors = {"good": "#1a9850", "ok": "#fdae61", "bad": "#d73027", "unknown": "#888888"}
    for r in rows:
        answered = r["error_m"] != "" and r["inliers"] >= cfg.min_inliers
        if not answered:
            kind = "unknown"
        elif r["error_m"] <= 100:
            kind = "good"
        elif r["error_m"] <= 500:
            kind = "ok"
        else:
            kind = "bad"
        true_pt = [r["true_lat"], r["true_lon"]]
        err = f"{r['error_m']:.0f} m" if r["error_m"] != "" else "no position"
        tip = f"{r['query_id']}: {err}, {r['inliers']} inliers"
        folium.CircleMarker(
            true_pt, radius=5, color=colors[kind], fill=True, fill_opacity=0.9, tooltip=tip
        ).add_to(groups[kind])
        if answered and kind != "good":
            folium.PolyLine(
                [true_pt, [r["est_lat"], r["est_lon"]]], color=colors[kind], weight=1, opacity=0.7
            ).add_to(groups[kind])
    for g in groups.values():
        g.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    out = cfg.results_dir / (filename or f"refine_{matcher_name}_map.html")
    m.save(str(out))
    print(f"Saved results map: {out}")
