"""Interactive map to check that the map and tiles line up with the real world."""

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


def _overlay_png(cfg: Config) -> tuple[str, list[list[float]]]:
    """Reproject the map to lon/lat and return it as a data URL plus its bounds."""
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
    west, north = apply_transform(transform, 0, 0)
    east, south = apply_transform(transform, width, height)
    buf = io.BytesIO()
    Image.fromarray(np.moveaxis(out, 0, -1)).save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return url, [[south, west], [north, east]]


def build_map(cfg: Config) -> None:
    """Write data/<area>_map.html with the satellite image and the tile grid."""
    with rasterio.open(cfg.map_path) as src:
        crs, transform = src.crs, src.transform
    size = cfg.tile_size_px

    m = folium.Map(location=[cfg.center_lat, cfg.center_lon], zoom_start=12, tiles=None)
    # OpenStreetMap blocks tile requests from local files (no Referer header),
    # so use Kartverket's open maps. No API key needed.
    for layer, name in (("topo", "Kartverket topo"), ("topograatone", "Kartverket gråtone")):
        folium.TileLayer(
            tiles=f"https://cache.kartverket.no/v1/wmts/1.0.0/{layer}/default/webmercator/{{z}}/{{y}}/{{x}}.png",
            attr='&copy; <a href="https://www.kartverket.no/">Kartverket</a>',
            name=name,
            max_zoom=18,
        ).add_to(m)

    url, bounds = _overlay_png(cfg)
    folium.raster_layers.ImageOverlay(
        image=url, bounds=bounds, opacity=0.85, name="Sentinel-2"
    ).add_to(m)

    grid = folium.FeatureGroup(name="Ruter", show=True)
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
    print(f"Lagret kart: {out}  (åpne i nettleseren)")
