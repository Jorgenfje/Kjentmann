"""Small geographic helpers shared by the pipeline."""

from __future__ import annotations

from pyproj import CRS, Transformer


def utm_crs_for(lat: float, lon: float) -> CRS:
    """Return the WGS84 UTM zone CRS that contains a point.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        The UTM CRS (EPSG:326xx north, EPSG:327xx south).
    """
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def square_bounds(
    center_lat: float, center_lon: float, size_km: float, crs: CRS
) -> tuple[float, float, float, float]:
    """Square area around a point, expressed in a projected CRS.

    Args:
        center_lat: Latitude of the centre.
        center_lon: Longitude of the centre.
        size_km: Side length in kilometres.
        crs: Projected CRS in metres (for example UTM).

    Returns:
        (min_x, min_y, max_x, max_y) in the units of ``crs``.
    """
    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = to_crs.transform(center_lon, center_lat)
    half = size_km * 1000 / 2
    return cx - half, cy - half, cx + half, cy + half


def to_lonlat(crs: CRS | str, x: float, y: float) -> tuple[float, float]:
    """Convert a point from ``crs`` to (lon, lat)."""
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)


def lonlat_bbox(
    center_lat: float, center_lon: float, size_km: float
) -> tuple[float, float, float, float]:
    """Bounding box in lon/lat for a square area, used for catalogue search."""
    crs = utm_crs_for(center_lat, center_lon)
    min_x, min_y, max_x, max_y = square_bounds(center_lat, center_lon, size_km, crs)
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = t.transform([min_x, max_x, max_x, min_x], [min_y, min_y, max_y, max_y])
    return min(lons), min(lats), max(lons), max(lats)


def apply_transform(transform, col: float, row: float) -> tuple[float, float]:
    """Pixel (col, row) to map (x, y) with an affine transform.

    Written out by hand to avoid version differences in the ``affine`` package.
    """
    t = transform
    return t.a * col + t.b * row + t.c, t.d * col + t.e * row + t.f
