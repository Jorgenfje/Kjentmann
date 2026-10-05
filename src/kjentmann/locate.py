"""Locate one photo taken from the air, anywhere in the world.

    kjentmann locate photo.jpg --near "Gardermoen" --altitude 3000

1. The rough position (place name or "lat,lon") sets the search area.
2. A Sentinel-2 map of that area is downloaded once and cached.
3. The photo is scaled to the map's 10 m pixels using altitude and field of view.
4. Heading and exact altitude are unknown, so several rotations and scales are
   tried, nearest map windows first, until one match is confident.
5. The result: position, an accuracy estimate, heading and altitude. If the photo
   has GPS coordinates in its EXIF data, the error is reported as well.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from kjentmann.config import Config
from kjentmann.match import Matcher, verify
from kjentmann.refine import transform_scale, window_around

ROTATIONS = (0, 90, 180, 270, 45, 135, 225, 315)
ZOOMS = (1.0, 0.8, 1.25, 0.65, 1.55)


def square(photo: np.ndarray) -> np.ndarray:
    """Centre square of the photo (rotations then keep the same frame)."""
    h, w = photo.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return np.ascontiguousarray(photo[y0 : y0 + s, x0 : x0 + s])


def view_matrix(side: int, angle_deg: float, zoom: float) -> np.ndarray:
    """2x3 map from view pixel to photo pixel: rotate about the centre and zoom.

    ``zoom`` < 1 shows a smaller area (closer), > 1 a larger area (higher up).
    """
    c = (side - 1) / 2
    a = math.radians(angle_deg)
    cos, sin = math.cos(a) * zoom, math.sin(a) * zoom
    return np.array([[cos, -sin, c - (cos * c - sin * c)], [sin, cos, c - (sin * c + cos * c)]])


def make_view(photo_sq: np.ndarray, angle_deg: float, zoom: float) -> np.ndarray:
    """The square photo rotated and zoomed. Outside the photo is black, never mirrored,
    so no invented content can produce a false match."""
    import cv2

    side = photo_sq.shape[0]
    if angle_deg == 0 and zoom == 1:
        return photo_sq
    return cv2.warpAffine(
        photo_sq,
        view_matrix(side, angle_deg, zoom),
        (side, side),
        flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def photo_to_view(side: int, angle_deg: float, zoom: float, x: float, y: float):
    """Where a photo pixel ends up in the view (inverse of ``view_matrix``)."""
    m = view_matrix(side, angle_deg, zoom)
    a, t = m[:, :2], m[:, 2]
    u = np.linalg.solve(a, np.array([x, y]) - t)
    return float(u[0]), float(u[1])


@dataclass
class Fix:
    """The answer for one photo."""

    found: bool
    lat: float | None = None
    lon: float | None = None
    accuracy_m: float | None = None
    heading_deg: float | None = None  # direction the top of the photo points, 0 = north
    altitude_m: float | None = None  # altitude implied by the matched scale
    inliers: int = 0
    seconds: float = 0.0
    windows_tried: int = 0
    exif_lat: float | None = None
    exif_lon: float | None = None
    error_m: float | None = None  # distance to the EXIF position, if any
    footprint: list | None = None  # photo corners as [lat, lon]


# --- inputs --------------------------------------------------------------------


def parse_near(text: str) -> tuple[float, float, str]:
    """'59.58,11.16' or a place name -> (lat, lon, label)."""
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", text)
    if m:
        return float(m.group(1)), float(m.group(2)), text.strip()
    return geocode(text)


def geocode_candidates(place: str, limit: int = 5) -> list[tuple[float, float, str]]:
    """Places matching a name, most prominent first (OpenStreetMap Nominatim)."""
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"q": place, "format": "json", "limit": limit}
    )
    req = urllib.request.Request(url, headers={"User-Agent": "kjentmann/0.7 (GitHub)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        hits = json.loads(r.read())
    return [(float(h["lat"]), float(h["lon"]), h.get("display_name", place)) for h in hits]


def geocode(place: str) -> tuple[float, float, str]:
    """The most prominent place with this name."""
    hits = geocode_candidates(place, limit=1)
    if not hits:
        raise SystemExit(f"Place '{place}' not found. Try coordinates, e.g. 59.58,11.16")
    return hits[0]


def exif_position(path: Path) -> tuple[float, float] | None:
    """GPS position stored in the photo, if any."""
    try:
        gps = Image.open(path).getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None

    def deg(v) -> float:
        d, m, s = (float(x) for x in v)
        return d + m / 60 + s / 3600

    lat, lon = deg(gps[2]), deg(gps[4])
    if gps.get(1) == "S":
        lat = -lat
    if gps.get(3) == "W":
        lon = -lon
    return lat, lon


def load_photo(path: Path) -> np.ndarray:
    """RGB array, upright according to EXIF orientation."""
    return np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))


def to_map_scale(
    photo: np.ndarray, altitude_m: float, fov_deg: float, pixel_m: float = 10.0
) -> np.ndarray:
    """Resize so one photo pixel covers ``pixel_m`` on the ground (camera pointing down)."""
    h, w = photo.shape[:2]
    ground_w = 2 * altitude_m * math.tan(math.radians(fov_deg) / 2)
    factor = (ground_w / w) / pixel_m
    nw, nh = max(32, round(w * factor)), max(32, round(h * factor))
    return np.asarray(Image.fromarray(photo).resize((nw, nh), Image.LANCZOS))


# --- map -----------------------------------------------------------------------


def area_config(base: Config, lat: float, lon: float, radius_km: float, photo_px: int) -> Config:
    """A config for the area around (lat, lon), with its own cached map."""
    size_km = 2 * radius_km + 6  # room for windows at the edge of the circle
    name = f"loc_{lat:.3f}_{lon:.3f}_{size_km:g}km".replace("-", "m")
    # Windows must be clearly larger than the photo at map scale.
    win_scale = max(2.0, math.ceil(photo_px * 1.6 / base.tile_size_px))
    return replace(
        base,
        area_name=name,
        center_lat=lat,
        center_lon=lon,
        size_km=size_km,
        date_from=base.locate_date_from,
        date_to=base.locate_date_to,
        nav_window_scale=float(win_scale),
    )


def _year_back(date: str, years: int) -> str:
    return f"{int(date[:4]) - years}{date[4:]}"


def map_attempts(cfg: Config) -> list[tuple[str, str, float]]:
    """(date_from, date_to, scene cloud limit) to try, strictest first.

    The scene cloud figure covers a whole 110 km scene, so a cloudy coast can
    hide a clear area. The cloud mask then checks the area itself.
    """
    out = [(cfg.date_from, cfg.date_to, cfg.max_cloud_cover)]
    for years in (0, 1, 2):
        out.append((_year_back(cfg.date_from, years), _year_back(cfg.date_to, years), 60.0))
    return out


def ensure_map(cfg: Config) -> None:
    """Download the map and cut tiles, unless already cached."""
    if not cfg.map_path.exists():
        from kjentmann.fetch import NoSceneError, fetch_scene

        print(f"Downloading satellite map ({cfg.size_km:g} × {cfg.size_km:g} km) ...")
        for date_from, date_to, max_cloud in map_attempts(cfg):
            try:
                fetch_scene(
                    cfg, date_from, date_to, cfg.map_path, cfg.map_meta_path, max_cloud=max_cloud
                )
                break
            except NoSceneError:
                continue
        else:
            raise NoSceneError(
                "No cloud-free satellite image of this area in the last three summers."
            )
    if not cfg.tiles_csv.exists():
        from kjentmann.tiles import build_tiles

        build_tiles(cfg)


# --- search --------------------------------------------------------------------


def locate_photo(
    base: Config,
    matcher: Matcher,
    photo_path: Path,
    near: str,
    altitude_m: float,
    radius_km: float | None = None,
    fov_deg: float | None = None,
    progress=None,
) -> Fix:
    """Find where a photo was taken. See the module docstring.

    ``progress``, if given, is called with the share of the search done (0 to 1).
    """
    import rasterio

    from kjentmann.evaluate import read_tiles
    from kjentmann.geo import apply_transform, to_lonlat
    from kjentmann.navigate import search_windows

    t0 = time.perf_counter()
    radius_km = radius_km or base.locate_radius_km
    fov_deg = fov_deg or base.locate_fov_deg
    lat, lon, label = parse_near(near)
    print(f"Searching within {radius_km:g} km of {label} ({lat:.4f}, {lon:.4f})")

    photo = square(to_map_scale(load_photo(photo_path), altitude_m, fov_deg))
    # Cache keys must be unique per photo and per map, since the matcher outlives one search.
    photo_id = hashlib.sha1(photo.tobytes()).hexdigest()[:16]
    side = photo.shape[0]
    cfg = area_config(base, lat, lon, radius_km, side)
    ensure_map(cfg)

    with rasterio.open(cfg.map_path) as src:
        image = src.read()
        map_tf, map_crs = src.transform, src.crs
    pixel_m = abs(map_tf.a)
    win_px = int(round(cfg.tile_size_px * cfg.nav_window_scale))
    from pyproj import Transformer

    cx, cy = Transformer.from_crs("EPSG:4326", map_crs, always_xy=True).transform(lon, lat)
    reach = radius_km * 1000 + win_px * pixel_m / math.sqrt(2)
    cands = sorted(
        (
            c
            for c in search_windows(cfg, read_tiles(cfg.tiles_csv), map_tf)
            if math.hypot(c.cx_m - cx, c.cy_m - cy) <= reach
        ),
        key=lambda c: math.hypot(c.cx_m - cx, c.cy_m - cy),
    )
    print(f"{len(cands)} map windows, {len(ROTATIONS)} headings, {len(ZOOMS)} altitudes ...")

    best = None  # (inliers, v, win, angle, zoom, pa, pb)
    tried = 0
    total = len(ZOOMS) * len(ROTATIONS) * len(cands)
    windows: dict = {}
    done = False
    # Altitude guess first (zoom 1), every heading; then the other altitudes.
    for zoom in ZOOMS:
        for angle in ROTATIONS:
            view = make_view(photo, angle, zoom)
            for c in cands:
                if c.key not in windows:
                    windows[c.key] = window_around(image, c.cx_px, c.cy_px, win_px)
                win = windows[c.key]
                pa, pb = matcher.match(
                    view,
                    win.pixels,
                    b_key=(cfg.area_name, c.key, win_px),
                    a_key=(photo_id, angle, zoom),
                )
                v = verify(pa, pb, cfg.ransac_px)
                s = transform_scale(v)
                inl = v.inliers if 0.6 <= s <= 1.7 else 0
                tried += 1
                if progress and tried % 10 == 0:
                    progress(tried / total)
                if best is None or inl > best[0]:
                    best = (inl, v, win, angle, zoom, pa, pb)
                if inl >= cfg.early_stop_inliers:
                    done = True
                    break
            if done:
                break
        if done:
            break

    fix = Fix(found=False, windows_tried=tried)
    exif = exif_position(photo_path)
    if exif:
        fix.exif_lat, fix.exif_lon = exif
    fix.seconds = round(time.perf_counter() - t0, 1)
    if best is None or best[0] < cfg.min_inliers:
        fix.inliers = best[0] if best else 0
        return fix

    inliers, v, win, angle, zoom, pa, pb = best
    m = v.transform

    def photo_ll(x: float, y: float) -> tuple[float, float]:
        """Photo pixel -> (lat, lon) through view, window and map."""
        u, w_ = photo_to_view(side, angle, zoom, x, y)
        px, py = v.map_point(u, w_)
        mx, my = apply_transform(map_tf, win.x0 + px, win.y0 + py)
        lo, la = to_lonlat(map_crs, mx, my)
        return la, lo

    fix.found = True
    fix.inliers = inliers
    c = (side - 1) / 2
    fix.lat, fix.lon = photo_ll(c, c)
    # Accuracy: spread of the agreeing matches around the fitted transform (10 m pixels).
    pred = pa @ m[:, :2].T + m[:, 2]
    resid = np.hypot(*(pred - pb).T)
    good = resid[resid < cfg.ransac_px]
    rms_px = float(np.sqrt(np.mean(good**2))) if len(good) else cfg.ransac_px
    fix.accuracy_m = round(max(pixel_m, 2 * rms_px * pixel_m), 1)
    # Heading: direction of the photo's top edge on the (north-up) map.
    top = photo_ll(c, 0)
    north = (top[0] - fix.lat) * 111_320
    east = (top[1] - fix.lon) * 111_320 * math.cos(math.radians(fix.lat))
    fix.heading_deg = round(math.degrees(math.atan2(east, north)) % 360)
    fix.altitude_m = round(altitude_m * transform_scale(v) / zoom, -1)
    fix.footprint = [list(photo_ll(x, y)) for x, y in ((0, 0), (side, 0), (side, side), (0, side))]
    if exif:
        from pyproj import Geod

        _, _, d = Geod(ellps="WGS84").inv(fix.lon, fix.lat, exif[1], exif[0])
        fix.error_m = round(d, 1)
    return fix


# --- output --------------------------------------------------------------------


def describe(fix: Fix) -> str:
    """Short summary for the terminal and the web page."""
    if not fix.found:
        return (
            f"Position not found ({fix.inliers} agreeing matches, {fix.windows_tried} tries). "
            "Try another altitude, a larger radius, or a photo taken more straight down. "
            "Photos showing the horizon are not supported yet."
        )
    lines = [
        f"Position: {fix.lat:.5f}, {fix.lon:.5f}  (± {fix.accuracy_m:.0f} m)",
        f"Photo heading {fix.heading_deg:.0f}° (0 = north), "
        f"estimated altitude {fix.altitude_m:.0f} m",
        f"{fix.inliers} agreeing matches, {fix.windows_tried} tries, {fix.seconds:.1f} s",
    ]
    if fix.error_m is not None:
        lines.append(f"Distance to the GPS position stored in the photo: {fix.error_m:.0f} m")
    return "\n".join(lines)


def result_map(
    fix: Fix, cfg: Config, out: Path, near_latlon=None, radius_km: float | None = None
) -> Path:
    """HTML map: search area, and if found the position, accuracy and photo footprint."""
    import folium

    from kjentmann.viz import _base_map

    near = list(near_latlon or (cfg.center_lat, cfg.center_lon))
    center = [fix.lat, fix.lon] if fix.found else near
    m = _base_map(replace(cfg, center_lat=center[0], center_lon=center[1]))
    m.location = center
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Aerial imagery (Esri)",
    ).add_to(m)
    if radius_km:
        folium.Circle(
            near,
            radius=radius_km * 1000,
            color="#888888",
            weight=2,
            dash_array="8 6",
            fill=False,
            tooltip=f"Search area: {radius_km:g} km around the position you gave",
        ).add_to(m)
        folium.CircleMarker(
            near, radius=4, color="#888888", fill=True, tooltip="Your rough position"
        ).add_to(m)
    if fix.found:
        folium.Polygon(
            fix.footprint, color="#ffcc00", weight=2, fill=False, tooltip="Your photo"
        ).add_to(m)
        folium.Circle(
            center, radius=fix.accuracy_m, color="#1a9850", fill=True, fill_opacity=0.2
        ).add_to(m)
        folium.Marker(
            center,
            tooltip=f"Kjentmann: here, ± {fix.accuracy_m:.0f} m",
            icon=folium.Icon(color="green"),
        ).add_to(m)
    if fix.exif_lat is not None:
        folium.Marker(
            [fix.exif_lat, fix.exif_lon],
            tooltip="GPS position stored in the photo",
            icon=folium.Icon(color="blue"),
        ).add_to(m)
    folium.LayerControl().add_to(m)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out))
    return out


def locate_cli(
    base: Config,
    matcher: Matcher,
    photo: str,
    near: str,
    altitude: float,
    radius_km: float | None,
    fov_deg: float | None,
) -> Fix:
    path = Path(photo)
    fix = locate_photo(base, matcher, path, near, altitude, radius_km, fov_deg)
    print()
    print(describe(fix))
    out = base.data_dir / "locate" / f"{path.stem}_map.html"
    lat, lon, _ = parse_near(near)
    result_map(fix, base, out, (lat, lon), radius_km or base.locate_radius_km)
    (out.with_suffix(".json")).write_text(json.dumps(asdict(fix), indent=2), encoding="utf-8")
    print(f"\nMap: {out}")
    return fix
