"""Cut test images with known positions from a scene of another date.

Each test image is a square crop at a random position, not aligned to the tile
grid. Its true centre is recorded, so the search can be scored against it.

The 'easy' profile gives plain north-up crops at map scale. Other profiles add
what a real camera would see: a heading error (rotation), unknown altitude
(scale), blur and sensor noise. The true centre is unchanged by these.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer

from kjentmann.config import Config
from kjentmann.geo import apply_transform


@dataclass(frozen=True)
class Query:
    """A test image and where it was really taken."""

    query_id: str
    center_lon: float
    center_lat: float
    center_x: float  # in the reference map CRS, metres
    center_y: float
    rotation_deg: float = 0.0
    scale: float = 1.0  # ground covered, relative to a map tile


def sample_offsets(
    height: int, width: int, size: int, count: int, seed: int, min_valid: np.ndarray | None = None
) -> list[tuple[int, int]]:
    """Random top-left offsets for crops that fit inside the image.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        size: Crop side in pixels.
        count: Number of crops wanted.
        seed: Random seed, so the test set is reproducible.
        min_valid: Optional boolean mask of valid pixels; crops touching
            no-data are skipped.

    Returns:
        A list of (row, col) offsets.
    """
    if height < size or width < size:
        return []
    rng = np.random.default_rng(seed)
    out: list[tuple[int, int]] = []
    attempts = 0
    while len(out) < count and attempts < count * 20:
        attempts += 1
        r = int(rng.integers(0, height - size + 1))
        c = int(rng.integers(0, width - size + 1))
        if min_valid is not None and not min_valid[r : r + size, c : c + size].all():
            continue
        out.append((r, c))
    return out


def build_queries(cfg: Config) -> list[Query]:
    """Cut test images from the query scene and write them with ground truth."""
    if not cfg.query_scene_path.exists():
        raise SystemExit("Fant ikke testscenen. Kjør 'kjentmann fetch-query' først.")
    with rasterio.open(cfg.map_path) as ref:
        map_crs = ref.crs
    size = cfg.tile_size_px

    with rasterio.open(cfg.query_scene_path) as src:
        image = src.read()
        transform, crs = src.transform, src.crs
    valid = image.max(axis=0) > 0
    perturbed = cfg.rotation_deg > 0 or cfg.scale_min != 1 or cfg.scale_max != 1
    # Room for the largest rotated, scaled footprint around each centre.
    span = int(np.ceil(size * cfg.scale_max * (1.415 if cfg.rotation_deg else 1))) + 2
    span = span if perturbed else size
    offsets = sample_offsets(
        image.shape[1], image.shape[2], span, cfg.query_count, cfg.query_seed, valid
    )
    rng = np.random.default_rng(cfg.query_seed + 1)
    rgb = np.ascontiguousarray(np.moveaxis(image[:3], 0, -1).astype(np.uint8))

    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_map = Transformer.from_crs(crs, map_crs, always_xy=True)

    cfg.queries_dir.mkdir(parents=True, exist_ok=True)
    queries = []
    for i, (r, c) in enumerate(offsets):
        cx, cy = c + span / 2, r + span / 2
        angle = float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
        scale = float(rng.uniform(cfg.scale_min, cfg.scale_max))
        crop = render_view(rgb, cx, cy, size, angle, scale)
        crop = degrade(crop, cfg.blur_sigma, cfg.noise_std, rng)
        x, y = apply_transform(transform, cx, cy)
        lon, lat = to_lonlat.transform(x, y)
        mx, my = to_map.transform(x, y)
        q = Query(f"q{i:04d}", lon, lat, mx, my, round(angle, 2), round(scale, 3))
        Image.fromarray(crop).save(cfg.queries_dir / f"{q.query_id}.png")
        queries.append(q)

    write_queries(queries, cfg.queries_csv)
    print(f"Lagret {len(queries)} testbilder i {cfg.queries_dir}")
    return queries


def render_view(
    rgb: np.ndarray, cx: float, cy: float, size: int, angle_deg: float, scale: float
) -> np.ndarray:
    """What a camera centred on (cx, cy) sees, rotated and at a given scale.

    Each output pixel covers ``scale`` map pixels. With angle 0 and scale 1
    this is an exact crop.
    """
    if angle_deg == 0 and scale == 1:
        x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
        return rgb[y0 : y0 + size, x0 : x0 + size].copy()
    a = np.deg2rad(angle_deg)
    cos, sin = np.cos(a) * scale, np.sin(a) * scale
    # OpenCV works in pixel-index coordinates, where the centre of pixel i is
    # at i. (cx, cy) is continuous (pixel i spans [i, i + 1)), so shift by 0.5.
    h = (size - 1) / 2
    px, py = cx - 0.5, cy - 0.5
    # Output pixel (u, v) -> source pixel, rotating about the image centre.
    m = np.array(
        [[cos, -sin, px - (cos * h - sin * h)], [sin, cos, py - (sin * h + cos * h)]],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        rgb,
        m,
        (size, size),
        flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )


def degrade(img: np.ndarray, blur_sigma: float, noise_std: float, rng) -> np.ndarray:
    """Add lens blur and sensor noise."""
    out = img
    if blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    if noise_std > 0:
        noisy = out.astype(np.float32) + rng.normal(0, noise_std, out.shape)
        out = np.clip(noisy, 0, 255).astype(np.uint8)
    return out


def write_queries(queries: list[Query], path: Path) -> None:
    """Write the ground truth CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(queries[0]).keys()))
        writer.writeheader()
        for q in queries:
            writer.writerow(asdict(q))


def read_queries(path: Path) -> list[Query]:
    """Read the ground truth CSV."""
    with path.open(encoding="utf-8") as f:
        return [
            Query(
                row["query_id"],
                float(row["center_lon"]),
                float(row["center_lat"]),
                float(row["center_x"]),
                float(row["center_y"]),
                float(row.get("rotation_deg") or 0),
                float(row.get("scale") or 1),
            )
            for row in csv.DictReader(f)
        ]
