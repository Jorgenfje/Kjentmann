"""Cut test images with known positions from a scene of another date.

Each test image is a square crop at a random position, not aligned to the tile
grid. Its true centre is recorded, so the search can be scored against it.
Images are north-up: in a real system the heading comes from a compass.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

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
    offsets = sample_offsets(
        image.shape[1], image.shape[2], size, cfg.query_count, cfg.query_seed, valid
    )

    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_map = Transformer.from_crs(crs, map_crs, always_xy=True)

    cfg.queries_dir.mkdir(parents=True, exist_ok=True)
    queries = []
    for i, (r, c) in enumerate(offsets):
        x, y = apply_transform(transform, c + size / 2, r + size / 2)
        lon, lat = to_lonlat.transform(x, y)
        mx, my = to_map.transform(x, y)
        q = Query(f"q{i:04d}", lon, lat, mx, my)
        crop = np.moveaxis(image[:3, r : r + size, c : c + size], 0, -1).astype(np.uint8)
        Image.fromarray(crop).save(cfg.queries_dir / f"{q.query_id}.png")
        queries.append(q)

    write_queries(queries, cfg.queries_csv)
    print(f"Lagret {len(queries)} testbilder i {cfg.queries_dir}")
    return queries


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
            )
            for row in csv.DictReader(f)
        ]
