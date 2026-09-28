"""Cut the reference map into overlapping tiles with known positions."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from kjentmann.config import Config
from kjentmann.geo import apply_transform, to_lonlat


@dataclass(frozen=True)
class Tile:
    """One square piece of the map and where it lies on the ground."""

    tile_id: str
    row: int
    col: int
    px_x: int  # left pixel column in the map
    px_y: int  # top pixel row in the map
    center_x: float  # map CRS, metres
    center_y: float
    center_lon: float
    center_lat: float


def tile_origins(length: int, size: int, stride: int) -> list[int]:
    """Start offsets along one axis so tiles cover the full length.

    The last tile is aligned to the edge, so nothing at the border is lost.
    """
    if length < size:
        return []
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def make_tiles(
    image: np.ndarray, transform, crs, size: int, overlap: float
) -> list[tuple[Tile, np.ndarray]]:
    """Split an image (bands, rows, cols) into overlapping square tiles.

    Args:
        image: Array with shape (bands, rows, cols).
        transform: Affine transform from pixel to map coordinates.
        crs: CRS of the map.
        size: Tile side in pixels.
        overlap: Fraction of overlap between neighbours, in [0, 1).

    Returns:
        A list of (Tile, pixels) pairs.
    """
    stride = max(1, int(round(size * (1 - overlap))))
    _, height, width = image.shape
    out = []
    for r, y0 in enumerate(tile_origins(height, size, stride)):
        for c, x0 in enumerate(tile_origins(width, size, stride)):
            cx, cy = apply_transform(transform, x0 + size / 2, y0 + size / 2)
            lon, lat = to_lonlat(crs, cx, cy)
            tile = Tile(
                tile_id=f"r{r:03d}_c{c:03d}",
                row=r,
                col=c,
                px_x=x0,
                px_y=y0,
                center_x=cx,
                center_y=cy,
                center_lon=lon,
                center_lat=lat,
            )
            out.append((tile, image[:, y0 : y0 + size, x0 : x0 + size]))
    return out


def save_tiles(tiles: list[tuple[Tile, np.ndarray]], out_dir: Path, csv_path: Path) -> None:
    """Write each tile as PNG and an index CSV with positions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for tile, pixels in tiles:
        rgb = np.moveaxis(pixels[:3], 0, -1).astype(np.uint8)
        Image.fromarray(rgb).save(out_dir / f"{tile.tile_id}.png")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(tiles[0][0]).keys()))
        writer.writeheader()
        for tile, _ in tiles:
            writer.writerow(asdict(tile))


def build_tiles(cfg: Config) -> list[Tile]:
    """Read the saved map and write its tiles to disk."""
    if not cfg.map_path.exists():
        raise SystemExit(f"Fant ikke {cfg.map_path}. Kjør 'kjentmann fetch' først.")
    with rasterio.open(cfg.map_path) as src:
        image = src.read()
        tiles = make_tiles(image, src.transform, src.crs, cfg.tile_size_px, cfg.tile_overlap)
    if not tiles:
        raise SystemExit("Kartet er mindre enn én rute. Øk area.size_km eller senk tiles.size_px.")
    save_tiles(tiles, cfg.tiles_dir, cfg.tiles_csv)
    ground_km = cfg.tile_size_px * abs(src.transform.a) / 1000
    print(f"Lagret {len(tiles)} ruter à {ground_km:.2f} km i {cfg.tiles_dir}")
    return [t for t, _ in tiles]
