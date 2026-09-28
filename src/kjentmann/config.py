"""Load and validate the project configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    """Typed view of config.yaml."""

    area_name: str
    center_lat: float
    center_lon: float
    size_km: float
    collection: str
    stac_url: str
    date_from: str
    date_to: str
    max_cloud_cover: float
    min_valid_fraction: float
    tile_size_px: int
    tile_overlap: float
    data_dir: Path

    @property
    def map_path(self) -> Path:
        """GeoTIFF with the reference map for the area."""
        return self.data_dir / "map" / f"{self.area_name}.tif"

    @property
    def map_meta_path(self) -> Path:
        """JSON with metadata about the scene the map came from."""
        return self.data_dir / "map" / f"{self.area_name}.json"

    @property
    def tiles_dir(self) -> Path:
        """Folder with one PNG per tile."""
        return self.data_dir / "tiles" / self.area_name

    @property
    def tiles_csv(self) -> Path:
        """Index of all tiles with their geographic position."""
        return self.tiles_dir / "tiles.csv"


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read config.yaml and return a Config.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If a value is outside its allowed range.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg = Config(
        area_name=raw["area"]["name"],
        center_lat=float(raw["area"]["center_lat"]),
        center_lon=float(raw["area"]["center_lon"]),
        size_km=float(raw["area"]["size_km"]),
        collection=raw["imagery"]["collection"],
        stac_url=raw["imagery"]["stac_url"],
        date_from=str(raw["imagery"]["date_from"]),
        date_to=str(raw["imagery"]["date_to"]),
        max_cloud_cover=float(raw["imagery"]["max_cloud_cover"]),
        min_valid_fraction=float(raw["imagery"]["min_valid_fraction"]),
        tile_size_px=int(raw["tiles"]["size_px"]),
        tile_overlap=float(raw["tiles"]["overlap"]),
        data_dir=Path(raw["paths"]["data_dir"]),
    )
    if not 0 <= cfg.tile_overlap < 1:
        raise ValueError("tiles.overlap must be in [0, 1)")
    if cfg.tile_size_px < 32:
        raise ValueError("tiles.size_px must be at least 32")
    if cfg.size_km <= 0:
        raise ValueError("area.size_km must be positive")
    return cfg
