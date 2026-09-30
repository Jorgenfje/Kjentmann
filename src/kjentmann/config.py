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
    query_date_from: str = "2025-04-15"
    query_date_to: str = "2025-10-15"
    query_count: int = 200
    query_seed: int = 42
    model_name: str = "vit_small_patch14_dinov2.lvd142m"
    model_image_size: int = 224
    model_batch_size: int = 32
    top_k: int = 5

    # Reference map ---------------------------------------------------------
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

    # Test images -----------------------------------------------------------
    @property
    def query_scene_path(self) -> Path:
        """GeoTIFF from another date, used to cut test images."""
        return self.data_dir / "map" / f"{self.area_name}_query.tif"

    @property
    def query_scene_meta_path(self) -> Path:
        """Metadata for the query scene."""
        return self.data_dir / "map" / f"{self.area_name}_query.json"

    @property
    def queries_dir(self) -> Path:
        """Folder with test images and their ground truth."""
        return self.data_dir / "queries" / self.area_name

    @property
    def queries_csv(self) -> Path:
        """Ground truth position for each test image."""
        return self.queries_dir / "queries.csv"

    # Index and results -----------------------------------------------------
    @property
    def index_dir(self) -> Path:
        """FAISS indexes, one per embedder."""
        return self.data_dir / "index" / self.area_name

    @property
    def results_dir(self) -> Path:
        """Evaluation reports and maps."""
        return self.data_dir / "results" / self.area_name


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
    query = raw.get("query", {})
    model = raw.get("model", {})
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
        query_date_from=str(query.get("date_from", "2025-04-15")),
        query_date_to=str(query.get("date_to", "2025-10-15")),
        query_count=int(query.get("count", 200)),
        query_seed=int(query.get("seed", 42)),
        model_name=str(model.get("name", "vit_small_patch14_dinov2.lvd142m")),
        model_image_size=int(model.get("image_size", 224)),
        model_batch_size=int(model.get("batch_size", 32)),
        top_k=int(model.get("top_k", 5)),
    )
    if not 0 <= cfg.tile_overlap < 1:
        raise ValueError("tiles.overlap must be in [0, 1)")
    if cfg.tile_size_px < 32:
        raise ValueError("tiles.size_px must be at least 32")
    if cfg.size_km <= 0:
        raise ValueError("area.size_km must be positive")
    if cfg.top_k < 1:
        raise ValueError("model.top_k must be at least 1")
    return cfg
