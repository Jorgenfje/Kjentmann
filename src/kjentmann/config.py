"""Load and validate the project configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    profile: str = "easy"
    rotation_deg: float = 0.0
    scale_min: float = 1.0
    scale_max: float = 1.0
    blur_sigma: float = 0.0
    noise_std: float = 0.0
    profiles: dict = field(default_factory=dict)
    scene: str = "default"
    scenes: dict = field(default_factory=dict)
    season_radius_km: float = 5.0
    spoof_radius_km: float = 5.0
    spoof_threshold_m: float = 100.0
    gps_noise_m: float = 5.0
    spoof_offsets_m: tuple = (50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)
    flight_steps: int = 60
    flight_step_m: float = 250.0
    spoof_start_step: int = 20
    drift_m_per_step: float = 25.0
    track_radius_km: float = 2.0
    locate_radius_km: float = 5.0
    locate_fov_deg: float = 70.0
    locate_date_from: str = "2025-05-01"
    locate_date_to: str = "2025-09-30"
    model_name: str = "vit_small_patch14_dinov2.lvd142m"
    model_image_size: int = 224
    model_batch_size: int = 32
    top_k: int = 5
    matcher: str = "lightglue"
    window_scale: float = 1.5
    upscale: int = 2
    max_keypoints: int = 2048
    ransac_px: float = 3.0
    min_inliers: int = 15
    min_scale: float = 0.7
    max_scale: float = 1.4
    tta_rotations: tuple = (0.0,)
    tta_scales: tuple = (1.0,)
    fine_rotations: tuple = ()
    early_stop_inliers: int = 40
    nav_radii_km: tuple = (2.0, 5.0, 10.0)
    nav_window_scale: float = 2.0
    nav_map_radius_km: float = 5.0

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
        return self.data_dir / "map" / f"{self.area_name}_query{self._scene_suffix}.tif"

    @property
    def query_scene_meta_path(self) -> Path:
        """Metadata for the query scene."""
        return self.data_dir / "map" / f"{self.area_name}_query{self._scene_suffix}.json"

    @property
    def _scene_suffix(self) -> str:
        """The default test scene keeps the original file names."""
        return "" if self.scene == "default" else f"_{self.scene}"

    @property
    def scene_prefers_snow(self) -> bool:
        """Whether this profile's test scene should be the snowiest available."""
        return bool(self.scenes.get(self.scene, {}).get("prefer_snow", False))

    @property
    def scene_dates(self) -> tuple[str, str, float]:
        """(date_from, date_to, max_cloud_cover) for this profile's test scene."""
        if self.scene == "default" or self.scene not in self.scenes:
            return self.query_date_from, self.query_date_to, self.max_cloud_cover
        sc = self.scenes[self.scene]
        return (
            str(sc["date_from"]),
            str(sc["date_to"]),
            float(sc.get("max_cloud_cover", self.max_cloud_cover)),
        )

    @property
    def _suffix(self) -> str:
        """Profiles other than 'easy' get their own folders."""
        return "" if self.profile == "easy" else f"_{self.profile}"

    @property
    def queries_dir(self) -> Path:
        """Folder with test images and their ground truth."""
        return self.data_dir / "queries" / f"{self.area_name}{self._suffix}"

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
        return self.data_dir / "results" / f"{self.area_name}{self._suffix}"

    def with_profile(self, name: str) -> Config:
        """Copy of this config using a named test-image profile."""
        if name not in self.profiles:
            raise ValueError(f"Unknown profile '{name}'. Available: {', '.join(self.profiles)}")
        p = self.profiles[name]
        return replace(
            self,
            profile=name,
            rotation_deg=float(p.get("rotation_deg", 0)),
            scale_min=float(p.get("scale_min", 1)),
            scale_max=float(p.get("scale_max", 1)),
            blur_sigma=float(p.get("blur_sigma", 0)),
            noise_std=float(p.get("noise_std", 0)),
            scene=str(p.get("scene", "default")),
        )


def _spoof_settings(sp: dict) -> dict:
    """Spoofing-detection settings from the 'spoof' section, with defaults."""
    return {
        "spoof_radius_km": float(sp.get("radius_km", 5)),
        "spoof_threshold_m": float(sp.get("threshold_m", 100)),
        "gps_noise_m": float(sp.get("gps_noise_m", 5)),
        "spoof_offsets_m": tuple(
            float(o) for o in sp.get("offsets_m", [50, 100, 250, 500, 1000, 2000])
        ),
        "flight_steps": int(sp.get("flight_steps", 60)),
        "flight_step_m": float(sp.get("flight_step_m", 250)),
        "spoof_start_step": int(sp.get("spoof_start_step", 20)),
        "drift_m_per_step": float(sp.get("drift_m_per_step", 25)),
        "track_radius_km": float(sp.get("track_radius_km", 2)),
    }


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
    ref = raw.get("refine", {})
    nav = raw.get("navigate", {})
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
        profiles=raw.get("query_profiles", {"easy": {}}),
        scenes=raw.get("scenes", {}),
        season_radius_km=float(raw.get("navigate", {}).get("season_radius_km", 5)),
        **_spoof_settings(raw.get("spoof", {})),
        locate_radius_km=float(raw.get("locate", {}).get("radius_km", 5)),
        locate_fov_deg=float(raw.get("locate", {}).get("fov_deg", 70)),
        locate_date_from=str(raw.get("locate", {}).get("date_from", "2025-05-01")),
        locate_date_to=str(raw.get("locate", {}).get("date_to", "2025-09-30")),
        model_name=str(model.get("name", "vit_small_patch14_dinov2.lvd142m")),
        model_image_size=int(model.get("image_size", 224)),
        model_batch_size=int(model.get("batch_size", 32)),
        top_k=int(model.get("top_k", 5)),
        matcher=str(ref.get("matcher", "lightglue")),
        window_scale=float(ref.get("window_scale", 1.5)),
        upscale=int(ref.get("upscale", 2)),
        max_keypoints=int(ref.get("max_keypoints", 2048)),
        ransac_px=float(ref.get("ransac_px", 3.0)),
        min_inliers=int(ref.get("min_inliers", 15)),
        min_scale=float(ref.get("min_scale", 0.7)),
        max_scale=float(ref.get("max_scale", 1.4)),
        tta_rotations=tuple(float(a) for a in ref.get("tta_rotations", [0])),
        tta_scales=tuple(float(z) for z in ref.get("tta_zooms", [1])),
        fine_rotations=tuple(float(a) for a in ref.get("fine_rotations", [])),
        early_stop_inliers=int(nav.get("early_stop_inliers", 40)),
        nav_radii_km=tuple(float(r) for r in nav.get("radii_km", [2, 5, 10])),
        nav_window_scale=float(nav.get("window_scale", 2.0)),
        nav_map_radius_km=float(nav.get("map_radius_km", 5)),
    )
    if not 0 <= cfg.tile_overlap < 1:
        raise ValueError("tiles.overlap must be in [0, 1)")
    if cfg.tile_size_px < 32:
        raise ValueError("tiles.size_px must be at least 32")
    if cfg.size_km <= 0:
        raise ValueError("area.size_km must be positive")
    if cfg.top_k < 1:
        raise ValueError("model.top_k must be at least 1")
    if "easy" in cfg.profiles:
        cfg = cfg.with_profile("easy")
    return cfg
