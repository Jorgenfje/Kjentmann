"""Search every test image against the tile index and score the result.

A search is a hit when a returned tile actually contains the true centre of
the test image. With 50 % overlap each point lies in one to four tiles, so we
also report what random guessing would score, to show the gain is real.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from math import comb
from pathlib import Path

import faiss
import numpy as np
import rasterio
from PIL import Image

from kjentmann.config import Config
from kjentmann.embed import Embedder
from kjentmann.geo import apply_transform
from kjentmann.queries import Query, read_queries


@dataclass(frozen=True)
class TileRef:
    """The parts of a tile needed for scoring."""

    tile_id: str
    px_x: int
    px_y: int
    center_x: float
    center_y: float
    center_lon: float
    center_lat: float


def read_tiles(path: Path) -> list[TileRef]:
    """Read tiles.csv in index order."""
    with path.open(encoding="utf-8") as f:
        return [
            TileRef(
                r["tile_id"],
                int(r["px_x"]),
                int(r["px_y"]),
                float(r["center_x"]),
                float(r["center_y"]),
                float(r["center_lon"]),
                float(r["center_lat"]),
            )
            for r in csv.DictReader(f)
        ]


def load_images(folder: Path, ids: list[str]) -> list[np.ndarray]:
    """Load PNGs by id, in order."""
    return [np.asarray(Image.open(folder / f"{i}.png").convert("RGB")) for i in ids]


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Exact inner-product index (cosine similarity on normalised vectors)."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def containing_tiles(q: Query, tiles: list[TileRef], transform, size: int) -> set[int]:
    """Indexes of tiles whose footprint contains the query centre."""
    col, row = apply_transform(~transform, q.center_x, q.center_y)
    return {
        i
        for i, t in enumerate(tiles)
        if t.px_x <= col < t.px_x + size and t.px_y <= row < t.px_y + size
    }


def random_recall(n_tiles: int, n_correct: int, k: int) -> float:
    """Chance that k random distinct tiles include at least one correct tile."""
    if n_correct == 0:
        return 0.0
    k = min(k, n_tiles)
    return 1.0 - comb(n_tiles - n_correct, k) / comb(n_tiles, k)


def evaluate_embedder(
    cfg: Config, embedder: Embedder, tiles: list[TileRef], queries: list[Query], transform
) -> tuple[dict, list[dict]]:
    """Index the tiles, search all queries and compute the scores."""
    k = cfg.top_k
    size = cfg.tile_size_px

    t0 = time.perf_counter()
    tile_vecs = embedder.embed(load_images(cfg.tiles_dir, [t.tile_id for t in tiles]))
    index_seconds = time.perf_counter() - t0
    index = build_index(tile_vecs)
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(cfg.index_dir / f"{embedder.name}.faiss"))

    q_images = load_images(cfg.queries_dir, [q.query_id for q in queries])
    t0 = time.perf_counter()
    q_vecs = embedder.embed(q_images)
    scores, idx = index.search(q_vecs, k)
    ms_per_query = (time.perf_counter() - t0) / max(len(queries), 1) * 1000

    rows = []
    hits1 = hitsk = 0
    errors, rand1, randk = [], [], []
    for qi, q in enumerate(queries):
        correct = containing_tiles(q, tiles, transform, size)
        ranked = [int(i) for i in idx[qi]]
        rank = next((r + 1 for r, i in enumerate(ranked) if i in correct), None)
        best = tiles[ranked[0]]
        err_m = float(np.hypot(best.center_x - q.center_x, best.center_y - q.center_y))
        hits1 += rank == 1
        hitsk += rank is not None
        errors.append(err_m)
        rand1.append(random_recall(len(tiles), len(correct), 1))
        randk.append(random_recall(len(tiles), len(correct), k))
        rows.append(
            {
                "query_id": q.query_id,
                "true_lat": q.center_lat,
                "true_lon": q.center_lon,
                "top1_tile": best.tile_id,
                "top1_lat": best.center_lat,
                "top1_lon": best.center_lon,
                "top1_score": float(scores[qi, 0]),
                "rank_of_correct": rank or "",
                "top1_error_m": round(err_m, 1),
            }
        )

    n = max(len(queries), 1)
    summary = {
        "embedder": embedder.name,
        "queries": len(queries),
        "tiles": len(tiles),
        "recall@1": hits1 / n,
        f"recall@{k}": hitsk / n,
        "random_recall@1": float(np.mean(rand1)),
        f"random_recall@{k}": float(np.mean(randk)),
        "median_top1_error_m": float(np.median(errors)),
        "index_seconds": index_seconds,
        "ms_per_query": ms_per_query,
    }
    return summary, rows


def evaluate(cfg: Config, embedders: list[Embedder]) -> list[dict]:
    """Run the evaluation for each embedder and write the reports."""
    for path, step in (
        (cfg.tiles_csv, "tiles"),
        (cfg.queries_csv, "queries"),
    ):
        if not path.exists():
            raise SystemExit(f"Fant ikke {path}. Kjør 'kjentmann {step}' først.")
    tiles = read_tiles(cfg.tiles_csv)
    queries = read_queries(cfg.queries_csv)
    with rasterio.open(cfg.map_path) as src:
        transform = src.transform

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for emb in embedders:
        print(f"Evaluerer {emb.name} ...")
        summary, rows = evaluate_embedder(cfg, emb, tiles, queries, transform)
        summaries.append(summary)
        with (cfg.results_dir / f"{emb.name}_queries.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        if emb.name != "pixel" or len(embedders) == 1:
            from kjentmann.viz import build_results_map

            build_results_map(cfg, emb.name, rows)

    (cfg.results_dir / "results.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    table = results_table(summaries, cfg.top_k)
    (cfg.results_dir / "results.md").write_text(table + "\n", encoding="utf-8")
    print()
    print(table)
    return summaries


def results_table(summaries: list[dict], k: int) -> str:
    """Markdown table for the README and LinkedIn."""
    lines = [
        f"| Method | Hit @1 | Hit @{k} | Chance @{k} | Median error @1 | ms per image |",
        "|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['embedder']} | {s['recall@1']:.0%} | {s[f'recall@{k}']:.0%} "
            f"| {s[f'random_recall@{k}']:.0%} | {s['median_top1_error_m'] / 1000:.2f} km "
            f"| {s['ms_per_query']:.0f} |"
        )
    return "\n".join(lines)
