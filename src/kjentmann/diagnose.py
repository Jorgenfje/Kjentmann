"""Which image change hurts the coarse search? Test one factor at a time.

Runs the coarse search (no extra rotations or zooms) on test images from
several profiles that differ in a single factor, and puts the results side by
side. The fine-matching step is skipped: it is reliable once the right tile is
among the candidates, so the coarse search is what we need to understand.
"""

from __future__ import annotations

from kjentmann.config import Config
from kjentmann.embed import Embedder
from kjentmann.evaluate import evaluate
from kjentmann.queries import build_queries

PROFILES = ("easy", "blur", "rotation", "scale", "realistic")
WHAT = {
    "easy": "nothing (plain crop)",
    "blur": "blur + noise only",
    "rotation": "rotation ±15° only",
    "scale": "scale 0.8–1.25 only",
    "realistic": "all of the above",
}


def diagnose(base: Config, embedder: Embedder) -> list[dict]:
    """Coarse-search scores for each single-factor profile, as one table."""
    k = base.top_k
    rows = []
    for name in PROFILES:
        if name not in base.profiles:
            continue
        cfg = base.with_profile(name)
        if not cfg.queries_csv.exists():
            build_queries(cfg)
        print(f"\n=== {name}: {WHAT[name]} ===")
        s = evaluate(cfg, [embedder])[0]
        rows.append({"profile": name, **s})

    lines = [
        f"| Profile | What changes | Right tile first | Right tile in top {k} | Chance @{k} |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['profile']} | {WHAT[r['profile']]} | {r['recall@1']:.0%} "
            f"| {r[f'recall@{k}']:.0%} | {r[f'random_recall@{k}']:.0%} |"
        )
    table = "\n".join(lines)
    out = base.data_dir / "results" / f"{base.area_name}_diagnose.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"Coarse search: {embedder.name}\n\n{table}\n", encoding="utf-8")
    print(f"\n{table}\n\nLagret: {out}")
    return rows
