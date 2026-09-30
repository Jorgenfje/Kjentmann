# Kjentmann

**GPS-free visual positioning: find where an overhead image was taken by matching it against satellite imagery.**

GNSS signals can be jammed and spoofed. In eastern Finnmark, Norway, interference is a daily problem, and the Norwegian Defence Research Establishment (FFI) recommends backup solutions for positioning. A downward-looking camera cannot be jammed. Kjentmann recognises the terrain and returns a position without any satellite signal.

*A "kjentmann" is Norwegian for a local guide: someone who knows the terrain and finds the way without a map.*

> **Status: v0.3.** Coarse search with DINOv2 and FAISS, then point matching with LightGlue for a position in metres and an "unknown" answer when confidence is low.

## How it works

1. **Map (once per area):** A cloud-free Sentinel-2 image is downloaded and cut into overlapping tiles with known positions.
2. **Fingerprints:** Each tile is turned into a vector with DINOv2 and stored in a FAISS index.
3. **Coarse search:** A new image gets its own fingerprint, and the 5 most similar tiles are retrieved.
4. **Fine matching:** For each of the 5 candidates, LightGlue matches keypoints (DISK) between the image and a map window around the tile. RANSAC keeps only matches that agree on one similarity transform (rotation, scale, shift). The candidate with most agreeing matches wins, and the transform places the image centre on the map.
5. **Confidence:** The number of agreeing matches is the confidence. Below a threshold, or with an implausible scale, the answer is "unknown" instead of a guess.

## Results

### v0.2: coarse search

Test set: 200 crops from a Sentinel-2 image taken on a **different date** than the map, so light, shadows and vegetation differ. Crops are placed at random, independent of the tile grid. A search counts as a hit when a returned tile contains the true centre of the test image.

| Method | Hit @1 | Hit @5 | Chance @5 | Median error @1 | ms per image |
|---|---|---|---|---|---|
| Raw pixels (baseline) | 12% | 28% | 9% | 7.70 km | <1 |
| **DINOv2 ViT-S/14** | **46%** | **74%** | 9% | **1.82 km** | 58 |

*Area: 20 × 20 km around Askim, Norway. 225 tiles of 2.56 × 2.56 km. 200 test images. GPU: NVIDIA RTX 3050. "Chance" is the exact hit rate of guessing 5 random tiles.*

DINOv2 finds the right tile first almost four times as often as comparing raw pixels, and has the right answer among its top 5 for three out of four images, against 9% by chance. The misses tend to point at a few look-alike tiles; v0.3 adds geometric verification with LightGlue to reject those, and re-ranks the top 5.

### v0.3: position in metres

Same 200 test images, now matched point by point against the 5 candidates.

**Easy profile** (plain crops: same sensor, north-up, same scale as the map):

| Result | Value |
|---|---|
| Answered (≥ 15 agreeing matches) | 88% |
| Median error when answered | 1 m |
| Within 100 m, of all images | 88% |
| Wrong answers (> 500 m) | 0% |

When the system answers, it is right; the remaining 12% get "unknown" instead of a guess. No threshold between 8 and 50 matches produced a single wrong answer.

**Why this is an upper bound, not a field result.** Sentinel-2 puts every pass on the same pixel grid, so a crop from another date lines up with the map pixel for pixel, at the same scale and heading. A camera under an aircraft does not. The **realistic profile** adds a heading error of up to ±15°, an altitude (scale) error of −20% to +25%, lens blur and sensor noise:

```bash
kjentmann queries --profile realistic
kjentmann refine  --profile realistic
```

*Realistic-profile results for Askim follow.*

The results map shows every test image where it was really taken: green if found first, orange if among the top 5, red if missed, with a line to the top guess.

Assumption: images are north-up. In a real system the heading comes from a compass.

## Getting started

Requires Python 3.10 or newer. An NVIDIA GPU makes DINOv2 faster but is not required.

```bash
git clone https://github.com/Jorgenfje/kjentmann.git
cd kjentmann
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# PyTorch with GPU support (NVIDIA). Without a GPU, skip this line.
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev]"

kjentmann all      # v0.1: map, tiles and an interactive map
kjentmann v02      # v0.2: test images, search and evaluation
kjentmann v03      # v0.3: point matching, position in metres
pytest             # tests (offline)
```

Output:

- `data/askim_map.html`: satellite image and tile grid over Kartverket's base map
- `data/results/askim/results.md`: accuracy table
- `data/results/askim/dinov2_map.html`: every test image on the map
- `data/results/askim/refine_lightglue_map.html`: estimated positions, coloured by error in metres

Area, dates, number of test images and model are set in `config.yaml`. Run `kjentmann --help` for all steps.

## Project layout

```
src/kjentmann/
  fetch.py      download Sentinel-2 scenes (only the pixels inside the area)
  tiles.py      cut the map into overlapping tiles with known positions
  queries.py    cut test images with ground truth from another date
  embed.py      fingerprints: DINOv2 and a raw-pixel baseline
  evaluate.py   FAISS search, scoring, exact chance baseline
  match.py      keypoint matching (LightGlue, SIFT) and RANSAC verification
  refine.py     re-ranking, position in metres, confidence threshold
  viz.py        interactive maps
tests/          offline tests with synthetic terrain
```

## Data

Sentinel-2 L2A from the Copernicus programme, via the open [Earth Search](https://earth-search.aws.element84.com/v1) STAC catalogue. No account or API key is needed. Base maps from [Kartverket](https://www.kartverket.no/).

Sentinel-2 has 10 m pixels. That suits images taken from aircraft altitude (a few kilometres across), but not low drone images, which cover too few pixels.

## Roadmap

- [x] v0.1 Map and tiles
- [x] v0.2 Coarse search (DINOv2 + FAISS) with evaluation
- [x] v0.3 Precise position (LightGlue) and confidence score
- [ ] v0.4 Realistic navigation: search within an uncertainty radius, detect spoofed GPS
- [ ] v0.5 Norwegian winter: accuracy by season
- [ ] v0.6 Online demo (Docker, Azure)
- [ ] v1.0 Demo with skydiving helmet footage, and launch

## Licence

MIT. Contains modified Copernicus Sentinel data, see the [terms](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
