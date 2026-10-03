# Kjentmann

**GPS-free visual positioning: find where an overhead image was taken by matching it against satellite imagery.**

GNSS signals can be jammed and spoofed. In eastern Finnmark, Norway, interference is a daily problem, and the Norwegian Defence Research Establishment (FFI) recommends backup solutions for positioning. A downward-looking camera cannot be jammed. Kjentmann recognises the terrain and returns a position without any satellite signal.

*A "kjentmann" is Norwegian for a local guide: someone who knows the terrain and finds the way without a map.*

> **Status: v0.4.** Search inside the uncertainty circle of an inertial position estimate, point matching with LightGlue, position in metres, and "unknown" instead of a guess when confidence is low.

## Result

200 test images with simulated camera conditions (heading error up to ±15°, altitude error −20% to +25%, lens blur, sensor noise), taken on a different date than the map. Each image gets a simulated inertial estimate that is off by up to the given radius.

| Uncertainty radius | Answered | Median error | Within 100 m | Wrong (> 500 m) | Time per image |
|---|---|---|---|---|---|
| 2 km | 100% | 3 m | 100% | 0% | 0.19 s |
| 5 km | 100% | 3 m | 100% | 0% | 0.21 s |
| 10 km | 100% | 3 m | 100% | 0% | 0.46 s |

*Area: 20 × 20 km around Askim, Norway. Map: Sentinel-2, 13 June 2025. Test images: Sentinel-2, 19 May 2025. GPU: NVIDIA RTX 3050.*

Read the [limitations](#limitations) before drawing conclusions: this is a controlled test, not a flight test.

## How it works

Without GPS, an aircraft still knows roughly where it is from inertial navigation (dead reckoning). That estimate drifts, so the true position lies somewhere inside an uncertainty circle. Kjentmann searches only inside that circle.

1. **Map (once per area, before the flight):** A cloud-free Sentinel-2 image is downloaded and cut into overlapping windows with known positions. It is stored on board; no internet is needed in flight.
2. **Candidates:** Map windows inside the uncertainty circle, nearest to the inertial estimate first.
3. **Point matching:** LightGlue matches keypoints (DISK) between the camera image and each window.
4. **Geometric check:** RANSAC keeps only matches that agree on one similarity transform (rotation, scale, shift), as expected from a camera looking straight down. The number of agreeing matches is the confidence.
5. **Position:** The transform places the image centre on the map. The search stops at the first confident window. Too few agreeing matches, or an implausible scale, gives "unknown".

The position can then correct the drifting inertial estimate, which shrinks the circle again.

## How we got here

The evaluation went through four steps. Each one exposed a weakness that shaped the next.

### 1. Global coarse search with DINOv2 (v0.2)

First approach: search the whole map. Each map tile gets a DINOv2 fingerprint in a FAISS index, and a test image retrieves the most similar tiles.

| Method | Right tile first | Right tile in top 5 | Chance (top 5) | Median error |
|---|---|---|---|---|
| Raw pixels (baseline) | 12% | 28% | 9% | 7.70 km |
| **DINOv2 ViT-S/14** | **46%** | **74%** | 9% | **1.82 km** |

### 2. Point matching on the top 5 (v0.3)

LightGlue and RANSAC on the five best tiles turned "somewhere in this tile" into metres: 88% answered, median error 1 m, no wrong answers.

That test was too easy. Sentinel-2 puts every pass on the same pixel grid, so a crop from another date lines up with the map pixel for pixel. With realistic camera conditions added, only **24%** of images were answered (34% after also searching over rotations). Every answer was still correct, so the problem had to be the coarse search.

### 3. Diagnosis: one factor at a time

`kjentmann diagnose` runs the coarse search on test images that differ in a single factor:

| Test images | Right tile in top 5 |
|---|---|
| Plain crop | 74% |
| **Blur + noise only** | **19%** |
| Rotation ±15° only | 70% |
| Scale 0.8 to 1.25 only | 70% |
| All combined | 12% |

The cause was blur and noise, not rotation or scale. DINOv2 fingerprints depend on fine texture, which blur and noise remove. The earlier fix had targeted the wrong problem.

### 4. Search inside the uncertainty circle (v0.4)

Real navigation systems already have an approximate position from inertial navigation. Using it removes the need for a global coarse search, and lets the reliable part, point matching with a geometric check, do the work. Result: 100% answered, as shown at the top.

## Limitations

- **Same sensor and season.** Map and test images are both Sentinel-2, 25 days apart in late spring. A real camera has different colours and optics, and the season may differ from the map. A winter test is next.
- **Simulated camera.** Heading, altitude, blur and noise are simulated. Camera tilt (not looking straight down) is not.
- **The circle always contains the truth.** The simulated inertial error is uniform inside the radius. A real inertial system can drift further than assumed.
- **Altitude.** Sentinel-2 has 10 m pixels, which suits images covering a few kilometres (aircraft altitude, or a skydiver at exit), not low drone images.

## Getting started

Requires Python 3.10 or newer. An NVIDIA GPU is strongly recommended; on CPU, matching is about 20 times slower.

```bash
git clone https://github.com/Jorgenfje/kjentmann.git
cd kjentmann
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# PyTorch and torchvision with GPU support (NVIDIA). Without a GPU, skip this line.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"

kjentmann all                          # map, tiles, interactive map
kjentmann v02                          # test images from another date, coarse search
kjentmann queries --profile realistic  # test images with camera conditions
kjentmann navigate --profile realistic # uncertainty-circle search (main result)
kjentmann diagnose                     # coarse search, one factor at a time
pytest                                 # tests (offline, synthetic terrain)
```

Check that the GPU is used: the first lines say `kjører på: cuda`.

Results are written to `data/results/<area>_<profile>/`, including interactive maps where each test image is coloured by error in metres. Area, dates, test profiles, radii and thresholds are set in `config.yaml`. Run `kjentmann --help` for all steps.

## Project layout

```
src/kjentmann/
  fetch.py      download Sentinel-2 scenes (only the pixels inside the area)
  tiles.py      cut the map into overlapping tiles with known positions
  queries.py    test images with ground truth; rotation, scale, blur, noise
  embed.py      fingerprints: DINOv2 and a raw-pixel baseline
  evaluate.py   FAISS search, scoring, exact chance baseline
  match.py      keypoint matching (LightGlue, SIFT) and RANSAC verification
  refine.py     coarse search + point matching on the top candidates
  diagnose.py   coarse search on single-factor test profiles
  navigate.py   search inside an uncertainty circle
  viz.py        interactive maps
tests/          offline tests with synthetic terrain
```

## Data

Sentinel-2 L2A from the Copernicus programme, via the open [Earth Search](https://earth-search.aws.element84.com/v1) STAC catalogue. No account or API key is needed. Base maps from [Kartverket](https://www.kartverket.no/).

## Roadmap

- [x] v0.1 Map and tiles
- [x] v0.2 Coarse search (DINOv2 + FAISS) with evaluation
- [x] v0.3 Precise position (LightGlue) and confidence score
- [x] v0.4 Search inside an uncertainty circle
- [ ] v0.4 Detect spoofed GPS by comparing it with the visual position
- [ ] v0.5 Norwegian winter: summer map against snow-covered test images
- [ ] v0.6 Online demo (Docker, Azure)
- [ ] v1.0 Demo with skydiving helmet footage, and launch

## Licence

MIT. Contains modified Copernicus Sentinel data, see the [terms](https://sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice).
