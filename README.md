# Kjentmann

**GPS-free visual positioning: find where an overhead image was taken by matching it against satellite imagery.**

GNSS can be jammed and spoofed, which is a daily problem in eastern Finnmark, Norway. A downward-looking camera cannot be jammed. Kjentmann recognises the terrain and returns a position without any satellite signal.

## Result

200 test images per season with simulated camera conditions (heading error ±15°, altitude error −20% to +25%, blur, noise), searched within a 5 km uncertainty circle against a June map:

| Test images | Snow | Answered | Median error | Wrong (> 500 m) | Time per image |
|---|---|---|---|---|---|
| Spring | 0% | 100% | 3 m | 0% | 0.26 s |
| Autumn | 0% | 99% | 4 m | 0% | 0.28 s |
| Winter | 74% | 91% | 8 m | 0% | 0.80 s |

No answer was wrong. When unsure, the system says "unknown" instead of guessing.

*Area: 20 × 20 km around Askim, Norway. Sentinel-2 imagery. GPU: RTX 3050.*

## How it works

1. **Map:** a cloud-free Sentinel-2 image, cut into windows with known positions, stored before the flight.
2. **Candidates:** windows inside the uncertainty circle of the inertial position estimate, nearest first.
3. **Matching:** LightGlue matches keypoints between the camera image and each window.
4. **Check:** RANSAC keeps only matches that agree on one rotation, scale and shift. Too few means "unknown".
5. **Position:** the agreed transform places the image centre on the map.

**Key finding:** a global search with DINOv2 found the right area for 74% of clean images, but only 19% once blur and noise were added (`kjentmann diagnose`). Using the inertial estimate instead removed the problem.

## Limitations

- Map and test images are both Sentinel-2. A real camera has different colours and optics.
- Camera tilt is not simulated, and the uncertainty circle always contains the true position.
- 10 m pixels suit images taken from a few kilometres up, not low drone images.

## Try it on your own photo

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,app]"

kjentmann locate photo.jpg --near "Gardermoen" --altitude 3000
streamlit run app.py                   # web page: upload a photo, see your position
```

Works anywhere in the world: the satellite map around the rough position is downloaded on first use. If the photo has GPS coordinates in its file, the error is shown.

MIT licence. Contains modified Copernicus Sentinel data. Base maps: Kartverket.
