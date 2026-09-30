"""Keypoint matching and geometric verification.

Given a test image and a map window, find points that appear in both and
estimate the transform between them. If enough matches agree on one
transform, we know exactly where the test image lies in the window.

Two matchers share one interface:

* ``SiftMatcher``: classic SIFT with a ratio test (OpenCV, no download).
* ``LightGlueMatcher``: DISK keypoints matched by LightGlue (Kornia), a learned
  matcher that holds up better under changes in light and season.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Verification:
    """Result of matching one test image against one map window."""

    inliers: int
    matches: int
    transform: np.ndarray | None  # 2x3: test image pixel -> window pixel

    def map_point(self, x: float, y: float) -> tuple[float, float] | None:
        """Where a test image pixel lands in the window, or None if unverified."""
        if self.transform is None:
            return None
        m = self.transform
        return float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(
            m[1, 0] * x + m[1, 1] * y + m[1, 2]
        )


class Matcher(Protocol):
    """Anything that finds corresponding points between two RGB images."""

    name: str

    def match(
        self, a: np.ndarray, b: np.ndarray, b_key=None, a_key=None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Matched points (N, 2) in a and b, in pixels (x, y).

        ``a_key`` and ``b_key`` let the matcher reuse features it already computed.
        """
        ...


class _QuerySlot:
    """Remembers features of the last query image, reused for all its candidates."""

    def __init__(self) -> None:
        self.key = None
        self.value = None

    def get(self, key, compute):
        if key is None:
            return compute()
        if key != self.key:
            self.key, self.value = key, compute()
        return self.value


def upscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Enlarge an image; 10 m pixels give detectors more to work with at 2x."""
    if factor == 1:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)


class SiftMatcher:
    """SIFT keypoints, nearest-neighbour matching and Lowe's ratio test."""

    name = "sift"

    def __init__(self, ratio: float = 0.8, max_keypoints: int = 4000, scale: int = 2) -> None:
        self.ratio = ratio
        self.scale = scale
        self.sift = cv2.SIFT_create(nfeatures=max_keypoints)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self._cache: dict[int, tuple] = {}
        self._query = _QuerySlot()

    def _features(self, img: np.ndarray, key: int | None):
        if key is not None and key in self._cache:
            return self._cache[key]
        gray = cv2.cvtColor(upscale(img, self.scale), cv2.COLOR_RGB2GRAY)
        kps, desc = self.sift.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2) / self.scale
        out = (pts, desc)
        if key is not None:
            self._cache[key] = out
        return out

    def match(
        self, a: np.ndarray, b: np.ndarray, b_key: int | None = None, a_key=None
    ) -> tuple[np.ndarray, np.ndarray]:
        pa, da = self._query.get(a_key, lambda: self._features(a, None))
        pb, db = self._features(b, b_key)
        if da is None or db is None or len(da) < 2 or len(db) < 2:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        pairs = self.bf.knnMatch(da, db, k=2)
        good = [
            m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance
        ]
        ia = np.array([m.queryIdx for m in good], dtype=int)
        ib = np.array([m.trainIdx for m in good], dtype=int)
        return pa[ia].reshape(-1, 2), pb[ib].reshape(-1, 2)


class LightGlueMatcher:
    """DISK keypoints + LightGlue matching (Kornia). Weights download on first use."""

    name = "lightglue"

    def __init__(
        self, max_keypoints: int = 2048, scale: int = 2, device: str | None = None
    ) -> None:
        import kornia.feature as KF
        import torch

        self.torch = torch
        self.KF = KF
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
        self.lg = KF.LightGlueMatcher("disk").to(self.device).eval()
        print(f"LightGlue kjører på: {self.device}")
        self.max_keypoints = max_keypoints
        self.scale = scale
        self._cache: dict[int, tuple] = {}
        self._query = _QuerySlot()

    def _features(self, img: np.ndarray, key: int | None):
        if key is not None and key in self._cache:
            return self._cache[key]
        torch = self.torch
        big = upscale(img, self.scale)
        t = torch.from_numpy(big).permute(2, 0, 1).float().div(255)[None].to(self.device)
        with torch.inference_mode():
            f = self.disk(t, n=self.max_keypoints, pad_if_not_divisible=True)[0]
        kps, desc = f.keypoints, f.descriptors
        lafs = self.KF.laf_from_center_scale_ori(
            kps[None], torch.ones(1, len(kps), 1, 1, device=self.device)
        )
        out = (kps, desc, lafs, big.shape[:2])
        if key is not None:
            self._cache[key] = out
        return out

    def match(
        self, a: np.ndarray, b: np.ndarray, b_key: int | None = None, a_key=None
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        ka, da, la, hwa = self._query.get(a_key, lambda: self._features(a, None))
        kb, db, lb, hwb = self._features(b, b_key)
        with torch.inference_mode():
            _, idx = self.lg(da, db, la, lb, hw1=hwa, hw2=hwb)
        idx = idx.cpu().numpy()
        pa = ka.cpu().numpy()[idx[:, 0]] / self.scale
        pb = kb.cpu().numpy()[idx[:, 1]] / self.scale
        return pa.astype(np.float32), pb.astype(np.float32)


def verify(pa: np.ndarray, pb: np.ndarray, ransac_px: float = 3.0) -> Verification:
    """Fit a similarity transform (rotation, scale, shift) with RANSAC.

    A camera looking straight down sees the ground up to rotation, scale and
    shift, so a similarity transform fits. Matches that disagree with it are
    rejected; the number that agree is our confidence.
    """
    n = len(pa)
    if n < 4:
        return Verification(0, n, None)
    m, mask = cv2.estimateAffinePartial2D(
        pa, pb, method=cv2.RANSAC, ransacReprojThreshold=ransac_px, maxIters=2000, confidence=0.999
    )
    if m is None or mask is None:
        return Verification(0, n, None)
    return Verification(int(mask.sum()), n, m)


def make_matcher(name: str, max_keypoints: int = 2048, scale: int = 2) -> Matcher:
    """Create a matcher by name ('sift' or 'lightglue')."""
    if name == "sift":
        return SiftMatcher(max_keypoints=max_keypoints * 2, scale=scale)
    if name == "lightglue":
        return LightGlueMatcher(max_keypoints=max_keypoints, scale=scale)
    raise ValueError(f"Ukjent matcher: {name}")
