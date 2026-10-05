"""Turn images into fingerprints (vectors) that can be compared.

Two embedders share one interface:

* ``PixelEmbedder``: a naive baseline that compares shrunken pixels directly.
* ``DinoEmbedder``: DINOv2, a self-supervised vision transformer whose features
  hold up well under changes in light and season.

Both return L2-normalised float32 vectors, so the inner product is the cosine
similarity.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image


class Embedder(Protocol):
    """Anything that maps RGB images to fingerprint vectors."""

    name: str

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        """Embed a list of HxWx3 uint8 images into an (n, d) float32 array."""
        ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Scale each row to unit length."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class PixelEmbedder:
    """Baseline: shrink to a small image and compare the raw pixels."""

    name = "pixel"

    def __init__(self, size: int = 16) -> None:
        self.size = size

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        out = []
        for img in images:
            small = Image.fromarray(img).resize((self.size, self.size), Image.BILINEAR)
            v = np.asarray(small, dtype=np.float32).ravel()
            out.append(v - v.mean())  # ignore overall brightness
        return l2_normalize(np.stack(out))


class DinoEmbedder:
    """DINOv2 fingerprint: the class token joined with the mean patch token."""

    name = "dinov2"

    def __init__(
        self,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        image_size: int = 224,
        batch_size: int = 32,
        pretrained: bool = True,
        device: str | None = None,
    ) -> None:
        import timm
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, img_size=image_size
        )
        self.model.eval().to(self.device)
        print(f"DINOv2 running on: {self.device}")
        if self.device == "cpu":
            print("  Warning: no GPU found. See the README on PyTorch with CUDA.")
        cfg = self.model.pretrained_cfg
        self.mean = np.array(cfg.get("mean", (0.485, 0.456, 0.406)), dtype=np.float32)
        self.std = np.array(cfg.get("std", (0.229, 0.224, 0.225)), dtype=np.float32)
        self.image_size = image_size
        self.batch_size = batch_size
        self.prefix = getattr(self.model, "num_prefix_tokens", 1)

    def _prepare(self, img: np.ndarray) -> np.ndarray:
        resized = Image.fromarray(img).resize((self.image_size, self.image_size), Image.BICUBIC)
        x = np.asarray(resized, dtype=np.float32) / 255.0
        x = (x - self.mean) / self.std
        return np.moveaxis(x, -1, 0)

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch = np.stack([self._prepare(im) for im in images[i : i + self.batch_size]])
                tokens = self.model.forward_features(torch.from_numpy(batch).to(self.device))
                cls = tokens[:, 0]
                patches = tokens[:, self.prefix :].mean(dim=1)
                out.append(torch.cat([cls, patches], dim=1).float().cpu().numpy())
        return l2_normalize(np.concatenate(out))


def make_embedder(name: str, cfg, pretrained: bool = True) -> Embedder:
    """Create an embedder by name ('pixel' or 'dinov2')."""
    if name == "pixel":
        return PixelEmbedder()
    if name == "dinov2":
        return DinoEmbedder(
            cfg.model_name, cfg.model_image_size, cfg.model_batch_size, pretrained=pretrained
        )
    raise ValueError(f"Unknown embedder: {name}")
