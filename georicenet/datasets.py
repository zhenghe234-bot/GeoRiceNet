from __future__ import annotations

import csv
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .priors import build_geometry_priors, rgb_prior


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    with manifest_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["_base_dir"] = str(manifest_path.parent)
    return rows


def load_density(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float32)
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as file:
            return np.asarray(file["density"], dtype=np.float32)
    raise ValueError(f"Unsupported density format: {path}")


def resize_density_to_target(density: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    original_sum = float(np.maximum(density, 0.0).sum())
    resized = cv2.resize(density.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized = np.maximum(resized, 0.0)
    resized_sum = float(resized.sum())
    if resized_sum > 1e-8:
        resized *= original_sum / resized_sum
    return resized.astype(np.float32)


class GeoRiceDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size_multiple: int = 16,
        crop_size: int = 0,
        train: bool = False,
    ) -> None:
        self.rows = read_manifest(manifest_path)
        self.image_size_multiple = image_size_multiple
        self.crop_size = int(crop_size)
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        base_dir = Path(row["_base_dir"])
        image_path = resolve_path(row["image_path"], base_dir)
        density_path = resolve_path(row["density_path"], base_dir)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        multiple = self.image_size_multiple
        width = max(multiple, round(width / multiple) * multiple)
        height = max(multiple, round(height / multiple) * multiple)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        image_rgb = np.asarray(image)
        image_tensor = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        target_hw = (height // 2, width // 2)
        density = resize_density_to_target(load_density(density_path), target_hw)
        rgb = rgb_prior(image_rgb, target_hw)
        geometry = build_geometry_priors(image_rgb, depth=None, target_hw=target_hw)
        count = float(row.get("count", density.sum()))

        if self.train and self.crop_size > 0:
            image_tensor, rgb, geometry, density = self._random_crop(image_tensor, rgb, geometry, density)
            count = float(density.sum())

        return {
            "image": image_tensor,
            "rgb_prior": torch.from_numpy(rgb),
            "geometry_prior": torch.from_numpy(geometry),
            "target_density": torch.from_numpy(density[None, :, :]),
            "count": torch.tensor(count, dtype=torch.float32),
            "image_path": str(image_path),
        }

    def _random_crop(
        self,
        image: torch.Tensor,
        rgb: np.ndarray,
        geometry: np.ndarray,
        density: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        _, image_h, image_w = image.shape
        crop_h = min(self.crop_size, image_h)
        crop_w = min(self.crop_size, image_w)
        crop_h = max(16, (crop_h // 16) * 16)
        crop_w = max(16, (crop_w // 16) * 16)
        y0 = int(torch.randint(0, max(1, image_h - crop_h + 1), (1,)).item() // 16 * 16)
        x0 = int(torch.randint(0, max(1, image_w - crop_w + 1), (1,)).item() // 16 * 16)
        image = image[:, y0 : y0 + crop_h, x0 : x0 + crop_w].contiguous()
        yd0, xd0 = y0 // 2, x0 // 2
        rgb = np.ascontiguousarray(rgb[:, yd0 : yd0 + crop_h // 2, xd0 : xd0 + crop_w // 2])
        geometry = np.ascontiguousarray(geometry[:, yd0 : yd0 + crop_h // 2, xd0 : xd0 + crop_w // 2])
        density = np.ascontiguousarray(density[yd0 : yd0 + crop_h // 2, xd0 : xd0 + crop_w // 2])
        return image, rgb, geometry, density
