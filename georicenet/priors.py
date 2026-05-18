from __future__ import annotations

import cv2
import numpy as np


def normalize01(array: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    array = array.astype(np.float32)
    low = np.percentile(array, low_percentile)
    high = np.percentile(array, high_percentile)
    return np.clip((array - low) / (high - low + 1e-6), 0.0, 1.0).astype(np.float32)


def rgb_prior(image_rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    rgb = image_rgb.astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)


def texture_map(image_rgb: np.ndarray, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = cv2.GaussianBlur(np.abs(lap) / 255.0, (0, 0), 3.0)
    if target_hw is not None:
        target_h, target_w = target_hw
        texture = cv2.resize(texture, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return np.clip(texture.astype(np.float32), 0.0, 1.0)


def glare_prior(image_rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    value = hsv[:, :, 2] / 255.0
    saturation = hsv[:, :, 1] / 255.0
    texture = texture_map(image_rgb)
    glare = value * (1.0 - saturation) * (1.0 - texture)
    glare = normalize01(glare, 2.0, 98.0)
    target_h, target_w = target_hw
    return cv2.resize(glare, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)


def depth_gradient_prior(depth: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    depth = normalize01(depth)
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = normalize01(np.sqrt(grad_x * grad_x + grad_y * grad_y))
    target_h, target_w = target_hw
    return cv2.resize(gradient, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)


def vegetation_mask(image_rgb: np.ndarray) -> np.ndarray:
    rgb = image_rgb.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    exg = 2.0 * g - r - b
    return (normalize01(exg, 5.0, 95.0) > 0.45).astype(np.float32)


def flat_water_prior(
    image_rgb: np.ndarray,
    depth_gradient: np.ndarray,
    glare: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_hw
    texture = texture_map(image_rgb, target_hw)
    veg = cv2.resize(vegetation_mask(image_rgb), (target_w, target_h), interpolation=cv2.INTER_AREA)
    flat = (1.0 - depth_gradient) * (1.0 - texture) * (1.0 - veg)
    flat = np.maximum(flat, 0.35 * glare)
    return np.clip(flat, 0.0, 1.0).astype(np.float32)


def geometry_saliency_prior(depth_gradient: np.ndarray, image_rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    texture = texture_map(image_rgb, target_hw)
    saliency = 0.6 * depth_gradient + 0.4 * texture
    return normalize01(saliency, 2.0, 98.0)


def build_geometry_priors(image_rgb: np.ndarray, depth: np.ndarray | None, target_hw: tuple[int, int]) -> np.ndarray:
    glare = glare_prior(image_rgb, target_hw)
    if depth is None:
        depth_gradient = np.zeros_like(glare, dtype=np.float32)
    else:
        depth_gradient = depth_gradient_prior(depth, target_hw)
    flat = flat_water_prior(image_rgb, depth_gradient, glare, target_hw)
    saliency = geometry_saliency_prior(depth_gradient, image_rgb, target_hw)
    return np.stack([glare, flat, depth_gradient, saliency], axis=0).astype(np.float32)
