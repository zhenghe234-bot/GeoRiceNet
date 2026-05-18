from __future__ import annotations

import torch
from torch.nn import functional as F


def total_variation_loss(tensor: torch.Tensor) -> torch.Tensor:
    vertical = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :]).mean()
    horizontal = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1]).mean()
    return vertical + horizontal


def surface_suppression_loss(density: torch.Tensor, flat_mask: torch.Tensor) -> torch.Tensor:
    return (F.relu(density) * flat_mask).sum(dim=(1, 2, 3)).mean()


def density_change_loss(density: torch.Tensor, baseline_density: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(density, baseline_density.detach())


def count_and_density_loss(
    pred_density: torch.Tensor,
    target_density: torch.Tensor,
    target_count: torch.Tensor,
    density_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_count = pred_density.sum(dim=(1, 2, 3))
    count_loss = F.smooth_l1_loss(pred_count, target_count)
    density_loss = F.l1_loss(pred_density, target_density, reduction="sum") / pred_density.shape[0]
    loss = count_loss + density_weight * density_loss
    return loss, {"count": count_loss.detach(), "density": density_loss.detach()}


def sobel_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    channels = tensor.shape[1]
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=tensor.device,
        dtype=tensor.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=tensor.device,
        dtype=tensor.dtype,
    ).view(1, 1, 3, 3)
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    grad_x = F.conv2d(tensor, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(tensor, kernel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)


def structure_consistency_loss(
    pred_density: torch.Tensor,
    target_density: torch.Tensor,
    geometry_prior: torch.Tensor,
) -> torch.Tensor:
    pred_grad = torch.log1p(sobel_magnitude(F.relu(pred_density)))
    target_grad = torch.log1p(sobel_magnitude(F.relu(target_density.detach())))
    saliency = geometry_prior[:, 3:4]
    flat = geometry_prior[:, 1:2]
    support = F.max_pool2d((target_density.detach() > 0).float(), kernel_size=15, stride=1, padding=7)
    consistency = torch.sqrt(((pred_grad - target_grad) * (0.5 + saliency)) ** 2 + 1e-6).mean()
    false_structure = (pred_grad * flat * (1.0 - support)).mean()
    return consistency + 0.25 * false_structure
