from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from georicenet.datasets import GeoRiceDataset
from georicenet.losses import density_change_loss, structure_consistency_loss, surface_suppression_loss, total_variation_loss
from georicenet.model import GeoRiceNet


def count_metrics(errors: list[float]) -> tuple[float, float]:
    values = np.asarray(errors, dtype=np.float64)
    return float(np.mean(np.abs(values))), float(np.sqrt(np.mean(values * values)))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def set_batchnorm_eval(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def train_one_epoch(model: GeoRiceNet, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace) -> float:
    model.train()
    if args.freeze_bn_stats:
        set_batchnorm_eval(model)
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        rgb_prior = batch["rgb_prior"].to(device)
        geometry_prior = batch["geometry_prior"].to(device)
        target_density = batch["target_density"].to(device)
        target_count = batch["count"].to(device)
        output = model(image, rgb_prior, geometry_prior)
        density = output["density"]
        baseline_density = output["baseline_density"]
        scale = output["scale"]
        pred_count = density.sum(dim=(1, 2, 3))
        count_loss = F.smooth_l1_loss(pred_count, target_count)
        density_loss = F.l1_loss(density, target_density, reduction="sum") / image.shape[0]
        structure_loss = structure_consistency_loss(density, target_density, geometry_prior)
        flat_loss = surface_suppression_loss(density, geometry_prior[:, 1:2])
        change_loss = density_change_loss(density, baseline_density)
        tv_loss = total_variation_loss(scale.mean(dim=1, keepdim=True))
        loss = (
            count_loss
            + args.density_weight * density_loss
            + args.structure_weight * structure_loss
            + args.flat_weight * flat_loss
            + args.change_weight * change_loss
            + args.tv_weight * tv_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model: GeoRiceNet, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    rows = []
    errors = []
    for batch in tqdm(loader, desc="eval", leave=False):
        image = batch["image"].to(device)
        rgb_prior = batch["rgb_prior"].to(device)
        geometry_prior = batch["geometry_prior"].to(device)
        target_count = batch["count"].to(device)
        output = model(image, rgb_prior, geometry_prior)
        pred_count = output["density"].sum(dim=(1, 2, 3))
        for idx, image_path in enumerate(batch["image_path"]):
            error = float(pred_count[idx].detach().cpu() - target_count[idx].detach().cpu())
            errors.append(error)
            rows.append(
                {
                    "image_path": image_path,
                    "target_count": f"{float(target_count[idx].detach().cpu()):.6f}",
                    "pred_count": f"{float(pred_count[idx].detach().cpu()):.6f}",
                    "abs_error": f"{abs(error):.6f}",
                    "scale_mean": f"{float(output['scale'][idx].mean().detach().cpu()):.6f}",
                }
            )
    mae, rmse = count_metrics(errors)
    return {"mae": mae, "rmse": rmse, "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GeoRiceNet from manifest CSV files.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/georicenet")
    parser.add_argument("--ricenet-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--density-weight", type=float, default=0.02)
    parser.add_argument("--structure-weight", type=float, default=0.2)
    parser.add_argument("--flat-weight", type=float, default=5e-4)
    parser.add_argument("--change-weight", type=float, default=0.05)
    parser.add_argument("--tv-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-bn-stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_loader = DataLoader(
        GeoRiceDataset(args.train_manifest, crop_size=args.crop_size, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(GeoRiceDataset(args.val_manifest), batch_size=1, shuffle=False, num_workers=0)
    model = GeoRiceNet(freeze_vgg=True, freeze_backend=True, freeze_output=False).to(device)
    if args.ricenet_checkpoint:
        model.load_ricenet_checkpoint(args.ricenet_checkpoint, map_location=device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args)
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_mae": f"{metrics['mae']:.6f}", "val_rmse": f"{metrics['rmse']:.6f}"}
        history.append(row)
        print(row)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, output_dir / "best_model.pth")
            write_rows(output_dir / "val_predictions.csv", metrics["rows"])
    write_rows(output_dir / "history.csv", history)


if __name__ == "__main__":
    main()
