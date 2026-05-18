from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from georicenet.gates import LearnedSpatialReliabilityGate, prior_reliability_weight
from georicenet.losses import total_variation_loss


class CachedFusionDataset(Dataset):
    def __init__(self, cache_dir: str | Path) -> None:
        self.paths = sorted(Path(cache_dir).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz cache files found in {cache_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        path = self.paths[index]
        item = np.load(path)
        return {
            "rgb_prior": torch.from_numpy(item["rgb_prior"].astype(np.float32)),
            "geometry_prior": torch.from_numpy(item["geometry_prior"].astype(np.float32)),
            "rgb_density": torch.from_numpy(item["rgb_density"].astype(np.float32)),
            "geo_density": torch.from_numpy(item["geo_density"].astype(np.float32)),
            "target_density": torch.from_numpy(item["target_density"].astype(np.float32)),
            "count": torch.tensor(float(item["count"]), dtype=torch.float32),
            "name": path.name,
        }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_one_epoch(gate: LearnedSpatialReliabilityGate, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace) -> float:
    gate.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        rgb_prior = batch["rgb_prior"].to(device)
        geometry_prior = batch["geometry_prior"].to(device)
        rgb_density = batch["rgb_density"].to(device)
        geo_density = batch["geo_density"].to(device)
        target_density = batch["target_density"].to(device)
        count = batch["count"].to(device)
        prior_weight = prior_reliability_weight(geometry_prior)
        fused, weight, _ = gate(rgb_prior, geometry_prior, rgb_density, geo_density, prior_weight)
        pred_count = fused.sum(dim=(1, 2, 3))
        count_loss = F.smooth_l1_loss(pred_count, count)
        density_loss = F.l1_loss(fused, target_density, reduction="sum") / fused.shape[0]
        prior_loss = F.mse_loss(weight, prior_weight)
        tv_loss = total_variation_loss(weight)
        loss = count_loss + args.density_weight * density_loss + args.prior_weight * prior_loss + args.tv_weight * tv_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(gate: LearnedSpatialReliabilityGate, loader: DataLoader, device: torch.device) -> dict[str, object]:
    gate.eval()
    rows = []
    errors = []
    for batch in tqdm(loader, desc="eval", leave=False):
        rgb_prior = batch["rgb_prior"].to(device)
        geometry_prior = batch["geometry_prior"].to(device)
        rgb_density = batch["rgb_density"].to(device)
        geo_density = batch["geo_density"].to(device)
        count = batch["count"].to(device)
        prior_weight = prior_reliability_weight(geometry_prior)
        fused, weight, _ = gate(rgb_prior, geometry_prior, rgb_density, geo_density, prior_weight)
        pred_count = fused.sum(dim=(1, 2, 3))
        for idx, name in enumerate(batch["name"]):
            error = float(pred_count[idx].detach().cpu() - count[idx].detach().cpu())
            errors.append(error)
            rows.append(
                {
                    "name": name,
                    "target_count": f"{float(count[idx].detach().cpu()):.6f}",
                    "pred_count": f"{float(pred_count[idx].detach().cpu()):.6f}",
                    "abs_error": f"{abs(error):.6f}",
                    "mean_weight": f"{float(weight[idx].mean().detach().cpu()):.6f}",
                }
            )
    values = np.asarray(errors, dtype=np.float64)
    return {"mae": float(np.mean(np.abs(values))), "rmse": float(np.sqrt(np.mean(values * values))), "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transferable RGB-Geo spatial reliability gate from cached densities.")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--val-cache", required=True)
    parser.add_argument("--output-dir", default="outputs/transfer_gate")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--density-weight", type=float, default=0.02)
    parser.add_argument("--prior-weight", type=float, default=0.15)
    parser.add_argument("--tv-weight", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_loader = DataLoader(CachedFusionDataset(args.train_cache), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(CachedFusionDataset(args.val_cache), batch_size=1, shuffle=False)
    gate = LearnedSpatialReliabilityGate().to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(gate, train_loader, optimizer, device, args)
        metrics = evaluate(gate, val_loader, device)
        row = {"epoch": epoch, "train_loss": f"{train_loss:.6f}", "val_mae": f"{metrics['mae']:.6f}", "val_rmse": f"{metrics['rmse']:.6f}"}
        history.append(row)
        print(row)
        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            torch.save({"model": gate.state_dict(), "args": vars(args), "epoch": epoch}, output_dir / "best_gate.pth")
            write_rows(output_dir / "val_predictions.csv", metrics["rows"])
    write_rows(output_dir / "history.csv", history)


if __name__ == "__main__":
    main()
