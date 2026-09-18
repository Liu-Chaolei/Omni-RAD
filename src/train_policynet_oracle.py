"""train_policynet_oracle.py — Stage 4: Oracle-T supervised TimestepPolicyNet training.

Trains TimestepPolicyNet offline using T_oracle_aug labels from Stage 2.
Does NOT require the full StableCodec forward pass — works on pre-extracted
features from oracle_labels.csv (or oracle_timestep_summary.csv).

Supports:
  - Regression (SmoothL1) and classification (CrossEntropy over T candidates)
  - Feature ablation: all / no_t_snr / only_t_snr / only_rate / custom subset
  - Train/val split with early stopping
  - Saves PolicyNet weights for integration into StableCodec_variable2_step_policy

Usage:
    python src/train_policynet_oracle.py \
        --data_csv results/stage2_oracle_sweep/oracle_labels.csv \
        --out_dir results/stage4_oracle_policynet_reg \
        --mode regression \
        --ablation all

    python src/train_policynet_oracle.py \
        --data_csv results/stage2_oracle_sweep/oracle_labels.csv \
        --out_dir results/stage4_oracle_policynet_cls \
        --mode classification \
        --ablation all

    # Ablation: without T_snr
    python src/train_policynet_oracle.py \
        --data_csv results/stage2_oracle_sweep/oracle_labels.csv \
        --out_dir results/stage4_oracle_policynet_no_tsnr \
        --ablation no_t_snr

    # Ablation: only T_snr
    python src/train_policynet_oracle.py \
        --data_csv results/stage2_oracle_sweep/oracle_labels.csv \
        --out_dir results/stage4_oracle_policynet_only_tsnr \
        --ablation only_t_snr
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required: pip install pandas")

from timestep_policy_net import TimestepPolicyNet, POLICY_FEATURE_DIM


# =========================================================================
# Feature column definitions (must match build_timestep_features order)
# =========================================================================

ALL_FEATURE_COLS = [
    "log_lambda",       # 0
    "actual_bpp",       # 1
    "T_snr",            # 2  (will be normalized by /999)
    "SNR_compress",     # 3
    "scales_mean",      # 4
    "scales_std",       # 5
    "scales_p90",       # 6
    "latent_energy",    # 7  (y_hat_energy)
    "latent_std",       # 8  (y_hat_std)
    "sample_energy",    # 9
    "latent_mean_abs",  # 10 (sample_std proxy)
    "res1_norm",        # 11
    "res1_energy",      # 12
]

ABLATION_CONFIGS = {
    "all": ALL_FEATURE_COLS,
    "no_t_snr": [c for c in ALL_FEATURE_COLS if c not in ("T_snr", "SNR_compress")],
    "only_t_snr": ["T_snr", "SNR_compress"],
    "only_rate": ["log_lambda", "actual_bpp"],
    "entropy_only": ["scales_mean", "scales_std", "scales_p90", "SNR_compress"],
    "entropy_latent": ["scales_mean", "scales_std", "scales_p90", "SNR_compress",
                       "latent_energy", "latent_std", "latent_mean_abs"],
}

TARGET_COL = "T_oracle_aug"


# =========================================================================
# Dataset
# =========================================================================

class OracleLabelDataset(Dataset):
    """Dataset from oracle_labels.csv with feature selection and normalization."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = TARGET_COL,
        normalize: bool = True,
        stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        available = [c for c in feature_cols if c in df.columns]
        if not available:
            raise ValueError(f"No feature columns found in CSV. Available: {list(df.columns)}")

        self.feature_cols = available
        self.target_col = target_col

        X = df[available].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)

        valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if normalize:
            if stats is None:
                self.mean = X.mean(axis=0)
                self.std = X.std(axis=0) + 1e-8
            else:
                self.mean = np.array([stats[c][0] for c in available], dtype=np.float32)
                self.std = np.array([stats[c][1] for c in available], dtype=np.float32)
            X = (X - self.mean) / self.std
        else:
            self.mean = np.zeros(len(available), dtype=np.float32)
            self.std = np.ones(len(available), dtype=np.float32)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_stats(self) -> Dict[str, Tuple[float, float]]:
        return {c: (float(self.mean[i]), float(self.std[i]))
                for i, c in enumerate(self.feature_cols)}


# =========================================================================
# Classification variant
# =========================================================================

class TimestepPolicyNetClassifier(nn.Module):
    """Classification variant: predicts logits over discrete T candidates."""

    def __init__(self, in_dim: int, hidden: int = 128, t_candidates: List[int] = None):
        super().__init__()
        if t_candidates is None:
            t_candidates = [800, 825, 850, 875, 900, 925, 950, 975, 999]
        self.register_buffer("t_candidates", torch.tensor(t_candidates, dtype=torch.float32))
        num_classes = len(t_candidates)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def predict_t(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.forward(features)
        weights = F.softmax(logits, dim=-1)
        return (weights * self.t_candidates.unsqueeze(0)).sum(dim=-1)


# =========================================================================
# Training loop
# =========================================================================

def train_regression(
    model: TimestepPolicyNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": []}

    for epoch in range(config["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Extract T_snr from features (index 2, normalized by /999)
            T_snr_batch = X_batch[:, 2] * 999.0

            T_pred = model(X_batch, T_snr_batch)
            loss = F.smooth_l1_loss(T_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            train_n += len(y_batch)

        scheduler.step()

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                T_snr_batch = X_batch[:, 2] * 999.0
                T_pred = model(X_batch, T_snr_batch)
                val_loss_sum += F.smooth_l1_loss(T_pred, y_batch, reduction="sum").item()
                val_mae_sum += (T_pred - y_batch).abs().sum().item()
                val_n += len(y_batch)

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        val_mae = val_mae_sum / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % config["log_every"] == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{config['epochs']} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_MAE={val_mae:.2f} lr={scheduler.get_last_lr()[0]:.2e}")

        if patience_counter >= config["patience"]:
            print(f"  Early stopping at epoch {epoch+1} (patience={config['patience']})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_loss": best_val_loss, "history": history, "epochs_trained": epoch + 1}


def train_classification(
    model: TimestepPolicyNetClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    t_candidates: List[int],
) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    t_arr = np.array(t_candidates)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_top1_acc": []}

    for epoch in range(config["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # Convert continuous T_oracle to class index (nearest candidate)
            target_idx = torch.argmin(
                (y_batch.unsqueeze(1) - model.t_candidates.unsqueeze(0)).abs(), dim=1
            )
            logits = model(X_batch)
            loss = F.cross_entropy(logits, target_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            train_n += len(y_batch)

        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                target_idx = torch.argmin(
                    (y_batch.unsqueeze(1) - model.t_candidates.unsqueeze(0)).abs(), dim=1
                )
                logits = model(X_batch)
                val_loss_sum += F.cross_entropy(logits, target_idx, reduction="sum").item()
                T_pred = model.predict_t(X_batch)
                val_mae_sum += (T_pred - y_batch).abs().sum().item()
                val_correct += (logits.argmax(dim=1) == target_idx).sum().item()
                val_n += len(y_batch)

        train_loss = train_loss_sum / train_n
        val_loss = val_loss_sum / val_n
        val_mae = val_mae_sum / val_n
        val_acc = val_correct / val_n
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["val_top1_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % config["log_every"] == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{config['epochs']} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_MAE={val_mae:.2f} top1_acc={val_acc:.3f}")

        if patience_counter >= config["patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {"best_val_loss": best_val_loss, "history": history, "epochs_trained": epoch + 1}


# =========================================================================
# Evaluation
# =========================================================================

def evaluate_model(model, val_loader, device, mode="regression"):
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            if mode == "regression":
                T_snr_batch = X_batch[:, 2] * 999.0
                T_pred = model(X_batch, T_snr_batch)
            else:
                T_pred = model.predict_t(X_batch)
            all_pred.append(T_pred.cpu())
            all_target.append(y_batch)

    pred = torch.cat(all_pred).numpy()
    target = torch.cat(all_target).numpy()

    mae = float(np.abs(pred - target).mean())
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))
    corr = float(np.corrcoef(pred, target)[0, 1]) if len(pred) > 2 else float("nan")

    # Top-1 accuracy (nearest candidate match)
    t_candidates = [800, 825, 850, 875, 900, 925, 950, 975, 999]
    t_arr = np.array(t_candidates)
    pred_cls = t_arr[np.argmin(np.abs(pred[:, None] - t_arr[None, :]), axis=1)]
    target_cls = t_arr[np.argmin(np.abs(target[:, None] - t_arr[None, :]), axis=1)]
    top1_acc = float((pred_cls == target_cls).mean())
    top2_mask = np.abs(pred[:, None] - t_arr[None, :]).argsort(axis=1)[:, :2]
    target_idx = np.argmin(np.abs(target[:, None] - t_arr[None, :]), axis=1)
    top2_acc = float(np.array([target_idx[i] in top2_mask[i] for i in range(len(target))]).mean())

    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "top1_accuracy": top1_acc,
        "top2_accuracy": top2_acc,
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
    }


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Oracle-T supervised TimestepPolicyNet training."
    )
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to oracle_labels.csv from Stage 2")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--mode", type=str, default="regression",
                        choices=["regression", "classification"],
                        help="Training mode: regression (SmoothL1) or classification (CE)")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=list(ABLATION_CONFIGS.keys()) + ["custom"],
                        help="Feature ablation config")
    parser.add_argument("--custom_features", type=str, nargs="+", default=None,
                        help="Custom feature columns (when --ablation custom)")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t_min", type=float, default=870.0)
    parser.add_argument("--t_max", type=float, default=999.0)
    parser.add_argument("--delta_max", type=float, default=30.0, help="max |delta_T| from T_snr")
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print(f"Loading data from: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    print(f"  Total rows: {len(df)}")

    # Resolve feature columns
    if args.ablation == "custom":
        feature_cols = args.custom_features or ALL_FEATURE_COLS
    else:
        feature_cols = ABLATION_CONFIGS[args.ablation]
    print(f"  Ablation: {args.ablation} ({len(feature_cols)} features)")
    print(f"  Features: {feature_cols}")

    # Build dataset
    dataset = OracleLabelDataset(df, feature_cols, target_col=TARGET_COL)
    print(f"  Valid samples: {len(dataset)}")

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Train: {train_size}, Val: {val_size}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    in_dim = len(dataset.feature_cols)

    # Build model
    if args.mode == "regression":
        model = TimestepPolicyNet(
            in_dim=in_dim, hidden=args.hidden,
            t_min=args.t_min, t_max=args.t_max,
            delta_max=args.delta_max,
        ).to(device)
    else:
        model = TimestepPolicyNetClassifier(
            in_dim=in_dim, hidden=args.hidden,
            t_candidates=t_candidates,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.mode}, params={n_params}")

    config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "log_every": args.log_every,
    }

    # Train
    print(f"\nTraining ({args.mode})...")
    start = time.time()
    if args.mode == "regression":
        result = train_regression(model, train_loader, val_loader, config, device)
    else:
        result = train_classification(model, train_loader, val_loader, config, device, t_candidates)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s, {result['epochs_trained']} epochs")

    # Evaluate
    print("\nEvaluation on validation set:")
    eval_metrics = evaluate_model(model, val_loader, device, mode=args.mode)
    for k, v in eval_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save
    ckpt_path = os.path.join(args.out_dir, "policynet_oracle.pth")
    save_dict = {
        "state_dict": model.state_dict(),
        "mode": args.mode,
        "ablation": args.ablation,
        "feature_cols": dataset.feature_cols,
        "in_dim": in_dim,
        "hidden": args.hidden,
        "t_min": args.t_min,
        "t_max": args.t_max,
        "norm_stats": dataset.get_stats(),
        "eval_metrics": eval_metrics,
        "config": config,
    }
    if args.mode == "classification":
        save_dict["t_candidates"] = t_candidates
    torch.save(save_dict, ckpt_path)
    print(f"\n  Checkpoint saved to: {ckpt_path}")

    # Save report
    report = {
        "data_csv": args.data_csv,
        "n_samples": len(dataset),
        "n_train": train_size,
        "n_val": val_size,
        "mode": args.mode,
        "ablation": args.ablation,
        "feature_cols": dataset.feature_cols,
        "in_dim": in_dim,
        "hidden": args.hidden,
        "n_params": n_params,
        "epochs_trained": result["epochs_trained"],
        "best_val_loss": result["best_val_loss"],
        "eval_metrics": eval_metrics,
        "elapsed_seconds": elapsed,
    }
    report_path = os.path.join(args.out_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved to: {report_path}")

    print("\nStage 4 training complete.")


if __name__ == "__main__":
    main()
