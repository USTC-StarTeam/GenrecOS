#!/usr/bin/env python3

import argparse
import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    RESULTS_DIR,
    cache_path,
    ensure_dir,
    evaluate_fixed_alpha,
    find_best_fixed_alpha,
    move_split_to_device,
    save_json,
    set_global_seed,
    wait_for_cache_ready,
)


class ContextAdaptiveFusion(nn.Module):
    def __init__(
        self,
        context_dim: int,
        stats_dim: int,
        stats_mean: torch.Tensor,
        stats_std: torch.Tensor,
        base_alpha: float,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.hidden_proj = nn.Sequential(
            nn.Linear(context_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )
        self.stats_proj = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self.coeff_head = nn.Sequential(
            nn.Linear(192 + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 8),
        )
        self.base_alpha = base_alpha
        self.residual_scale = residual_scale
        self.register_buffer("stats_mean", stats_mean)
        self.register_buffer("stats_std", stats_std.clamp_min(1e-6))

    def forward(
        self,
        context_hidden: torch.Tensor,
        stats: torch.Tensor,
        sas_scores: torch.Tensor,
        llm_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        norm_stats = (stats - self.stats_mean) / self.stats_std
        context_vec = self.hidden_proj(context_hidden.float())
        stats_vec = self.stats_proj(norm_stats.float())
        coeffs = torch.tanh(self.coeff_head(torch.cat([context_vec, stats_vec], dim=1)))
        weights = coeffs[:, :7]
        bias = 0.05 * coeffs[:, 7:8]

        base_scores = self.base_alpha * sas_scores + (1.0 - self.base_alpha) * llm_scores
        diff = sas_scores - llm_scores
        feature_stack = torch.stack(
            [
                sas_scores,
                llm_scores,
                diff,
                diff.abs(),
                sas_scores * llm_scores,
                sas_scores.square(),
                llm_scores.square(),
            ],
            dim=-1,
        )
        residual = (feature_stack * weights.unsqueeze(1)).sum(dim=-1) + bias
        fused_scores = base_scores + self.residual_scale * residual
        return fused_scores, coeffs


def compute_ndcg_at_k(topk: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    matches = topk[:, :k] == targets.unsqueeze(1)
    hit_indices = matches.float().argmax(dim=1)
    hits = matches.any(dim=1)
    denom = torch.log2(hit_indices.float() + 2.0)
    ndcg = torch.where(hits, 1.0 / denom, torch.zeros_like(denom))
    return ndcg.mean().item()


def compute_metrics_from_scores(scores: torch.Tensor, targets: torch.Tensor, ks: List[int]) -> Dict[str, float]:
    topk = scores.topk(max(ks), dim=1).indices
    metrics = {}
    for k in ks:
        hits = (topk[:, :k] == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"HR@{k}"] = hits
    metrics["NDCG@10"] = compute_ndcg_at_k(topk, targets, 10)
    return metrics


def evaluate_model(
    model: ContextAdaptiveFusion,
    split: Dict[str, torch.Tensor],
    batch_size: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_sums = None
    coeff_sum = None
    total_rows = 0
    with torch.inference_mode():
        for start in range(0, split["targets"].size(0), batch_size):
            end = min(start + batch_size, split["targets"].size(0))
            scores, coeffs = model(
                split["context_hidden"][start:end],
                split["stats"][start:end],
                split["sas_scores"][start:end],
                split["llm_scores"][start:end],
            )
            batch_metrics = compute_metrics_from_scores(scores.float(), split["targets"][start:end], [1, 5, 10, 20])
            batch_rows = end - start
            if metric_sums is None:
                metric_sums = {k: 0.0 for k in batch_metrics}
            for key, value in batch_metrics.items():
                metric_sums[key] += value * batch_rows
            coeff_batch = coeffs.float().sum(dim=0)
            coeff_sum = coeff_batch if coeff_sum is None else coeff_sum + coeff_batch
            total_rows += batch_rows

    metrics = {key: value / total_rows for key, value in metric_sums.items()}
    coeff_mean = (coeff_sum / total_rows).tolist()
    coeff_stats = {f"coeff_{idx}": coeff_mean[idx] for idx in range(len(coeff_mean))}
    return metrics, coeff_stats


def train_model(
    model: ContextAdaptiveFusion,
    train_split: Dict[str, torch.Tensor],
    val_split: Dict[str, torch.Tensor],
    epochs: int,
    train_batch_size: int,
    eval_batch_size: int,
    lr: float,
    weight_decay: float,
    label_smoothing: float,
    patience: int,
) -> Tuple[ContextAdaptiveFusion, List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    history = []
    best_state = None
    best_val_hr1 = -1.0
    best_val_hr10 = -1.0
    patience_counter = 0
    num_train = train_split["targets"].size(0)
    device = train_split["targets"].device

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(num_train, device=device)
        train_loss_sum = 0.0
        train_hr1_sum = 0.0
        num_batches = 0

        for start in range(0, num_train, train_batch_size):
            end = min(start + train_batch_size, num_train)
            idx = perm[start:end]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores, coeffs = model(
                    train_split["context_hidden"][idx],
                    train_split["stats"][idx],
                    train_split["sas_scores"][idx],
                    train_split["llm_scores"][idx],
                )
                ce_loss = F.cross_entropy(
                    scores.float(),
                    train_split["targets"][idx],
                    label_smoothing=label_smoothing,
                )
                reg_loss = coeffs.square().mean()
                loss = ce_loss + 0.002 * reg_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.inference_mode():
                top1 = scores.argmax(dim=1)
                hr1 = (top1 == train_split["targets"][idx]).float().mean().item()

            train_loss_sum += loss.item()
            train_hr1_sum += hr1
            num_batches += 1

        train_loss = train_loss_sum / max(num_batches, 1)
        train_hr1 = train_hr1_sum / max(num_batches, 1)
        val_metrics, coeff_stats = evaluate_model(model, val_split, eval_batch_size)
        scheduler.step(val_metrics["HR@1"])

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_hr1": train_hr1,
            "val_hr1": val_metrics["HR@1"],
            "val_hr10": val_metrics["HR@10"],
            "val_ndcg10": val_metrics["NDCG@10"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        epoch_record.update(coeff_stats)
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} "
            f"train_hr1={train_hr1:.4f} val_hr1={val_metrics['HR@1']:.4f} "
            f"val_hr10={val_metrics['HR@10']:.4f}"
        )

        improved = val_metrics["HR@1"] > best_val_hr1
        same_hr1 = math.isclose(val_metrics["HR@1"], best_val_hr1, rel_tol=0.0, abs_tol=1e-8)
        if improved or (same_hr1 and val_metrics["HR@10"] > best_val_hr10):
            best_val_hr1 = val_metrics["HR@1"]
            best_val_hr10 = val_metrics["HR@10"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train context-adaptive fusion using the SFT model.")
    parser.add_argument("--cache_tag", type=str, default="sft_best_full")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--run_name", type=str, default="gate_default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=192)
    parser.add_argument("--eval_batch_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--residual_scale", type=float, default=0.20)
    parser.add_argument("--poll_seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(RESULTS_DIR)
    set_global_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cache_meta = wait_for_cache_ready(args.cache_tag, args.poll_seconds)
    print(f"Cache ready, loading from {args.cache_tag}: {cache_meta['split_sizes']}")

    device = torch.device(args.device)
    train_split = move_split_to_device(torch.load(cache_path(args.cache_tag, "train_features.pt"), map_location="cpu", weights_only=False), device)
    val_split = move_split_to_device(torch.load(cache_path(args.cache_tag, "val_features.pt"), map_location="cpu", weights_only=False), device)
    test_split = move_split_to_device(torch.load(cache_path(args.cache_tag, "test_features.pt"), map_location="cpu", weights_only=False), device)

    base_alpha, val_fixed = find_best_fixed_alpha(
        val_split["sas_scores"],
        val_split["llm_scores"],
        val_split["targets"],
        args.eval_batch_size,
    )
    test_fixed = evaluate_fixed_alpha(
        test_split["sas_scores"],
        test_split["llm_scores"],
        test_split["targets"],
        base_alpha,
        args.eval_batch_size,
    )

    model = ContextAdaptiveFusion(
        context_dim=train_split["context_hidden"].size(1),
        stats_dim=train_split["stats"].size(1),
        stats_mean=train_split["stats"].float().mean(dim=0),
        stats_std=train_split["stats"].float().std(dim=0),
        base_alpha=base_alpha,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)

    model, history = train_model(
        model=model,
        train_split=train_split,
        val_split=val_split,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
    )

    val_metrics, val_coeff = evaluate_model(model, val_split, args.eval_batch_size)
    test_metrics, test_coeff = evaluate_model(model, test_split, args.eval_batch_size)

    model_path = os.path.join(RESULTS_DIR, f"{args.run_name}_model.pt")
    torch.save(model.state_dict(), model_path)
    result_path = os.path.join(RESULTS_DIR, f"{args.run_name}_results.json")
    save_json(
        result_path,
        {
            "cache_tag": args.cache_tag,
            "run_name": args.run_name,
            "config": vars(args),
            "split_sizes": cache_meta["split_sizes"],
            "baseline": {
                "best_fixed_alpha": base_alpha,
                "fixed_fusion_val": val_fixed,
                "fixed_fusion_test": test_fixed,
            },
            "dynamic_fusion": {
                "val": val_metrics,
                "test": test_metrics,
                "val_coeff_mean": val_coeff,
                "test_coeff_mean": test_coeff,
                "history": history,
                "model_path": model_path,
            },
        },
    )
    print(f"Saved results to {result_path}")


if __name__ == "__main__":
    main()
