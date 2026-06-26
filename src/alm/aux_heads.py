"""Auxiliary supervision heads predicting target composition from AtomsMapper output."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from composition_utils import MAX_COUNT, N_ELEMENTS


class AuxHead(nn.Module):
    """Base class; subclasses set target_kind / target_dim and implement forward/loss."""
    target_kind: str = ""
    target_dim: int = 0

    def forward(self, am_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        return {}


class CompositionHead(AuxHead):
    """Multi-hot presence over Z=1..100, BCEWithLogits; pos_weight escapes the all-negative trivial minimum."""
    target_kind = "composition"
    target_dim = 100

    def __init__(self, in_dim: int = 512, pos_weight: float = 32.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, self.target_dim)
        # buffer (moves with .to(device), not trained); 32 ~= (1-p_pos)/p_pos for ~3/100 positives.
        self.register_buffer("pos_weight", torch.full((self.target_dim,), pos_weight))

    def forward(self, am_out: torch.Tensor) -> torch.Tensor:
        return self.proj(am_out)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            pred, target.float(), pos_weight=self.pos_weight
        )

    def metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        with torch.no_grad():
            pred_bin = torch.sigmoid(pred) > 0.5
            tgt = target.bool()
            tp = (pred_bin & tgt).sum().float()
            fp = (pred_bin & ~tgt).sum().float()
            fn = (~pred_bin & tgt).sum().float()
            return {
                "precision": tp / (tp + fp + 1e-9),
                "recall":    tp / (tp + fn + 1e-9),
            }


class CompositionCountHead(AuxHead):
    """Per-element counts: presence BCE (as CompositionHead) + count_lambda * CE on positive slots (counts in [0, MAX_COUNT])."""
    target_kind = "composition_count"
    target_dim = N_ELEMENTS

    def __init__(self, in_dim: int = 512, pos_weight: float = 32.0,
                 count_lambda: float = 1.0, max_count: int = MAX_COUNT):
        super().__init__()
        self.max_count = max_count
        self.count_lambda = count_lambda
        self.presence = nn.Linear(in_dim, N_ELEMENTS)
        self.count = nn.Linear(in_dim, N_ELEMENTS * (max_count + 1))
        self.register_buffer("pos_weight", torch.full((N_ELEMENTS,), pos_weight))

    def forward(self, am_out: torch.Tensor) -> dict:
        return {
            "presence": self.presence(am_out),
            "count": self.count(am_out).view(-1, N_ELEMENTS, self.max_count + 1),
        }

    def loss(self, pred: dict, target: torch.Tensor) -> torch.Tensor:
        counts_long = target.long().clamp_(0, self.max_count)
        presence_target = (counts_long > 0).float()

        bce = F.binary_cross_entropy_with_logits(
            pred["presence"], presence_target, pos_weight=self.pos_weight,
        )

        # CE on positive slots only; skip if the batch has none (keeps graph valid).
        pos_mask = (counts_long > 0)
        if pos_mask.any():
            count_logits_flat = pred["count"][pos_mask]
            count_target_flat = counts_long[pos_mask]
            ce = F.cross_entropy(count_logits_flat, count_target_flat)
        else:
            ce = torch.tensor(0.0, device=bce.device, dtype=bce.dtype)

        return bce + self.count_lambda * ce

    def metrics(self, pred: dict, target: torch.Tensor) -> dict:
        with torch.no_grad():
            counts_long = target.long().clamp_(0, self.max_count)
            presence_target = (counts_long > 0)

            pres_pred = torch.sigmoid(pred["presence"]) > 0.5
            tp = (pres_pred & presence_target).sum().float()
            fp = (pres_pred & ~presence_target).sum().float()
            fn = (~pres_pred & presence_target).sum().float()

            count_pred = pred["count"].argmax(dim=-1)
            pos_mask = presence_target
            if pos_mask.any():
                count_exact = (count_pred[pos_mask] == counts_long[pos_mask]).float().mean()
                count_mae = (count_pred[pos_mask] - counts_long[pos_mask]).abs().float().mean()
            else:
                count_exact = torch.tensor(0.0, device=tp.device)
                count_mae = torch.tensor(0.0, device=tp.device)

            # whole-composition match: all slots correct (positive or zero) per sample.
            whole_exact = (count_pred == counts_long).all(dim=-1).float().mean()

            return {
                "precision": tp / (tp + fp + 1e-9),
                "recall":    tp / (tp + fn + 1e-9),
                "count_exact_pos":   count_exact,
                "count_mae_pos":     count_mae,
                "whole_exact":       whole_exact,
            }


def build_aux_head(kind: str | None, in_dim: int = 512,
                   count_lambda: float = 1.0) -> AuxHead | None:
    """Factory; returns None when no aux loss is requested. count_lambda only affects composition_count."""
    if kind in (None, "none", ""):
        return None
    if kind == "composition":
        return CompositionHead(in_dim=in_dim)
    if kind == "composition_count":
        return CompositionCountHead(in_dim=in_dim, count_lambda=count_lambda)
    raise ValueError(f"unknown aux_target_kind: {kind!r}; "
                     f"expected one of: composition, composition_count, none")
