"""Shared hand-set direction-code helper used by both training and inference."""

import torch


def apply_handset_direction(alm_emb: torch.Tensor, K: int, direction) -> torch.Tensor:
    """Overwrite the K-th token block with a +1/-1/neutral direction code at structure-RMS scale."""
    if alm_emb.dim() == 1:
        alm_emb = alm_emb.unsqueeze(0)
    B = alm_emb.shape[0]
    hd = alm_emb.shape[1] // K
    ae = alm_emb.view(B, K, hd)
    struct = ae[:, :K - 1, :]  # keep autograd graph; only the K-th block is overwritten
    scale = struct.detach().pow(2).mean(dim=(1, 2)).clamp_min(1e-6).sqrt()
    code = torch.zeros(B, hd, device=alm_emb.device, dtype=alm_emb.dtype)
    half = hd // 2
    if not torch.is_tensor(direction):
        direction = torch.full((B,), float(direction), device=alm_emb.device)
    tdv = direction.to(alm_emb.device).view(-1).float()
    hi = tdv > 0
    lo = tdv < 0
    if hi.any():
        code[hi, :half] = scale[hi].unsqueeze(1)
    if lo.any():
        code[lo, half:] = scale[lo].unsqueeze(1)
    return torch.cat([struct, code.unsqueeze(1)], dim=1).reshape(B, K * hd)
