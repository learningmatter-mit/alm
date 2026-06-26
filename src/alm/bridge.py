"""Language-to-atomistic bridge modules feeding MatterGen's conditioning interface."""

import torch
import torch.nn as nn

class AtomsMapper(nn.Module):
    def __init__(self, hidden_dim: int = 4096, mid_dim: int = 2048, out_dim: int = 512, K: int = 8):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, K*hidden_dim) flat or (B, K, hidden_dim) direct.
        if x.dim() == 2:
            B = x.size(0)
            x = x.view(B, self.K, self.hidden_dim)
        return self.proj(x).mean(dim=1)


class _ProducerBlock(nn.Module):
    """One Q-Former block: queries cross-attend source, self-attend, then MLP (pre-LN + residual)."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.ln_cross_q = nn.LayerNorm(dim)
        self.ln_cross_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)

        self.ln_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)

        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        # queries: (B, M, dim), source: (B, S, dim)
        q = self.ln_cross_q(queries)
        kv = self.ln_cross_kv(source)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        queries = queries + attn_out

        s = self.ln_self(queries)
        self_out, _ = self.self_attn(s, s, s, need_weights=False)
        queries = queries + self_out

        queries = queries + self.mlp(self.ln_mlp(queries))
        return queries


class AtomsMapperProducerConsumer(nn.Module):
    def __init__(self, hidden_dim: int = 4096, out_dim: int = 512,
                 num_queries: int = 16, depth: int = 2, n_heads: int = 8,
                 source_len: int = 136, n_context: int = 128,
                 mlp_ratio: int = 4, dropout: float = 0.0,
                 pool: str = "none", out_norm: bool = False):
        super().__init__()
        if hidden_dim <= 0 or out_dim <= 0 or source_len <= 0 or num_queries <= 0:
            raise ValueError(
                "hidden_dim, out_dim, source_len, and num_queries must be positive; "
                f"got hidden_dim={hidden_dim}, out_dim={out_dim}, "
                f"source_len={source_len}, num_queries={num_queries}"
            )
        if n_context < 0 or n_context > source_len:
            raise ValueError(
                f"n_context must be in [0, source_len]; got n_context={n_context}, "
                f"source_len={source_len}"
            )
        if out_dim % n_heads != 0:
            raise ValueError(
                f"out_dim={out_dim} must be divisible by n_heads={n_heads}"
            )
        if pool not in ("none", "mean", "query0"):
            raise ValueError(
                f"pool must be 'none'|'mean'|'query0'; got {pool!r}"
            )
        self.pool = pool  # "none" emits (B,M,512) for IP-Adapter; "mean"/"query0" collapse to (B,512).
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_queries = num_queries
        self.depth = depth
        self.n_heads = n_heads
        self.source_len = source_len  # S = N + K
        self.n_context = n_context    # N

        self.in_proj = nn.Linear(hidden_dim, out_dim)

        self.pos_emb = nn.Parameter(torch.zeros(1, source_len, out_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        # type 0 for the first N context rows, type 1 for the last K atoms rows.
        self.type_emb = nn.Embedding(2, out_dim)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        type_ids = torch.zeros(source_len, dtype=torch.long)
        if n_context < source_len:
            type_ids[n_context:] = 1
        self.register_buffer("type_ids", type_ids, persistent=False)

        self.query_emb = nn.Parameter(torch.zeros(1, num_queries, out_dim))
        nn.init.normal_(self.query_emb, std=0.02)

        self.blocks = nn.ModuleList([
            _ProducerBlock(out_dim, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])

        self.final_ln = nn.LayerNorm(out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        # Zero-init so the bridge is a near-zero residual into the zero-init-V consumer at step 0.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        # LayerNorm after out_proj bounds the CFG delta scale; zero-init output stays ~0.
        self.out_norm = nn.LayerNorm(out_dim) if out_norm else None

        self.dir_head = nn.Linear(out_dim, 2)
        # Learned +/-direction prototype for the cosine-margin directional loss.
        self.dir_proto = nn.Parameter(torch.randn(out_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, S*hidden_dim) flat or (B, S, hidden_dim) direct.
        if x.dim() == 2:
            expected = self.source_len * self.hidden_dim
            if x.size(1) != expected:
                raise ValueError(
                    f"flat QFormer input has dim={x.size(1)}, expected "
                    f"source_len*hidden_dim={self.source_len}*{self.hidden_dim}={expected}. "
                    "Check bridge_source/qformer_context_tokens/K checkpoint metadata."
                )
            x = x.view(x.size(0), self.source_len, self.hidden_dim)
        elif x.dim() == 3:
            if x.size(1) != self.source_len or x.size(2) != self.hidden_dim:
                raise ValueError(
                    f"sequence QFormer input has shape={tuple(x.shape)}, expected "
                    f"(B, {self.source_len}, {self.hidden_dim}). Check ALM extraction."
                )
        else:
            raise ValueError(
                f"QFormer input must be flat (B,S*H) or sequence (B,S,H); got {tuple(x.shape)}"
            )
        B = x.size(0)

        src = self.in_proj(x)                      # (B, S, out_dim)
        src = src + self.pos_emb
        src = src + self.type_emb(self.type_ids).unsqueeze(0)

        queries = self.query_emb.expand(B, -1, -1)  # (B, M, out_dim)

        for block in self.blocks:
            queries = block(queries, src)

        queries = self.final_ln(queries)
        out = self.out_proj(queries)                # (B, M, out_dim)
        if self.out_norm is not None:
            out = self.out_norm(out)
        # qformer_pool ablation: collapse M query tokens to (B, out_dim).
        if self.pool == "mean":
            out = out.mean(dim=1)                   # (B, out_dim)
        elif self.pool == "query0":
            out = out[:, 0, :]                      # (B, out_dim)
        return out

    def direction_logits(self, am_out: torch.Tensor,
                         pool: str = "mean") -> torch.Tensor:
        # am_out (B, M, out_dim) or already-pooled (B, out_dim) -> (B, 2).
        if am_out.dim() == 2:
            return self.dir_head(am_out)
        if pool == "mean":
            pooled = am_out.mean(dim=1)
        elif pool == "query0":
            pooled = am_out[:, 0, :]
        else:
            raise ValueError(f"unknown pool: {pool!r} (expected 'mean'|'query0')")
        return self.dir_head(pooled)

    def direction_cosine(self, am_out: torch.Tensor,
                         pool: str = "mean") -> torch.Tensor:
        """cos(pooled am_out, learned dir_proto) -> (B,) for the directional cosine-margin loss."""
        if am_out.dim() == 2:
            pooled = am_out
        elif pool == "query0":
            pooled = am_out[:, 0, :]
        else:  # mean
            pooled = am_out.mean(dim=1)
        p = self.dir_proto / (self.dir_proto.norm() + 1e-8)
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
        return (pooled * p).sum(dim=-1)            # (B,) cosine in [-1, 1]


class AtomsMapperConsumerOnly(nn.Module):
    def __init__(self, hidden_dim: int = 4096, mid_dim: int = 4096,
                 out_dim: int = 512, K: int = 8):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # Named `proj` to match AtomsMapper so load sites auto-detect mid_dim via proj.0.weight.shape[0].
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), self.K, self.hidden_dim)
        return self.proj(x)  # (B, K, out_dim)
