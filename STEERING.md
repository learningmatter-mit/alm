# Steering MatterGen with a Custom Conditioning Embedding

A self-contained recipe for driving MatterGen's crystal diffusion from an arbitrary `(B, D_in)` embedding produced by any model. This repo steers MatterGen from a frozen LLM's hidden states; the recipe generalizes so any producer module works.

## 1. What this is

MatterGen's adapter framework (`GemNetTAdapter` wrapping `GemNetTCtrl`) conditions the score network on extra signals layered on top of a pretrained checkpoint, through stock classifier-free-guidance (CFG) plumbing. Steering it with a custom embedding adds **exactly one** new piece of state: one `cond_field` named (here) `alm_embedding`, backed by one producer `nn.Module` that maps the `(B, D_in)` embedding into the diffusion conditioning space. Everything else (`PropertyEmbedding.forward`, the CFG dropout/null/`torch.where` machinery, and the per-block injection inside `GemNetTCtrl`) is MatterGen's own code, reused unchanged. The producer projects to MatterGen's `hidden_dim` (**512**) and emits one of two shapes: a single pooled vector `(B, 512)` added globally per atom, or a token sequence `(B, M, 512)` cross-attended per atom. The trainer attaches the embedding **live (no `.detach()`)**, so gradient from the diffusion loss flows straight back into the producer and the model that generated the embedding.

## 2. Data-flow chain

```
your model
   │   e : (B, D_in)               # arbitrary embedding (here D_in = K*4096 = 32768)
   ▼
ChemGraph["alm_embedding"] = e      # attach as a cond_field (in the trainer)
   ▼
diffusion_module.calc_loss(chemgraph)         # train
   │   internally runs, per cond_field:
   ▼
PropertyEmbedding.forward(batch)              # mattergen/property_embeddings.py
   │   1. scaler(x)                            # Identity for raw embeddings
   │   2. conditional_embedding_module(x)      # YOUR producer -> (B,512) or (B,M,512)
   │   3. unconditional_embedding_module(x)    # null branch (Zeros* / Learned*)
   │   4. torch.where(use_unconditional_embedding, uncond, cond)   # CFG per-row pick
   ▼
cond_adapt[field]  (B,512) | (B,M,512)
   ▼
GemNetTCtrl.forward  (mattergen/common/gemnet/gemnet_ctrl.py)
   │   per atom (gather cond by `batch`), per block i:
   │     pooled (B,512):  h_adapt = concat-MLP([h, cond]); mixin (zero-init)   # pre-block
   │     sequence (B,M,512): cross-attn(q=h, k=v=cond); mixin                  # IP-Adapter, post-block
   │     × bridge_gate[i]              # "alm_embedding" only (special-cased)
   │     × cond_adapt_mask_per_atom    # 1=conditional, 0=unconditional (CFG)
   │     h = h + masked_gated_residual
   ▼
diffusion score  →  L_diff  →  ∂/∂(producer)   # gradient reaches producer + upstream model
```

The conditional branch runs for **every** row, even masked ones; CFG selects per row via `torch.where`. The zero-init final projection (producer side) and zero-init mixin / zero-init attention-V (consumer side) make the bridge a no-op at step 0, so a fine-tuned adapter starts identical to the base model.

## 3. The 5 MatterGen touch-points

The minimal set of fork edits to reproduce this (see `external/mattergen_alm_steering.patch`):

| File | What |
|---|---|
| `mattergen/common/utils/globals.py` | Append your field name to `PROPERTY_SOURCE_IDS` (the registry of valid cond_fields). |
| `mattergen/property_embeddings.py` | (Only if you need a sequence-shaped null) add `EmbeddingSequence` / `ZerosEmbeddingSequence`; and the `PropertyEmbedding.forward` broadcast fix so `(B,1)` CFG mask aligns against `(B,M,512)` embeddings (`while use_unconditional_embedding.dim() < conditional_embedding.dim(): unsqueeze(-1)`). |
| `mattergen/common/gemnet/gemnet_ctrl.py` | The consumer: `cond_adapt_use_ipa` (sequence → post-block cross-attn) / `cond_adapt_use_xattn` (sequence → pre-block cross-attn) options; the default concat-MLP path handles pooled `(B,512)`. Also the per-block `bridge_gate` (special-cased on field name `"alm_embedding"`). |
| `mattergen/adapter.py` | `GemNetTAdapter.__init__` rewrites each adapter field's `unconditional_embedding_module` to a `Zeros*` of the matching shape (dispatches on a `K` attribute → `ZerosEmbeddingSequence`, else `ZerosEmbedding`); skips any `Learned*` null. |
| `mattergen/scripts/finetune.py` | `init_adapter_lightningmodule_from_pretrained` preserves user-supplied `adapter.gemnet` kwargs (e.g. `cond_adapt_use_ipa`, `cond_adapt_n_heads`) across the merge with the pretrained denoiser config. |

A **pooled-only** bridge (`(B,512)`, concat-MLP consumer) needs just the `globals.py` registry entry plus a producer module and a YAML. Touch-points 3 (consumer) and 4 (auto-rewrite) already cover `(B,512)` with stock code.

## 4. Plug-in recipe

### A. Register the cond_field name

`mattergen/common/utils/globals.py`:

```python
PROPERTY_SOURCE_IDS = [
    ...,
    "alm_embedding",   # your field name
]
```

### B. Producer module contract

Write a plain `nn.Module`, importable by a bare module name (see section 7):

- `forward(x)` takes a **positional** tensor `x` of shape `(B, D_in)`. Sampling collates the value to `(B, D_in)`; accept a flat `(B, D_in)` and reshape internally when the true shape is `(B, K, ...)`.
- Return `(B, 512)` (pooled) **or** `(B, M, 512)` (sequence), where `512 == diffusion hidden_dim` (`lightning_module.diffusion_module.model.hidden_dim`).
- **Zero-init the final projection** so the conditional branch starts at ~0 (no-op bridge at step 0). Example (`AtomsMapper` in `src/alm/bridge.py`, pooled):

```python
class AtomsMapper(nn.Module):
    def __init__(self, hidden_dim=4096, mid_dim=2048, out_dim=512, K=8):
        super().__init__()
        self.K, self.hidden_dim, self.out_dim = K, hidden_dim, out_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim), nn.GELU(), nn.Linear(mid_dim, out_dim))

    def forward(self, x):                       # x: (B, K*hidden_dim) or (B, K, hidden_dim)
        if x.dim() == 2:
            x = x.view(x.size(0), self.K, self.hidden_dim)
        return self.proj(x).mean(dim=1)         # (B, 512)
```

For a sequence producer, drop the `.mean(dim=1)` (returns `(B, K, 512)`; see `AtomsMapperConsumerOnly` in `src/alm/bridge.py`), or run learned queries that cross-attend the input and emit `(B, M, 512)` (see `AtomsMapperProducerConsumer` in `src/alm/bridge.py`, whose `out_proj` weight+bias are zero-init).

### C. cond_field YAML

`mattergen/conf/lightning_module/diffusion_module/model/property_embeddings/<field>.yaml`. For a **pooled `(B,512)`** producer (`src/alm/.../alm_embedding.yaml`):

```yaml
_target_: mattergen.property_embeddings.PropertyEmbedding
name: alm_embedding
unconditional_embedding_module:           # rewritten to ZerosEmbedding by adapter.py
  _target_: mattergen.property_embeddings.EmbeddingVector
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}
conditional_embedding_module:
  _target_: bridge.AtomsMapper            # bare module name (needs src/alm on PYTHONPATH)
  hidden_dim: 4096
  mid_dim: 2048
  out_dim: ${lightning_module.diffusion_module.model.hidden_dim}
  K: 8
scaler:
  _target_: torch.nn.Identity             # raw embeddings: no scaling
```

For a **sequence `(B,M,512)`** producer, the unconditional must be sequence-shaped so its `K` attribute triggers the `ZerosEmbeddingSequence` rewrite and matches the conditional shape under `torch.where`:

```yaml
unconditional_embedding_module:
  _target_: mattergen.property_embeddings.EmbeddingSequence
  hidden_dim: ${lightning_module.diffusion_module.model.hidden_dim}
  K: 8          # = M, the producer's output sequence length
conditional_embedding_module:
  _target_: bridge.AtomsMapperConsumerOnly
  ...
```

### D. Wire the consumer (sequence output only)

Tell `GemNetTCtrl` to cross-attend the sequence; a pooled `(B,512)` needs nothing and goes through the default concat-MLP path. In the adapter cfg's `adapter.gemnet`:

```yaml
cond_adapt_use_ipa: ["alm_embedding"]   # post-block IP-Adapter cross-attn (zero-init V)
cond_adapt_n_heads: 4                    # MUST divide emb_size_atom (512)
```

(`cond_adapt_use_xattn` is the pre-block variant.) This repo sets it inline in `src/alm/train/stage3.py::_make_adapter_cfg`.

### E. Supply the embedding at train + infer

**Train.** Attach the live embedding and call the loss (in `src/alm/train/stage3.py`):

```python
chemgraph["alm_embedding"] = e          # e: (B, D_in); NO .detach() for grad flow
loss_diff, metrics = diffusion_module.calc_loss(chemgraph)
```

**Infer.** Stamp the per-prompt vector onto the condition loader and sample at guidance `g` (in `src/alm/inference/generate_stage3.py`):

```python
props_to_stamp["alm_embedding"] = e_vec.detach().cpu().unsqueeze(0)  # (1, D_in)
# overrides into the sampling hydra config:
#   sampler_partial.guidance_scale=<g>
#   +condition_loader_partial.num_samples=<N>
```

`draw_samples_from_sampler`'s `properties_to_condition_on={field: e.unsqueeze(0)}` serves only the trained-fields assert; the condition loader stamps the *actual* conditioning values onto every ChemGraph. **Operating point `g = 0.5`.**

## 5. CFG & guidance

- **Dropout (train):** `diffusion_module.pre_corruption_fn.p_unconditional` (default `0.2`) randomly routes rows to the unconditional branch, so the model learns both conditional and unconditional scores.
- **Null rewrite:** `GemNetTAdapter.__init__` overwrites each adapter field's `unconditional_embedding_module` with a `Zeros*` of the matching shape (`ZerosEmbeddingSequence` if the original carried a `K` attribute, else `ZerosEmbedding`), so a new field never perturbs the unconditional score. A `Learned*` null opts out of this rewrite and keeps a learnable baseline.
- **Guidance `g`** (`sampler_partial.guidance_scale`): `g=0` ⇒ conditioning fully off (`alm_embedding` removed, the *no-bridge baseline* and never an operating point); `g=1` ⇒ pure conditional; `0<g<1` interpolates; `g>1` extrapolates (NaNs at high g). **`g=0.5` is the operating point.**
- **`dropout_fields_iid` AND-gate pitfall:** with `dropout_fields_iid=False` (default), the not-NaN mask AND-s across **all** cond_fields, so a NaN in *any* field forces *every* field unconditional for that row, silently starving the embedding. Adding a second cond_field (e.g. a `task_direction` scalar) whose value is NaN on most rows demands `diffusion_module.pre_corruption_fn.dropout_fields_iid = True` (`dropout_fields_iid`) to decouple per-field dropout.

## 6. Bridge-kind matrix

Selected in `src/alm/train/stage3.py::_make_adapter_cfg` (arg `bridge_kind`):

| bridge_kind | producer output | consumer path | `_make_adapter_cfg` |
|---|---|---|---|
| `pool` | `(B, 512)` | concat-MLP, pre-block (`h_adapt = MLP([h, cond])`), zero-init mixin | no `cond_adapt_use_*`; just `bridge_gate_init` |
| `producer-consumer` | `(B, M, 512)` (M learned queries) | IP-Adapter cross-attn, **post-block**, zero-init V | `cond_adapt_use_ipa: [alm_embedding]`, `cond_adapt_n_heads` |
| `producer-consumer-pool` | `(B, 512)` (same producer, mean-pooled in `forward`) | concat-MLP, pre-block (same as `pool`) | no `cond_adapt_use_*`; `bridge_gate_init` |
| `consumer-only` | `(B, K, 512)` (per-position proj, no pool) | IP-Adapter cross-attn, **post-block**, zero-init V | `cond_adapt_use_ipa: [alm_embedding]`, `cond_adapt_n_heads` |

Sequence consumers require `cond_adapt_n_heads` to divide `512`. The unconditional YAML must match: `EmbeddingVector` for pooled, `EmbeddingSequence(K=M)` for sequence.

## 7. Common mistakes

- **Bare-module `_target_` needs `src/alm` on `PYTHONPATH`.** The YAML imports producers as bare names like `bridge.AtomsMapper` instead of dotted package paths. Set `PYTHONPATH=<repo>/src/alm:$PYTHONPATH` for `mattergen-finetune`, `mattergen-generate`, **and at every Lightning checkpoint resume/load**. Hydra re-instantiates the producer from the stored config on every resume, so the path must be present for resumes as well as the initial launch. (ALM's own entry points already set this up via `import alm`; only the MatterGen-native CLIs need the export.)
- **`512` / `4096` / `K=8` are magic constants.** `512` is the diffusion `hidden_dim`, **hardcoded** in `_make_adapter_cfg` (the model config is not consulted). `4096` is the producer's `D_in`-per-token here; `K=8` is its token count. Change all three consistently when the dims differ, and keep the producer's emitted `out_dim == 512`.
- **`"alm_embedding"` is special-cased for `bridge_gate`.** `GemNetTCtrl` creates and applies the per-block `bridge_gate` parameter **only** for a field literally named `"alm_embedding"`. A differently-named field takes the plain un-gated additive path. Rename in both `globals.py` and the gate check on any change.
- **Producer `D_in` is implicit.** Nothing validates that the stamped embedding matches the producer's expected input width; a mismatch surfaces as a reshape error inside `forward`. Validate explicitly in `forward`; the learnable-query producer raises with the expected `source_len*hidden_dim`.
- **`IPA_INIT_MODE` and `IPA_C_SCALE` read from `os.environ`.** The IP-Adapter consumer's V-projection and mixin init scheme answers to environment variables (`IPA_INIT_MODE` ∈ {A,B,C}, default `A` = zero-init V + zero-init mixin; `IPA_C_SCALE` default `1e-3`) instead of config. Set them in the launch env for reproducibility.
