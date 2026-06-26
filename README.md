# Atomistic Language Models (ALM)

[![Code License](https://img.shields.io/badge/Code_License-Apache_2.0-olive)](https://opensource.org/licenses/Apache-2.0)

Unifying natural language and atomistics to understand, generate, and optimize materials, introduced in [**Atomistic Language Modeling**](https://arxiv.org/abs/2606.21395).

The **Atomistic Language Models (ALM)s** comprise an LLM backbone that (1) **understands** crystal structures (property prediction, Q&A), (2) **generates** them from natural-language descriptions, and (3) **edits/optimizes** them as instructed in text. This is achieved by bridging the LM (Qwen3) to a denoising-diffusion decoder through continuous projectors.

- **Understanding:** a frozen OrbV3 encoder embeds each atom, and a trainable MLP projects each embedding into the LLM feature space as soft tokens.
- **Generation:** the embeddings of K=8 learnable `[atoms_i]` output tokens are projected through a producer-consumer bridge (a learnable-query producer feeding a cross-attention consumer) into the diffusion decoder for crystal structure prediction (CSP) and _de novo_ generation (DNG).
- **T2C-FK:** a Text-to-Crystal Feynman-Kac particle sampler scores partial denoising trajectories to steer generation toward the requested composition and stoichiometry, or any other reward.

> **Steering MatterGen from your own model?** [`STEERING.md`](STEERING.md) is a self-contained recipe for conditioning MatterGen's diffusion on an arbitrary `(B, D_in)` embedding: the 5 fork touch-points, the producer-module contract, the cond_field YAML, CFG and guidance, and the bridge-kind matrix.

This repository is the **lean release**: the code to retrain all three stages and to generate and evaluate outputs for every benchmark family in the paper.

## Repository layout

```
src/alm/                 # the installable package (src layout)
  model.py               #   AtomisticLanguageModel (OrbV3 encoder + projector + LLM)
  bridge.py              #   AtomsMapper (pool) / ...ProducerConsumer / ...ConsumerOnly  (Hydra _target_: bridge.<Class>)
  aux_heads.py  samplers.py  direction_code.py  paths.py  _alm_bootstrap.py
  utils/                 #   datasets, collate, task registry (__init__) + composition_utils.py
  train/                 #   stage1.py (projector align) · stage2.py (LoRA) · stage3.py (bridge)
  inference/             #   generate.py (understand / generate) · generate_stage{1,2,3}.py
  eval/                  #   evaluation harness (see eval/README.md)
    lib/                 #     loader · text_generation · metrics · parsers · structure_metrics · fk_rewards · llm_judge · baselines · runs
    generation/          #     eval_csp · eval_dng · eval_edit · eval_almbench · eval_bridge_csp · eval_planner_csp · gen_dng_{native,bridge} · score_dng_hull · rerank_csp · compute_csp_rmse · eval_{app_consistency,atomtxt_direction,doping,polymorph} · csp_recovery
    understanding/       #     eval_llm4mat · eval_mattext · eval_mascqa · eval_language_retention · eval_mat2{props,mcq} · eval_matterchat · eval_gnome_fe · eval_text_{baseline,conditional} · eval_knowledge_retention_judge · aggregate_results
    eval_prompts/        #     prompt / target / prior-bound JSON consumed by the evals
scripts/                 # data prep, embedding caching, pair building (incl. ALM Bench task builders)
external/                # setup_mattergen.sh + mattergen_alm_steering.patch
pyproject.toml  environment.yml
```

After `pip install -e .` the entry points are exposed as console scripts (`alm-generate`, `alm-eval-csp`, `alm-eval-dng`, and the rest below) and as `python -m alm.<group>.<name>`; multi-GPU training launches with `torchrun --nproc-per-node=N -m alm.train.stageK`. Importing the `alm` package runs a one-time `sys.path` setup (`src/alm/_alm_bootstrap.py`) that exposes the flat module namespace the MatterGen fork's Hydra configs instantiate by bare `_target_` (e.g. `bridge.AtomsMapper`), so our own entry points need no `PYTHONPATH` export.

## Installation

Python 3.10, **CUDA-12 only** (the stack pins `torch==2.9.0+cu128`; CUDA-13 `nvidia-*` wheels collide on shared `.so` paths, so never let them install).

```bash
# Option A (conda/mamba, recommended): creates the env + installs torch (cu128) + this package
conda env create -f environment.yml && conda activate alm

# Option B (venv): install a CUDA-12 torch FIRST, then the package
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .

# Both options then need the MatterGen fork (required for Stage 3 / CSP / DNG; not on PyPI):
bash external/setup_mattergen.sh
cd external/mattergen && bash install_for_h200.sh && bash build_pyg_for_torch29.sh && cd ../..
```

**FlashAttention-2.** The model loads with `attn_implementation="flash_attention_2"` by default (this is the exact configuration the released checkpoints were trained and evaluated with). Install it with `pip install flash-attn --no-build-isolation`; because it often has to compile against your specific CUDA/torch, prebuilt wheels save a lot of time (see **https://mjunya.com/flash-attention-prebuild-wheels/**). If you can't install it, pass `--attn_implementation sdpa` (or `load_alm(..., attn_implementation="sdpa")`) to fall back to PyTorch SDPA, which is correctness-equivalent and slightly slower.

## Quickstart: download a checkpoint and run

After installing (above), grab the released weights and run inference, with no training and no cluster:

```bash
hf download LearningMatter/AtomisticLanguageModels --local-dir ./checkpoints  # one repo, subdir per model
export ALM_CHECKPOINTS=./checkpoints

# (a) understanding: text (optionally + a structure) -> text   [ALM Core]
alm-generate understand \
    --alm_checkpoint $ALM_CHECKPOINTS/alm-core \
    --prompt "What makes a material a good thermoelectric?"

# (b) generation: text description -> crystal structure (CIF)   [ALM Gen, consumer-only bridge / mattergen_base]
alm-generate generate \
    --alm_checkpoint $ALM_CHECKPOINTS/alm-gen \
    --atoms_mapper   $ALM_CHECKPOINTS/alm-gen/atoms_mapper.pt \
    --mattergen_pretrained mattergen_base \
    --prompt "A cubic rock-salt oxide of magnesium." --num_samples 4 --guidance_factor 0.5 --out_dir gen_out

# (c) reproduce a benchmark number (de-novo S/U/N/MSUN; headline g=0.5: SUN 7.80% (MP-20), MSUN 35.2% (LeMat-GenBench))
alm-eval-dng \
    --alm_checkpoint $ALM_CHECKPOINTS/alm-gen \
    --atoms_mapper   $ALM_CHECKPOINTS/alm-gen/atoms_mapper.pt \
    --mattergen_pretrained mattergen_base --guidance_factor 0.5 --num_samples 1000 --out_root out --run_id dng
```

## Models & data (HuggingFace)

Two repos under the [`LearningMatter`](https://huggingface.co/LearningMatter) org: a **model** repo (`AtomisticLanguageModels`, one subdir per model) and a **dataset** repo (`ALM-Bench`). Bridge/projector blobs are weights-only (optimizer state stripped); ALM Edit additionally bundles a full fine-tuned Qwen3-8B.

**Model repo** `LearningMatter/AtomisticLanguageModels`:

| subdir | what | size |
|---|---|---|
| `stage1-projector/` | OrbV3→Qwen3 projector | ~70 MB |
| `alm-core/`         | **ALM Core** (understanding): Qwen3-8B LoRA (r128) + projector | ~3.7 GB |
| `alm-gen/`          | **ALM Gen** (DNG): consumer-only bridge (r8) over `mattergen_base` | ~0.3 GB |
| `alm-edit/`         | **ALM Edit** (CSP and editing): producer-consumer bridge + full-FT Qwen3-8B (`llm_full_ft/`) + `csp_backbone/` decoder | ~17 GB |

**Dataset repo** `LearningMatter/ALM-Bench`: `alm_bench/` (atomtxt·app·ood·polymorph·doping + held-out `eval/`) delineated from `pretraining/` (describe·csp), ~4 GB.

ALM Gen loads its r8 bridge LoRA directly (pass the subdir as `--alm_checkpoint`); ALM Edit is full-FT (auto-detected `llm_full_ft/`; pass `--bridge_lora_dir none`).

**Other training data** (for retraining from scratch):
- **LLM4Mat-Bench:** download the folder from [Google Drive](https://drive.google.com/drive/folders/12n3H9BU3AoQn7ikeR7PUrmmPRZ4LyvdX?usp=share_link).
- **GPT-Narratives:** [`yjeong/GPT-Narratives-for-Materials`](https://huggingface.co/datasets/yjeong/GPT-Narratives-for-Materials) (the `describe`, `csp`, and `ood` buckets derive from this via `scripts/build_*_pairs.py`).
- **CSP/DNG benchmarks:** MP-20 and MPTS-52 via `helper` download scripts; MP-2020 hull via `scripts/fetch_mp_hull.py`.

**Where things go** (set once, then the commands above/below work):
```bash
export ALM_CHECKPOINTS=./checkpoints              # where weights were downloaded
export ALM_DATA_ROOT=/path/to/data               # LLM4Mat-Bench/, GPT-Narratives/, alm-data/ live here
export ALM_EVAL_RESULTS_ROOT=./eval_results       # where eval scripts write metrics.json
```
Then point `--data_parent_path`, `--pairs_parquets`, `--alm_checkpoint`, etc. at these locations (the `scripts/build_*` and `scripts/cache_*` utilities build the cached embeddings + `pairs*.parquet` each training stage consumes). To reproduce our exact training mixture, use the `alm-data` buckets directly.

## Retraining

Three stages, each consuming the previous stage's checkpoint. Launch each with `torchrun -m alm.train.stageN` (examples use 8 GPUs; scale `--nproc-per-node` to your node).

### Stage 1: projector alignment
```bash
torchrun --nproc-per-node=8 -m alm.train.stage1 \
    --data_parent_path  $ALM_DATA_ROOT/LLM4Mat-Bench \
    --cached_embs_parent_path /tmp/cached_embs \
    --learning_rate 1e-3 --num_epochs 1 --checkpoint_save_path runs/stage1
```

### Stage 2: LoRA instruction tuning
```bash
torchrun --nproc-per-node=8 -m alm.train.stage2 \
    --resume_from_stage1 runs/stage1/<ckpt> \
    --total_optim_steps 12000 \
    --lora_rank 128 --lora_alpha 256 --lora_lr 2e-4 --projector_lr 2e-5 \
    --save_dir runs/stage2          # resume: --resume_from_stage2 runs/stage2/step=N
```

### Stage 3: producer-consumer bridge
Requires the from-scratch CSP-mode MatterGen backbone (`csp_backbone`, built via `scripts/build_csp_backbone_cache.py` + a `mattergen-train --config-name=csp` run) and the 7-bucket `pairs*.parquet` (built by `scripts/build_*_pairs.py`).
```bash
torchrun --nproc-per-node=8 -m alm.train.stage3 \
    --alm_checkpoint runs/stage2/step=12000 --out_dir runs/stage3 \
    --pairs_parquets pairs.parquet,pairs_csp.parquet,pairs_ood.parquet,pairs_app.parquet,pairs_atomtxt.parquet,pairs_polymorph.parquet,pairs_doping.parquet \
    --pairs_weights 0.08,0.15,0.08,0.04,0.40,0.15,0.10 \
    --bridge_kind producer-consumer --bridge_tenc_fuse --full_finetuning \
    --aux_target_kind composition --aux_lambda 1.0 --contrastive_lambda 0.02 \
    --lm_loss_json_lambda 0.5 --mattergen_model_path runs/csp_backbone
```

## Inference (arbitrary prompts)

One CLI over the same generative machinery the eval harness uses:
```bash
# understand: text (optionally + a structure file) -> text answer
alm-generate understand \
    --alm_checkpoint runs/stage2/step=12000 \
    --prompt "What makes a material a good thermoelectric?"
alm-generate understand \
    --alm_checkpoint runs/stage2/step=12000 --structure my_crystal.cif \
    --prompt "Describe this material and predict its band gap."

# generate: text description -> crystal structure(s) (CIF), via the diffusion bridge
alm-generate generate \
    --alm_checkpoint runs/stage2/step=12000 \
    --atoms_mapper runs/stage3/step=30000/atoms_mapper.pt \
    --prompt "A cubic rock-salt oxide of magnesium." --num_samples 4 --out_dir gen_out
```

## Evaluating

Run any entry point with `--help` for the full flag list. Output roots default to `./eval_results` (override with `ALM_EVAL_RESULTS_ROOT`). Every eval runs as `python -m alm.eval.<group>.<name>`; the headline benchmarks also install as the console scripts `alm-eval-csp`, `alm-eval-dng`, `alm-eval-edit`, `alm-eval-almbench`, `alm-eval-llm4mat`, and `alm-eval-aggregate`.

```bash
# Stage-2 understanding (LLM4Mat-Bench, MatText, MaScQA, language retention)
python -m alm.eval.understanding.eval_llm4mat            --checkpoint runs/stage2/step=12000
python -m alm.eval.understanding.eval_mattext            --checkpoint runs/stage2/step=12000
python -m alm.eval.understanding.eval_mascqa             --checkpoint runs/stage2/step=12000
python -m alm.eval.understanding.eval_language_retention --model alm --checkpoint runs/stage2/step=12000
python -m alm.eval.understanding.aggregate_results       --run_id step=12000      # headline table

# Stage-3 CSP (MP-20 / MPTS-52): native CSP-mode backbone, composition-enforced
python -m alm.eval.generation.eval_csp --ckpt_dir runs/csp_backbone \
    --max_rows 1000 --guidance_factor 0.5 --out_dir eval_results/csp
#   (planner front-end variant: python -m alm.eval.generation.eval_planner_csp)

# Stage-3 DNG (de-novo generation -> S/U/N/MSUN), optionally T2C-FK steered
python -m alm.eval.generation.eval_dng \
    --alm_checkpoint runs/stage2/step=12000 \
    --atoms_mapper runs/stage3/step=30000/atoms_mapper.pt \
    --num_samples 1000 --guidance_factor 1.0 --out_root eval_results/dng --run_id alm_dng
python -m alm.eval.generation.score_dng_hull --cif_dir eval_results/dng/alm_dng ...   # strict-SUN re-score

# ALM Bench: text-conditioned editing, all four tasks in one command
python -m alm.eval.generation.eval_almbench \
    --alm_checkpoint runs/stage2/step=12000 \
    --atoms_mapper runs/stage3/step=30000/atoms_mapper.pt \
    --mattergen_model_path runs/csp_backbone \
    --guidance_factor 0.5 --max_rows 100 --out_dir eval_results/almbench
#   (single task: python -m alm.eval.generation.eval_edit --task {atomtxt,polymorph,doping,app}; app needs OPENAI_API_KEY)
```

## License

Apache-2.0 (see `LICENSE`).
