# ALM evaluation harness

This directory holds the full eval suite. **Understanding** (Stage-2) benchmarks each have one script taking `--checkpoint <stage2 step=N/ dir>` and writing `metrics.json` + `predictions.jsonl` under `$ALM_EVAL_RESULTS_ROOT/{benchmark}/{step=N}/` (default `./eval_results`). The **Generation** (Stage-3) evals (CSP, DNG, ALM-Bench editing) are documented in the top-level [`README.md`](../../../README.md) (`eval_csp.py`, `eval_dng.py`, `eval_edit.py`, `eval_almbench.py`).

After `pip install -e .`, run each script as `python -m alm.eval.understanding.<name>` (the headline benchmarks also install as the `alm-eval-*` console scripts).

## Shared modules
- `lib/loader.py`: `load_alm(checkpoint, ...)` (LoRA + projector, optional merge) and the base-Qwen3 reference loader for language-retention.
- `lib/text_generation.py`: batched greedy `inputs_embeds` generation; `atomistic=True/False` switch (`generate_batch`).
- `parsers.py`: `extract_number`, `extract_choice` (run the file directly for a smoke test).
- `metrics.py`: `mae`, `rmse`, `mad_mae_ratio`, `accuracy`, `weighted_f1`.
- `structure_metrics.py`: validity, match-rate, and RMSD for generated structures.
- `runs.py`: resolves the per-run output dir and writes `metrics.json` / `predictions.jsonl`.
- `baselines.py`: static cited numbers from each benchmark paper (update as you fill the table).

## Per-benchmark scripts (understanding)

| Script | Source | Metric |
|---|---|---|
| `eval_llm4mat.py` | LLM4Mat-Bench held-out (`_DATASET_PROPERTIES` in `alm/utils.py`) | per-config × per-property MAE + MAD:MAE + validity_rate |
| `eval_matterchat.py` | MP test split (LLM4Mat-Bench mp/test proxy) | per-task MAE/RMSE or accuracy/weighted_f1 |
| `eval_mattext.py` | HF `n0w0f/MatText` test configs (live OrbV3 from CIF) | MAE per task |
| `eval_gnome_fe.py` | LLM4Mat-Bench `gnome` split, formation energy | MAE + RMSE + MAD:MAE |
| `eval_mat2props.py` | GPT-Narratives parquet; last 10% held-out unless `--id_list` | per-property MAE |
| `eval_mat2mcq.py` | 4-way element-MCQ synthesized from GPT-Narratives `atoms` (deterministic per `split_seed`) | accuracy |
| `eval_language_retention.py` | HF `cais/mmlu`, `openai/gsm8k`, `Idavidrein/gpqa` (gated; needs HF auth) | accuracy per task |
| `eval_mascqa.py` | `MaScQADataset(split="validation")` (131 stratified Qs) | mcq_accuracy + numerical_mae |

## One-shot examples

```bash
# LLM4Mat MP val smoke (5 props × 1000 samples each):
python -m alm.eval.understanding.eval_llm4mat --checkpoint <stage2>/step=12000 \
    --configs mp --split validation --max_samples 1000

# MatText (live OrbV3 path):
python -m alm.eval.understanding.eval_mattext --checkpoint <stage2>/step=12000 \
    --tasks perovskites,kvrh,gvrh --max_samples 1000

# Language retention: ALM vs the Qwen3-8B base reference:
python -m alm.eval.understanding.eval_language_retention --checkpoint <stage2>/step=12000 --task all --max_samples 200
python -m alm.eval.understanding.eval_language_retention --model base --task all --max_samples 200

# MaScQA (held-out 131 Qs):
python -m alm.eval.understanding.eval_mascqa --checkpoint <stage2>/step=12000

# Roll the understanding benchmarks into the headline table:
python -m alm.eval.understanding.aggregate_results --run_id step=12000
```

## Data notes
- Stage the eval datasets with the `scripts/` data-prep utilities (`cache_embeddings_*`, `build_*`); set `ALM_DATA_ROOT` to where they live. See the top-level README "Models & data".
- LLM4Mat-Bench `test` split ships CSVs but no `*.db` / `*_test_atom.flat.bin` cache, so pass `--split validation` until you cache test-split embeddings.
- The `alex_mp_20` LLM4Mat-Bench config is not cached by default; `eval_llm4mat.py` skips it unless you cache it via `scripts/cache_embeddings_atomistic_orbv3.py`.
