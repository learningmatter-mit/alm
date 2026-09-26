# ALM evaluation harness

This directory holds the full eval suite. **Understanding** (Stage-2) benchmarks each have one script taking `--checkpoint <stage2 step=N/ dir>` and writing `metrics.json` + `predictions.jsonl` under `$ALM_EVAL_RESULTS_ROOT/{benchmark}/{step=N}/` (default `./eval_results`). The **Generation** (Stage-3) evals (CSP, DNG, ALM-Bench editing) are documented in the top-level [`README.md`](../../../README.md) (`eval_csp.py`, `eval_dng.py`, `eval_edit.py`, `eval_almbench.py`).

After `pip install -e .`, run each script as `python -m alm.eval.understanding.<name>` (the headline benchmarks also install as the `alm-eval-*` console scripts).

## Shared modules
- `lib/loader.py`: `load_alm(checkpoint, ...)` (LoRA + projector, optional merge) and the base-Qwen3 reference loader for language-retention.
- `lib/text_generation.py`: batched greedy `inputs_embeds` generation; `atomistic=True/False` switch (`generate_batch`).
- `lib/parsers.py`: `extract_number`, `extract_choice` (run the file directly for a smoke test).
- `lib/metrics.py`: `mae`, `rmse`, `mad_mae_ratio`, `accuracy`, `weighted_f1`.
- `lib/structure_metrics.py`: validity, match-rate, and RMSD for generated structures.
- `lib/runs.py`: resolves the per-run output dir and writes `metrics.json` / `predictions.jsonl`.
- `lib/baselines.py`: published baseline numbers used in the headline tables.

## Per-benchmark scripts (understanding)

| Script | Source | Metric |
|---|---|---|
| `eval_llm4mat.py` | LLM4Mat-Bench held-out (`_DATASET_PROPERTIES` in `alm/utils/__init__.py`) | per-config × per-property MAE + MAD:MAE + validity_rate |
| `eval_matterchat.py` | MP test split (LLM4Mat-Bench mp/test proxy) | per-task MAE/RMSE or accuracy/weighted_f1 |
| `eval_mattext.py` | HF `n0w0f/MatText` `*-train-filtered` configs, one `--fold` split (live OrbV3 from CIF) | MAE per task |
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
python -m alm.eval.understanding.eval_language_retention --model alm --checkpoint <stage2>/step=12000 --task all --max_samples 200
python -m alm.eval.understanding.eval_language_retention --model base --task all --max_samples 200

# MaScQA (held-out 131 Qs):
python -m alm.eval.understanding.eval_mascqa --checkpoint <stage2>/step=12000

# Roll the understanding benchmarks into the headline table:
python -m alm.eval.understanding.aggregate_results --run_id step=12000
```

## Scoring conventions
- Classification accuracies (MatterChat classification tasks, Mat2MCQ, MMLU, GPQA, GSM8K, MaScQA MCQ) count unparseable and leaked answers as wrong, so the denominator is every question. The `*_valid_only` keys give the same metric over parseable answers only, and `validity_rate` / `leak_rate` report the failures.
- Regression metrics (MAE, RMSE, MAD:MAE) can only use outputs that contain a number; `validity_rate` reports the fraction that did.

## Data notes
- The generation evals (ALM Bench editing, DNG) read the `LearningMatter/ALM-Bench` dataset from `$ALM_DATA_ROOT/ALM-Bench`: each `eval_edit.py --task` / `eval_{atomtxt_direction,polymorph,doping,app_consistency}.py` defaults to `alm_bench/eval/<task>.parquet` (`strain` uses `eval/doping.parquet`; `describe`, which has no held-out split, samples `pretraining/describe.parquet`), and the DNG evals use `pretraining/describe.parquet` for prompts and the novelty reference.
- Stage the eval datasets with the `scripts/` data-prep utilities (`cache_embeddings_atomistic.py`, `build_*`); set `ALM_DATA_ROOT` to where they live. See the top-level README "Models & data".
- LLM4Mat-Bench `test` split ships CSVs but no `*.db` / `*_test_atom.flat.bin` cache, so pass `--split validation` until you cache test-split embeddings.
- The `alex_mp_20` LLM4Mat-Bench config is not cached by default; `eval_llm4mat.py` skips it unless you cache it via `scripts/cache_embeddings_atomistic.py --source ase_db --dataset_name alex_mp_20 --split <split> --data_path <LLM4Mat-Bench>/alex_mp_20/<split>.db`.
