#!/usr/bin/env python
"""CSP match rate (M@1, M@K) for the bridged model: the LLM planner predicts atom counts and the ALM embedding conditions a CSP-mode MatterGen decoder."""

import argparse
import json
import os
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

import eval_planner_csp as epc
from bridge import canonical_bridge_kind
from generate_stage3 import load_alm_and_pl_module, get_alm_embedding
from paths import CHECKPOINTS, RUNS


def build_csp_condition_loader(target_comp: dict, K: int, alm_emb: torch.Tensor | None,
                               task_direction: float | None = None):
    """Observed-atom CSP condition loader (atoms fixed from target_comp) with the per-prompt alm_embedding stamped on every chemgraph."""
    from torch.utils.data import DataLoader
    from mattergen.common.data.collate import collate
    from mattergen.common.data.condition_factory import ChemGraphlistDataset, _collate_fn
    from mattergen.common.data.transform import SetProperty
    from mattergen.common.utils.data_utils import create_chem_graph_from_composition

    cg = create_chem_graph_from_composition(target_comp)
    graphs = [cg] * K
    # Stamp cond fields on the bridge vector's device (CPU only when alm_emb is None).
    _cond_device = alm_emb.device if alm_emb is not None else torch.device("cpu")
    if alm_emb is not None:
        emb_row = alm_emb.detach().reshape(1, -1).to(torch.float32)
        setp = SetProperty("alm_embedding", emb_row)
        graphs = [setp(g) for g in graphs]
    if task_direction is not None:
        # Scalar +/-1 task_direction cond_field, (1,) float32 to match training.
        setp_td = SetProperty("task_direction",
                              torch.tensor([float(task_direction)], dtype=torch.float32,
                                           device=_cond_device))
        graphs = [setp_td(g) for g in graphs]
    ds = ChemGraphlistDataset(list(graphs))
    return DataLoader(
        ds, batch_size=K, shuffle=False,
        collate_fn=partial(_collate_fn, collate_fn=collate),
    )


def apply_bridge_lora(alm, lora_dir: Path, device):
    """Apply the bridge LoRA on top of a Stage-2-merged ALM, then merge it (two-stage reload of a bridge run)."""
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    lora_dir = Path(lora_dir)
    with open(lora_dir / "adapter_config.json") as f:
        saved = json.load(f)
    cfg = LoraConfig(
        r=saved["r"], lora_alpha=saved["lora_alpha"], lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=saved.get("target_modules"),
    )
    alm.llm = get_peft_model(alm.llm, cfg)
    sd = load_file(str(lora_dir / "adapter_model.safetensors"))
    sd = {k.replace(".lora_A.weight", ".lora_A.default.weight")
           .replace(".lora_B.weight", ".lora_B.default.weight"): v
          for k, v in sd.items()}
    cur_sd = alm.llm.state_dict()
    for k in list(sd.keys()):
        if k in cur_sd and sd[k].shape != cur_sd[k].shape:
            old, cur = sd[k], cur_sd[k]
            if old.ndim == cur.ndim and all(o <= c for o, c in zip(old.shape, cur.shape)):
                new = cur.clone()
                new[tuple(slice(0, s) for s in old.shape)] = old.to(new.dtype)
                sd[k] = new
    alm.llm.load_state_dict(sd, strict=False)
    alm.llm = alm.llm.merge_and_unload()
    alm.llm = alm.llm.to(device)
    # Restore trained [atoms_i] rows if --unfreeze_atoms_i_embeds persisted them; else no-op.
    _pstate = Path(lora_dir).parent / "projector_and_state.pt"
    _restored_atoms_rows = False
    if _pstate.exists():
        _blob = torch.load(_pstate, map_location="cpu", weights_only=False)
        if "atoms_i_embed_rows" in _blob and "output_atom_token_ids" in _blob:
            _emb = alm.llm.get_input_embeddings().weight
            _ids = list(_blob["output_atom_token_ids"])
            with torch.no_grad():
                _emb[_ids] = _blob["atoms_i_embed_rows"].to(_emb.dtype).to(_emb.device)
            _restored_atoms_rows = True
            print(f"  [bridge-lora] restored {len(_ids)} [atoms_i] embedding rows from {_pstate.name}")
    row_note = "trained atoms_i rows restored" if _restored_atoms_rows else "atoms_i rows = Stage-2 init"
    print(f"  [bridge-lora] applied r={saved['r']} adapter from {lora_dir} "
          f"({row_note})", flush=True)


def _raw_alm_embedding_dim(atoms_mapper_path: Path, K: int, hidden_dim: int) -> int:
    """Flattened ALM embedding dim the saved bridge expects (matches get_alm_embedding's output).

    Q-Former bridges read (input atoms + N context + K [atoms_i]) hidden states; pool reads K.
    """
    ckpt = torch.load(atoms_mapper_path, map_location="cpu", weights_only=False)
    bridge_kind = canonical_bridge_kind(ckpt.get("bridge_kind", "pool"))
    if bridge_kind in ("producer-consumer", "producer-consumer-pool"):
        n_src = (int(ckpt.get("qformer_input_atoms", 0))
                 + int(ckpt.get("qformer_context_tokens", 128)) + int(K))
        return n_src * int(hidden_dim)
    return int(K) * int(hidden_dim)


def build_csp_sampler(pl_module, guidance_scale: float, diffusion_steps: int | None = None):
    """Compose the CSP sampling config and bind pl_module; guidance_scale = CFG strength on the bridge."""
    import hydra
    from mattergen.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH

    overrides = [f"sampler_partial.guidance_scale={guidance_scale}"]
    if diffusion_steps is not None:
        overrides.append(f"sampler_partial.N={int(diffusion_steps)}")
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        os.path.abspath(str(DEFAULT_SAMPLING_CONFIG_PATH)), version_base="1.1"
    ):
        cfg = hydra.compose(config_name="csp", overrides=overrides)
    sampler_partial = hydra.utils.instantiate(cfg.sampler_partial)
    return sampler_partial(pl_module=pl_module)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alm_checkpoint", type=Path,
                    default=Path(os.path.join(CHECKPOINTS, "alm-core")))
    ap.add_argument("--atoms_mapper", type=Path, required=True,
                    help="atoms_mapper.pt from a Stage-3 step=N dir (bridge and fine-tuned decoder weights).")
    ap.add_argument("--mattergen_model_path", type=str,
                    default=os.path.join(RUNS, "csp_backbone"),
                    help="CSP-mode MatterGen model dir containing config.yaml and checkpoints/.")
    ap.add_argument("--bridge_lora_dir", type=Path, default=None,
                    help="Bridge LoRA dir applied on top of the Stage-2-merged ALM (default: <atoms_mapper parent>/lora_adapter; 'none' skips it).")
    ap.add_argument("--max_rows", type=int, default=1000)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--guidance_factor", type=float, default=0.0,
                    help="CFG guidance scale on the alm_embedding bridge (0 = pure conditional).")
    ap.add_argument("--diffusion_steps", type=int, default=None)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--planner_alm_checkpoint", type=Path, default=None,
                    help="Optional separate checkpoint used only for composition planning (default: the bridge ALM).")
    ap.add_argument("--composition_source", choices=["planner", "teacher"], default=None,
                    help="planner: the LLM predicts atom counts; teacher: use ground-truth counts (upper bound). Default: planner.")
    ap.add_argument("--use_oracle_comp", action="store_true",
                    help="Same as --composition_source teacher.")
    ap.add_argument("--bridge_off", action="store_true",
                    help="Stamp a zero alm_embedding (ablation: planner composition only).")
    ap.add_argument("--benchmark", default="mp_20", choices=list(epc.BENCHMARK_CSV.keys()))
    ap.add_argument("--prompt_version", default="v5",
                    choices=["v1", "v2", "v3", "v4", "v5"])
    ap.add_argument("--write_gens", action="store_true",
                    help="Persist generated structures to disk (default: match in memory only).")
    args = ap.parse_args()
    if args.composition_source is None:
        args.composition_source = "teacher" if args.use_oracle_comp else "planner"
    elif args.use_oracle_comp and args.composition_source != "teacher":
        ap.error("--use_oracle_comp means --composition_source teacher; "
                 "do not combine it with --composition_source planner.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bridge-csp] writing to {args.out_dir} "
          f"(composition_source={args.composition_source})", flush=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_targets = epc.load_targets(args.benchmark)[: args.max_rows]
    src = f"{args.benchmark} CSV"
    if args.num_shards > 1:
        targets = [t for i, t in enumerate(all_targets)
                   if (i % args.num_shards) == args.shard_idx]
        print(f"  shard {args.shard_idx}/{args.num_shards}: {len(targets)}/{len(all_targets)} rows from {src}",
              flush=True)
    else:
        targets = all_targets
        print(f"  {len(targets)} rows from {src}", flush=True)

    print("  loading ALM and bridged CSP decoder ...", flush=True)
    _ckdir = Path(args.atoms_mapper).parent
    _is_full_ft = (_ckdir / "llm_full_ft" / "qwen3_state_dict.pt").exists()
    _alm_ckpt = str(_ckdir) if _is_full_ft else str(args.alm_checkpoint)
    if _is_full_ft:
        print(f"  full fine-tuned checkpoint: loading Qwen3 from {_ckdir}/llm_full_ft "
              f"(LoRA overlay skipped)", flush=True)
    alm, tok, pl_module, K = load_alm_and_pl_module(
        alm_checkpoint=_alm_ckpt,
        atoms_mapper=str(args.atoms_mapper),
        use_cached_embeddings=True,   # text-only planner prompts, skip OrbV3
        device=device,
        model_path=args.mattergen_model_path,
    )
    # Two-stage load: bridge LoRA on top of the Stage-2-merged ALM (else [atoms_i] uses the wrong LoRA).
    _bld = args.bridge_lora_dir
    if _bld is None:
        _bld = Path(args.atoms_mapper).parent / "lora_adapter"
    if _is_full_ft:
        print("  [bridge-lora] skipped: full fine-tuned LLM checkpoint", flush=True)
    elif str(_bld).lower() != "none":
        if not Path(_bld).exists():
            raise FileNotFoundError(f"bridge_lora_dir not found: {_bld}")
        apply_bridge_lora(alm, _bld, device)
    else:
        print("  [bridge-lora] skipped (--bridge_lora_dir none): using the Stage-2 LoRA only", flush=True)
    alm.eval()

    # Optional separate planner model; the bridge ALM still supplies alm_embedding.
    planner_alm, planner_tok = alm, tok
    if args.planner_alm_checkpoint is not None and args.composition_source == "planner":
        from loader import load_alm as _load_alm
        print(f"  loading separate planner from {args.planner_alm_checkpoint} ...", flush=True)
        planner_alm, planner_tok = _load_alm(
            checkpoint=str(args.planner_alm_checkpoint),
            use_cached_embeddings=True, merge_lora=True, is_trainable=False, device=device)
        planner_alm.eval()
        print(f"  planner loaded (t={time.time()-t0:.0f}s)", flush=True)

    pl_module.eval()
    cond_fields = pl_module.diffusion_module.model.cond_fields_model_was_trained_on
    has_alm = "alm_embedding" in cond_fields
    bridge_off_dim = _raw_alm_embedding_dim(args.atoms_mapper, K, alm.llm_hidden_dim)
    print(f"  decoder cond_fields={cond_fields} has_alm_embedding={has_alm} "
          f"(t={time.time()-t0:.0f}s)", flush=True)
    if not has_alm:
        print("  warning: decoder has no alm_embedding cond field, so the bridge has no effect", flush=True)

    from omegaconf import OmegaConf
    from mattergen.generator import draw_samples_from_sampler

    sampler = build_csp_sampler(pl_module, args.guidance_factor, args.diffusion_steps)
    matcher = StructureMatcher(**epc.TOL)

    results = []
    K_eff = args.K

    for i, (mp_id, target) in enumerate(targets):
        formula = str(target.composition.reduced_formula)
        gt_counts = dict(Counter([str(s.specie.symbol) for s in target]))
        n_atoms_target = sum(gt_counts.values())
        prompt = epc.planner_prompt(formula, n_atoms_target)

        plan_text = None
        if args.composition_source == "teacher":
            target_comp = gt_counts
            planner_correct = True
        else:
            plan_text, parsed = epc.llm_plan(prompt, planner_alm, planner_tok,
                                             prompt_version=args.prompt_version)
            target_comp, _fu = epc.comp_from_plan(parsed, args.prompt_version,
                                                  target_atoms=n_atoms_target)
            if target_comp is None:
                results.append({"row_id": mp_id, "formula": formula,
                                "matched_n1": False, "matched_nK": False,
                                "first_match_idx": -1, "n_gen": 0,
                                "planner_correct": False, "planner_parse_fail": True,
                                "planner_text": (plan_text or "")[:300]})
                continue
            planner_correct = (target_comp == gt_counts)

        # Bridge vector: per-prompt [atoms_i] hidden states (zeros if ablated/inert).
        alm_emb = None
        if has_alm and not args.bridge_off:
            alm_emb = get_alm_embedding(alm, tok, prompt, device,
                                        json_counts=target_comp)
        elif has_alm and args.bridge_off:
            alm_emb = torch.zeros(bridge_off_dim, device=device)

        if (i % 5) == 0:
            extra = "" if args.composition_source == "teacher" else f"  plan={target_comp} correct={planner_correct}"
            print(f"  [{i:3d}/{len(targets)}] {mp_id} {formula} (gt={gt_counts}){extra} "
                  f"t={time.time()-t0:.0f}s", flush=True)

        try:
            loader = build_csp_condition_loader(target_comp, K_eff, alm_emb)
            out_path = (args.out_dir / "gens" / mp_id) if args.write_gens else None
            if out_path is not None:
                out_path.mkdir(parents=True, exist_ok=True)
            samples = draw_samples_from_sampler(
                sampler=sampler,
                condition_loader=loader,
                properties_to_condition_on=None,   # alm_embedding is stamped on chemgraphs
                output_path=out_path,
                cfg=OmegaConf.create({}) if out_path is not None else None,
                record_trajectories=False,
            )
        except Exception as e:
            print(f"    [{mp_id}] gen failed: {e}", flush=True)
            results.append({"row_id": mp_id, "formula": formula,
                            "matched_n1": False, "matched_nK": False,
                            "first_match_idx": -1, "n_gen": 0,
                            "planner_correct": planner_correct,
                            "planner_parse_fail": False, "gen_error": str(e)[:200]})
            continue

        matched_n1 = matched_nK = False
        first_match_idx = -1
        for j, s in enumerate(samples):
            if not isinstance(s, Structure):
                try:
                    s = AseAtomsAdaptor.get_structure(s)
                except Exception:
                    continue
            try:
                if matcher.fit(target, s):
                    matched_nK = True
                    if j == 0:
                        matched_n1 = True
                    if first_match_idx < 0:
                        first_match_idx = j
            except Exception:
                pass
        results.append({"row_id": mp_id, "formula": formula,
                        "matched_n1": matched_n1, "matched_nK": matched_nK,
                        "first_match_idx": first_match_idx, "n_gen": len(samples),
                        "planner_correct": planner_correct, "planner_parse_fail": False,
                        "gt_counts": gt_counts, "planner_counts": target_comp,
                        "planner_text": (plan_text or "")[:300]})

    # Rates divide by every targeted row: planner parse failures and generation errors count as misses.
    summary = epc.summarize_csp_rows(results)
    n = summary["n_rows"]
    headline = {
        "n_rows": n, "n_scored": summary["n_scored"], "K": args.K,
        "guidance_factor": args.guidance_factor,
        "match_rate_n1": summary["match_rate_n1"],
        "match_rate_nK": summary["match_rate_nK"],
        "planner_correct_rate": (summary["planner_correct_rate"]
                                 if args.composition_source == "planner" else 1.0),
        "planner_parse_fail": summary["planner_parse_fail"],
        "n_gen_failed": summary["n_gen_failed"],
        "gen_failed_rate": summary["gen_failed_rate"],
        "atoms_mapper": str(args.atoms_mapper),
        "mattergen_model_path": args.mattergen_model_path,
        "has_alm_embedding_cond": has_alm,
        "bridge_off": args.bridge_off,
        "use_oracle_comp": args.composition_source == "teacher",
        "composition_source": args.composition_source,
        "benchmark": args.benchmark,
        "prompt_version": args.prompt_version,
        "note": "planner composition + ALM bridge embedding, CSP-mode MatterGen decoder",
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(headline, indent=2))
    with (args.out_dir / "predictions.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n[bridge-csp] {args.benchmark} ({n} rows, {summary['n_scored']} scored, "
          f"K={args.K}, g={args.guidance_factor}):")
    print(f"  M@1 = {headline['match_rate_n1']:.4f}  ({summary['n_match_n1']}/{n})")
    print(f"  M@K = {headline['match_rate_nK']:.4f}  ({summary['n_match_nK']}/{n})")
    print(f"  gen_failed = {summary['n_gen_failed']}")
    if args.composition_source == "planner":
        print(f"  planner_correct = {headline['planner_correct_rate']:.4f}  "
              f"parse_fail={summary['planner_parse_fail']}")
    print(f"  total time = {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
