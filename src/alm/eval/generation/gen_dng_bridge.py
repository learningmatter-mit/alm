"""DNG-style generation: text prompt -> bridged csp_backbone CSP-mode decoder -> flat CIFs for score_dng_hull.py."""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ALM_ROOT)

from pymatgen.core import Structure  # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402

from eval_dng import _sample_prompts_from_parquet  # noqa: E402
from eval_bridge_csp import (  # noqa: E402
    apply_bridge_lora,
    build_csp_condition_loader,
    build_csp_sampler,
)
from generate_stage3 import load_alm_and_pl_module, get_alm_embedding  # noqa: E402
from paths import DATA_ROOT, CHECKPOINTS, RUNS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--alm_checkpoint", type=Path,
                    default=Path(os.path.join(CHECKPOINTS, "alm_checkpoints/stage2_checkpoints/step=12000")),
                    help="Stage-2 base ckpt (the bridge LoRA is overlaid on the merged base).")
    ap.add_argument("--atoms_mapper", type=Path, required=True,
                    help="<bridge variant>/step=N/atoms_mapper.pt (bridge + cond layers).")
    ap.add_argument("--mattergen_model_path", type=str,
                    default=os.path.join(RUNS, "csp_backbone"),
                    help="LOCAL csp_backbone CSP-mode backbone dir (config.yaml + checkpoints/). "
                         "MUST match the baseline's backbone so the comparison is matched.")
    ap.add_argument("--bridge_lora_dir", type=Path, default=None,
                    help="Bridge LoRA dir (default: <atoms_mapper parent>/lora_adapter). "
                         "'none' to skip (Stage-2 LoRA only).")
    ap.add_argument("--pairs_parquet", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs.parquet")),
                    help="pairs.parquet for prompt sampling (same source as the baseline).")
    ap.add_argument("--n_prompts", type=int, default=1000,
                    help="N prompts (match the baseline).")
    ap.add_argument("--prompts_seed", type=int, default=42,
                    help="Prompt RNG seed (match the baseline).")
    ap.add_argument("--max_n_atoms", type=int, default=30,
                    help="Drop comps with >max_n_atoms (csp_backbone distrib; match the baseline).")
    ap.add_argument("--min_n_atoms", type=int, default=3,
                    help="Floor on the CSP cell atom count. Small cells (esp. heavy-element) go "
                         "zero-edge mid-diffusion and crash on GemNet's CPU zero-tensor; matches "
                         "build_sampler_and_loader's min_n_atoms. Planner formulas are scaled up "
                         "toward max_n_atoms; comps that can't reach this floor are dropped.")
    ap.add_argument("--K", type=int, default=1,
                    help="Generations per prompt. DNG convention K=1 (matches baseline).")
    ap.add_argument("--guidance_factor", type=float, default=1.0,
                    help="CFG guidance scale on the alm_embedding bridge "
                         "(0 = pure conditional, bridge ON; for bridge-OFF use --bridge_off).")
    ap.add_argument("--diffusion_steps", type=int, default=None)
    # FK SMC resamples among the K polymorphs of one composition (composition preserved).
    ap.add_argument("--fk_rewards", type=str, default="",
                    help="FK reward spec, e.g. 'mattersim_energy:1.0'. Empty = no FK (default).")
    ap.add_argument("--fk_direction", type=str, default="lower",
                    choices=["lower", "higher"],
                    help="Energy-reward direction: 'lower'=stability (quality steering).")
    ap.add_argument("--fk_resample_every", type=int, default=5)
    ap.add_argument("--fk_t_start_frac", type=float, default=0.5,
                    help="Apply FK only after t < T·(1−frac) (low-noise half, where the "
                         "Tweedie x̂₀ energy is meaningful). 0.5 = second half.")
    ap.add_argument("--fk_lambda", type=float, default=1.0)
    ap.add_argument("--fk_potential", type=str, default="diff",
                    choices=["diff", "sum", "max"])
    ap.add_argument("--fk_ess_threshold_frac", type=float, default=0.5)
    ap.add_argument("--fk_keep_top_k", type=int, default=-1)
    ap.add_argument("--fk_log_w_clip", type=float, default=10.0)
    ap.add_argument("--bridge_off", action="store_true",
                    help="Stamp a zero alm_embedding (ablation: JSON->csp_backbone == baseline path).")
    ap.add_argument("--composition_source", choices=["teacher", "planner"], default="teacher",
                    help="teacher (default) = use the prompt's deduped GT element set (== baseline). "
                         "planner = have the ALM emit a JSON composition from the prompt text.")
    ap.add_argument("--prompt_version", default="v5",
                    choices=["v1", "v2", "v3", "v4", "v5"],
                    help="Planner prompt template (only used when composition_source=planner).")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cif_root = args.out_dir / "cifs"
    cif_root.mkdir(parents=True, exist_ok=True)
    print(f"[bridge-dng] writing -> {args.out_dir} "
          f"(composition_source={args.composition_source}, g={args.guidance_factor}, "
          f"bridge_off={args.bridge_off})", flush=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompts, prompt_ids, elements_per_prompt, _json_counts = _sample_prompts_from_parquet(
        args.pairs_parquet,
        n_prompts=args.n_prompts,
        seed=args.prompts_seed,
        parent_filter=None,
    )
    # Filter on REAL atom count (full multiset); deduped count-1 cells go zero-edge mid-diffusion and crash.
    kept = [(p, pid, els, jc) for p, pid, els, jc in
            zip(prompts, prompt_ids, elements_per_prompt, _json_counts)
            if args.min_n_atoms <= sum(jc.values()) <= args.max_n_atoms]
    print(f"  sampled {len(prompts)} prompts, {len(kept)} after "
          f"{args.min_n_atoms}<=N<={args.max_n_atoms} real-atom-count filter (full multiset)",
          flush=True)
    if args.num_shards > 1:
        kept = [k for i, k in enumerate(kept) if (i % args.num_shards) == args.shard_idx]
        print(f"  shard {args.shard_idx}/{args.num_shards}: {len(kept)} rows", flush=True)
    if not kept:
        print("[bridge-dng] no rows to process; exiting", flush=True)
        return 0

    print(f"  loading ALM + bridged csp_backbone decoder ...", flush=True)
    _ckdir = Path(args.atoms_mapper).parent
    _is_full_ft = (_ckdir / "llm_full_ft" / "qwen3_state_dict.pt").exists()
    _alm_ckpt = str(_ckdir) if _is_full_ft else str(args.alm_checkpoint)
    if _is_full_ft:
        print(f"  full-FT checkpoint detected -> loading full Qwen3 from "
              f"{_ckdir}/llm_full_ft (LoRA overlay skipped)", flush=True)
    alm, tok, pl_module, K = load_alm_and_pl_module(
        alm_checkpoint=_alm_ckpt,
        atoms_mapper=str(args.atoms_mapper),
        use_cached_embeddings=True,   # text-only prompts; skip OrbV3
        device=device,
        model_path=args.mattergen_model_path,
    )
    # Two-stage load: overlay the bridge variant's fresh LoRA on the Stage-2-merged ALM.
    _bld = args.bridge_lora_dir
    if _bld is None:
        _bld = Path(args.atoms_mapper).parent / "lora_adapter"
    if _is_full_ft:
        print("  [bridge-lora] SKIPPED (full-FT — full Qwen3 weights loaded directly)", flush=True)
    elif str(_bld).lower() != "none":
        if not Path(_bld).exists():
            raise FileNotFoundError(f"bridge_lora_dir not found: {_bld}")
        apply_bridge_lora(alm, _bld, device)
    else:
        print("  [bridge-lora] SKIPPED (--bridge_lora_dir none): using Stage-2 LoRA only", flush=True)
    alm.eval()
    pl_module.eval()

    cond_fields = pl_module.diffusion_module.model.cond_fields_model_was_trained_on
    has_alm = "alm_embedding" in cond_fields
    print(f"  decoder cond_fields={cond_fields} has_alm_embedding={has_alm} "
          f"(t={time.time()-t0:.0f}s)", flush=True)
    if not has_alm:
        print("  [WARN] decoder has no alm_embedding cond_field — bridge inert; "
              "this measures JSON->csp_backbone only.", flush=True)
    # bridge-off zero vector dim = raw flattened K*hidden (pool bridge).
    from eval_bridge_csp import _raw_alm_embedding_dim  # noqa: E402
    bridge_off_dim = _raw_alm_embedding_dim(args.atoms_mapper, K, alm.llm_hidden_dim)

    if args.composition_source == "planner":
        import eval_planner_csp as epc  # noqa: E402

    from mattergen.generator import draw_samples_from_sampler  # noqa: E402

    sampler = build_csp_sampler(pl_module, args.guidance_factor, args.diffusion_steps)

    if args.fk_rewards:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
        from fk_rewards import parse_rewards as _fk_parse_rewards  # noqa: E402
        from generate_stage3 import (  # noqa: E402
            _ensure_fk_hook_installed, _install_fk_on_sampler,
        )
        reward = _fk_parse_rewards(args.fk_rewards, direction=args.fk_direction)
        _ensure_fk_hook_installed(pl_module)
        st = pl_module._fk_state
        st.enabled = True
        st.reward = reward
        st.target_counts = None
        st.enforce_target_counts = False
        st.n_particles = args.K            # informational; fk_denoise reads batch size live
        st.resample_every = args.fk_resample_every
        st.t_start_frac = args.fk_t_start_frac
        st.lambda_ = args.fk_lambda
        st.potential = args.fk_potential
        st.ess_threshold_frac = args.fk_ess_threshold_frac
        st.keep_top_k = args.fk_keep_top_k
        st.log_w_clip = args.fk_log_w_clip
        st.stratify_resample_by_n_atoms = False
        _install_fk_on_sampler(sampler, st)
        print(f"[fk] DNG-quality steering ON: rewards={args.fk_rewards} dir={args.fk_direction} "
              f"N(K)={args.K} resample_every={st.resample_every} t_start_frac={st.t_start_frac} "
              f"lambda={st.lambda_} potential={st.potential} ess_thr={st.ess_threshold_frac}",
              flush=True)

    records = []
    n_saved = n_planner_parse_fail = 0
    n_size_reject = 0
    for i, (prompt, pid, elems, jc) in enumerate(kept):
        if i % 25 == 0:
            print(f"  [{i}/{len(kept)}] (t={time.time()-t0:.0f}s) ...", flush=True)

        if args.composition_source == "teacher":
            # GT-composition ceiling arm (full multiset); leaks composition, not the headline.
            target_comp = dict(jc)
            json_counts_for_bridge = target_comp
        else:
            # Planner (headline, de-novo): LLM proposes a formula, scaled to [min,max] n_atoms.
            _txt, parsed = epc.llm_plan(prompt, alm, tok,
                                        prompt_version=args.prompt_version)
            pfu, _fu = epc.comp_from_plan(parsed, args.prompt_version, target_atoms=None)
            if pfu is None or sum(int(v) for v in pfu.values()) <= 0:
                n_planner_parse_fail += 1
                records.append({"prompt_id": pid, "planner_parse_fail": True})
                continue
            s = sum(int(v) for v in pfu.values())
            # Smallest Z reaching the min_n_atoms floor; supercells are off-distribution.
            Z = max(1, -(-args.min_n_atoms // s))        # ceil(min_n_atoms / s)
            target_comp = {el: int(n) * Z for el, n in pfu.items()}
            n_at = sum(target_comp.values())
            if n_at < args.min_n_atoms or n_at > args.max_n_atoms:
                n_size_reject += 1
                records.append({"prompt_id": pid, "planner_size_reject": True,
                                "n_atoms": n_at, "formula_unit": pfu})
                continue
            json_counts_for_bridge = target_comp

        alm_emb = None
        if has_alm and not args.bridge_off:
            # Prompt is already framed; wrap_user_template=False avoids double-wrapping off-distribution.
            alm_emb = get_alm_embedding(alm, tok, prompt, device,
                                        json_counts=json_counts_for_bridge,
                                        wrap_user_template=False)
        elif has_alm and args.bridge_off:
            alm_emb = torch.zeros(bridge_off_dim, device=device)

        try:
            loader = build_csp_condition_loader(target_comp, args.K, alm_emb)
            # In-memory only; we write our own flat CIFs (MatterGen's save_structures
            # uses a shared /tmp path that concurrent shards would clobber).
            structures = draw_samples_from_sampler(
                sampler=sampler,
                condition_loader=loader,
                properties_to_condition_on=None,   # alm_embedding stamped on chemgraphs
                output_path=None,
                cfg=None,
                record_trajectories=False,
            )
        except Exception as e:
            records.append({"prompt_id": pid, "err": str(e)[:160]})
            continue

        k_saved = 0
        for k, s in enumerate(structures[:args.K]):
            if not isinstance(s, Structure):
                try:
                    s = AseAtomsAdaptor.get_structure(s)
                except Exception:
                    continue
            try:
                p = cif_root / f"{pid}__k{k}.cif"
                p.write_text(s.to(fmt="cif"))
                k_saved += 1
            except Exception:
                pass
        n_saved += k_saved
        records.append({
            "prompt_id": pid,
            "n_atoms_target": sum(target_comp.values()),
            "formula_target": "".join(f"{el}{n}" for el, n in sorted(target_comp.items())),
            "prompt": prompt[:200],
            "k_saved": k_saved,
            "composition_source": args.composition_source,
            "bridge_off": args.bridge_off,
            "guidance_factor": args.guidance_factor,
        })

    (args.out_dir / f"records_shard{args.shard_idx}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records))
    print(f"\n[bridge-dng] shard{args.shard_idx} done: {len(records)} rows, "
          f"{n_saved} CIFs saved in {cif_root} "
          f"(planner_parse_fail={n_planner_parse_fail}, t={time.time()-t0:.0f}s)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
