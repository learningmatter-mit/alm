#!/usr/bin/env python
"""Per-task instruction-following editing eval for the planner-stage bridge checkpoint (polymorph/doping/app/atomtxt/strain/describe/ood). App task needs OPENAI_API_KEY."""
from __future__ import annotations


import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Pauling electronegativity.*")
warnings.filterwarnings("ignore", message=".*fractional coordinates.*")

import numpy as np
import pyarrow.parquet as pq
import torch

_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher

import eval_planner_csp as epc
from eval_bridge_csp import (
    apply_bridge_lora,
    build_csp_condition_loader,
    build_csp_sampler,
)
from generate_stage3 import (load_alm_and_pl_module, get_alm_embedding,
                              _install_sega_on_sampler)

# SEGA paired direction phrases per property: (higher_phrase, lower_phrase), differing only in direction.
_SEGA_PHRASE = {
    "formation_energy": ("with higher formation energy per atom (less thermodynamically stable)",
                         "with lower formation energy per atom (more thermodynamically stable)"),
    "density":          ("that is denser, with a higher mass density",
                         "that is less dense, with a lower mass density"),
    "band_gap":         ("with a larger electronic band gap",
                         "with a smaller electronic band gap"),
}


def _sega_prompts(prop: str, direction: int) -> tuple[str, str]:
    """(asked_prompt, opposite_prompt) for SEGA; direction +1=higher / -1=lower."""
    hi, lo = _SEGA_PHRASE.get(prop, _SEGA_PHRASE["formation_energy"])
    asked_phrase = hi if direction > 0 else lo
    opp_phrase = lo if direction > 0 else hi
    base = "<atoms>\nGenerate a variant of this material with the same composition {p}."
    return base.format(p=asked_phrase), base.format(p=opp_phrase)

from structure_metrics import validity_geom, validity_charge
from paths import DATA_ROOT, CHECKPOINTS, RUNS


DEFAULT_PARQUET = {
    "polymorph": os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_polymorph_under_hull.parquet"),
    "doping": os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_doping_strain_sub1M.parquet"),
    # strain shares the doping parquet; the dV_±X.XXpct target is in the row_id.
    "strain": os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_doping_strain_sub1M.parquet"),
    "app": os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_app.parquet"),
    "atomtxt": os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_atomtxt.parquet"),
    # describe/ood: text->structure recovery (no input); describe verbose, ood terse, same materials.
    "describe": os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs.parquet"),
    "ood": os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_ood.parquet"),
}

# dV_±X.XXpct target volume-change (strain task), encoded in the row_id.
_DV_RE = re.compile(r"dV_([+-]?[0-9.]+)pct")


def _parse_target_dv_pct(row_id: str):
    m = _DV_RE.search(str(row_id))
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

# atomtxt directional planner: text-only substitution prompt giving the input composition.
_DOPING_PLANNER_TMPL = (
    "A crystal cell has the composition {counts} (element: number of atoms in the cell). "
    "Apply this edit to it: {instruction} "
    "Output ONLY the resulting per-cell composition as JSON: {{\"counts\": {{\"El\": n, ...}}}}."
)


# ── X→Y substitution parsing ─────────────────────────────────────────────────
_PATTERNS = [
    r"replacing all ([A-Z][a-z]?) atoms with ([A-Z][a-z]?)",
    r"substitute ([A-Z][a-z]?) sites with ([A-Z][a-z]?)",
    r"Replace the ([A-Z][a-z]?) sublattice with ([A-Z][a-z]?)",
    r"Apply a ([A-Z][a-z]?)->\s*([A-Z][a-z]?) substitution",
    r"replacing all ([A-Z][a-z]?) sites with ([A-Z][a-z]?)",
]
_SUB_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))


def _parse_substitution(prompt: str):
    m = _SUB_RE.search(prompt)
    if not m:
        return None
    groups = m.groups()
    for i in range(0, len(groups), 2):
        if groups[i] is not None and groups[i + 1] is not None:
            return groups[i], groups[i + 1]
    return None


# ── inline atoms_struct → ASE / pymatgen ─────────────────────────────────────
def _ase_atoms_from_struct(atoms_struct: dict) -> Atoms:
    elements = [str(e) for e in atoms_struct["elements"]]
    coords = np.asarray(atoms_struct["coords"], dtype=np.float64)
    lattice = np.asarray(atoms_struct["lattice_mat"], dtype=np.float64)
    cartesian = bool(atoms_struct.get("cartesian", True))
    if cartesian:
        return Atoms(symbols=elements, positions=coords, cell=lattice, pbc=True)
    return Atoms(symbols=elements, scaled_positions=coords, cell=lattice, pbc=True)


def _ase_to_struct(atoms_struct: dict):
    try:
        elements = [str(e).strip() for e in atoms_struct["elements"]]
        coords = np.asarray(atoms_struct["coords"], dtype=np.float64)
        lattice = np.asarray(atoms_struct["lattice_mat"], dtype=np.float64)
        cartesian = bool(atoms_struct.get("cartesian", True))
        return Structure(lattice=lattice, species=elements, coords=coords,
                         coords_are_cartesian=cartesian)
    except Exception:
        return None


def _live_orbv3_features(alm, atoms_obj, device) -> torch.Tensor:
    """Raw (N, 256) pre-projector OrbV3 node features (what get_alm_embedding expects)."""
    from orb_models.forcefield import atomic_system
    from orb_models.forcefield.base import batch_graphs
    with torch.no_grad():
        atoms_capped = atoms_obj if alm.max_atoms is None else atoms_obj[:alm.max_atoms]
        graph = batch_graphs([atomic_system.ase_atoms_to_atom_graphs(
            atoms_capped, alm.atomistic_model.system_config, device=device,
        )])
        results = alm.atomistic_model.model(graph)
        return results["node_features"]  # (N_atoms, 256)


def _structurally_valid(struct: Structure, min_dist: float = 0.5) -> bool:
    try:
        if len(struct) < 1:
            return False
        d = struct.distance_matrix
        n = d.shape[0]
        d = d + np.eye(n) * 1e6
        return float(d.min()) > min_dist
    except Exception:
        return False


def _per_cell_counts(struct: Structure) -> dict:
    # Disordered sites (only from external CIFs) raise on .specie; fall back to dominant element.
    def _elem(s):
        if getattr(s, "is_ordered", True):
            return str(s.specie.symbol)
        return str(max(s.species, key=s.species.get).symbol)
    return dict(Counter(_elem(s) for s in struct))


# ── row selection (hash-deterministic, sharded) ─────────────────────────────
def _select_rows(parquet_path: Path, task: str, max_rows: int, seed: int,
                 num_shards: int, shard_idx: int, row_start: int = 0,
                 atomtxt_property: str | None = None) -> list[dict]:
    """Hash-deterministic eval subset; sharding applied after the top-max_rows cut (row_idx % num_shards)."""
    pf = pq.ParquetFile(parquet_path)
    need_input = task in ("polymorph", "doping", "atomtxt", "strain")
    cols = ["row_id", "parent", "source_idx", "user_prompt"]
    if need_input:
        cols += ["input_atoms_struct", "input_source_idx"]
    if task in ("describe", "ood"):
        cols += ["atoms_struct"]
    avail = set(pf.schema_arrow.names)
    cols = [c for c in cols if c in avail]
    candidates: list[dict] = []
    for batch in pf.iter_batches(batch_size=10000, columns=cols):
        for r in batch.to_pylist():
            if task in ("doping", "strain"):
                sub = _parse_substitution(r["user_prompt"])
                if sub is None:
                    continue
                r["_donor"], r["_dopant"] = sub
            if task == "strain":
                dv = _parse_target_dv_pct(r["row_id"])
                if dv is None:
                    continue
                r["_target_dv_pct"] = dv
            if task == "atomtxt":
                # row_id encodes GT: ...-{property}-(higher|lower); band_gap skipped (no predictor).
                m = re.search(
                    r"-(formation_energy|density|band_gap|volume)-(higher|lower|larger|smaller)$",
                    str(r["row_id"]))
                if m is None or m.group(1) == "band_gap":
                    continue
                r["_property"] = m.group(1)
                r["_direction"] = +1 if m.group(2) in ("higher", "larger") else -1
                if atomtxt_property and atomtxt_property != "all" \
                        and r["_property"] != atomtxt_property:
                    continue
            h = int(hashlib.md5(f"{r['row_id']}:{seed}".encode()).hexdigest(), 16)
            r["_h"] = h
            candidates.append(r)
    candidates.sort(key=lambda r: r["_h"])
    selected = candidates[row_start:row_start + max_rows]
    if num_shards > 1:
        selected = [r for i, r in enumerate(selected) if (i % num_shards) == shard_idx]
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True,
                    choices=["polymorph", "doping", "app", "atomtxt", "strain",
                             "describe", "ood"])
    ap.add_argument("--strain_tol_pct", type=float, default=5.0,
                    help="strain only: tolerance (in percentage points) on the relaxed "
                         "volume-change match. strain_correct = doping_correct AND "
                         "|relaxed dV%% - target dV%%| < strain_tol_pct.")
    ap.add_argument("--atomtxt_property", default="all",
                    choices=["all", "formation_energy", "density", "volume"],
                    help="atomtxt only: restrict the eval set to one property so each "
                         "property gets a clean fixed-N slice (band_gap is never scored "
                         "— no predictor). 'all' keeps the natural mixed distribution.")
    ap.add_argument("--doping_via_planner", action="store_true",
                    help="Doping: have the LLM planner emit the substituted composition "
                         "from text (input formula + edit), then CSP that composition "
                         "bridge-OFF. The honest planner→CSP path; bypasses the bridge. "
                         "correct_substitution then = planner emitted right comp AND CSP "
                         "produced a valid structure.")
    ap.add_argument("--alm_checkpoint", type=Path,
                    default=Path(os.path.join(CHECKPOINTS, "alm_checkpoints/stage2_checkpoints/step=12000")),
                    help="Stage-2 base ckpt (merged r=64 LoRA + projector). The fresh "
                         "r=8 bridge LoRA is applied on top from the ckpt's lora_adapter/.")
    ap.add_argument("--atoms_mapper", type=Path, required=True,
                    help="The bridge ckpt's step=N/atoms_mapper.pt. bridge_lora_dir "
                         "auto-derives to its sibling lora_adapter/.")
    ap.add_argument("--bridge_lora_dir", type=Path, default=None,
                    help="Override the fresh r=8 bridge LoRA dir "
                         "(default: <atoms_mapper parent>/lora_adapter). "
                         "Pass 'none' to skip (debug: Stage-2 LoRA only).")
    ap.add_argument("--mattergen_model_path", type=str,
                    default=os.path.join(RUNS, "csp_backbone"),
                    help="LOCAL csp_backbone CSP-mode backbone dir (config.yaml + checkpoints/).")
    ap.add_argument("--parquet", type=Path, default=None,
                    help="Task parquet (default: per-task under stage3_outputs/stage3a/).")
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--row_start", type=int, default=0,
                    help="Offset into the hash-sorted candidate pool; lets parallel jobs cover "
                         "disjoint row windows (e.g. multi-node FK gen) without duplication.")
    ap.add_argument("--K", type=int, default=8, help="Candidates per prompt.")
    ap.add_argument("--guidance_factor", type=float, default=0.0,
                    help="CFG guidance scale on the alm_embedding bridge "
                         "(0 = pure conditional; the bridge should matter for editing).")
    ap.add_argument("--diffusion_steps", type=int, default=None)
    ap.add_argument("--diffusion_seed", type=int, default=1337)
    ap.add_argument("--gen_retries", type=int, default=2,
                    help="Per-row seed-bump retries on a GemNet empty-graph "
                         "cuda/cpu device-mismatch. The empty-graph branch is in "
                         "external/ (untouchable here); a fresh diffusion seed "
                         "perturbs the trajectory and usually keeps the graph "
                         "non-empty, recovering the row. 0 = no retry (old behavior).")
    ap.add_argument("--strict_n_scored", action="store_true",
                    help="Hard-exit(2) when this shard scored 0 rows. Default OFF "
                         "for sharded runs: a shard that scored 0 writes "
                         "metrics.json (n_scored=0) and exits 0 so surviving shards "
                         "still aggregate; the run-level aggregator FATALs only when "
                         "EVERY shard scored 0. Turn ON for a single-shard/manual "
                         "run where 0 truly is total failure.")
    ap.add_argument("--handset_direction", action="store_true",
                    help="Thread each row's requested direction (_direction = +1 higher / "
                         "-1 lower) into get_alm_embedding so the 9th [atoms_i] hidden block "
                         "is overwritten with the same hand-set code used at training. "
                         "Without this the trained direction channel is inert at eval. "
                         "Use ONLY for --num_output_atom_tokens 9 handset checkpoints.")
    ap.add_argument("--scalar_direction", action="store_true",
                    help="Stamp the row's requested ±1 as the scalar task_direction "
                         "cond_field (SetProperty) at sampling, so CFG (guidance_factor) "
                         "steers it. Use ONLY for checkpoints trained with "
                         "--use_task_direction_cond.")
    # SEGA CFG (atomtxt): s = s_null + sega_g*(s_asked - s_opp); opp branch isolates the directional residual.
    ap.add_argument("--sega_difference", action="store_true",
                    help="atomtxt: enable prompt-difference (SEGA) CFG on the bridge.")
    ap.add_argument("--sega_g", type=float, default=3.0,
                    help="SEGA guidance scale on (s_asked - s_opp). Extrapolative; "
                         "3-7 typical. Replaces --guidance_factor when --sega_difference.")
    # Feynman-Kac directional steering (atomtxt): n_particles=K SMC ensemble, signed-property reward.
    ap.add_argument("--fk", action="store_true",
                    help="Enable FK directional steering (atomtxt only).")
    ap.add_argument("--fk_lambda", type=float, default=0.5)
    ap.add_argument("--fk_resample_every", type=int, default=10)
    ap.add_argument("--fk_t_start_frac", type=float, default=0.5,
                    help="Fraction of the schedule after which FK resampling starts "
                         "(0.5 = late half only — where x̂₀ energy is meaningful).")
    ap.add_argument("--fk_log_w_clip", type=float, default=50.0)
    ap.add_argument("--fk_potential", type=str, default="diff",
                    choices=["diff", "sum", "max"])
    ap.add_argument("--fk_ess_threshold_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0, help="Eval-set hash seed.")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    # StructureMatcher tolerances (CDVAE / CrystaLLM defaults), polymorph distinctness.
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle_tol", type=float, default=10.0)
    # MatterSim relax (polymorph energy + app characterization).
    ap.add_argument("--skip_relax", action="store_true",
                    help="Skip MatterSim relax. For polymorph this disables the PRIMARY "
                         "lower-energy metric; for app the judge sees raw-gen properties.")
    ap.add_argument("--single_point_energy", action="store_true",
                    help="atomtxt: score the formation_energy direction on the RAW generated "
                         "geometry via MatterSim single-point (max_n_steps=0), not the relaxed "
                         "energy. Sanity check on whether the bridge moves energy pre-relaxation.")
    ap.add_argument("--save_gens", type=Path, default=None,
                    help="atomtxt: persist every generated structure + (input, prompt, direction, "
                         "relaxed energies, direction_correct) to <save_gens>/gens_shard{idx}.parquet. "
                         "With --fk this builds the reward-distillation dataset "
                         "(filter to requested_direction=higher AND direction_correct).")
    ap.add_argument("--save_cifs", type=Path, default=None,
                    help="Showcase dump (all tasks): after scoring, for the first "
                         "--save_cifs_max rows whose per-task success flag is True, write "
                         "the INPUT structure (if any) + the first SUCCESSFUL generated "
                         "candidate as CIF into <save_cifs>/, plus one metadata JSON line per "
                         "saved example to <save_cifs>/saved_meta.jsonl (row_id, prompt, "
                         "target, why-success). Read-only on the eval itself.")
    ap.add_argument("--save_cifs_max", type=int, default=2,
                    help="Max number of successful examples to dump per task (default 2).")
    ap.add_argument("--mattersim_potential_path", type=str, default=None)
    # App-judge controls (mirror eval_app_consistency).
    ap.add_argument("--prompt_version", default="v2",
                    choices=["v1", "v2", "v3", "v4", "v5"],
                    help="Planner JSON format for the app composition (v2 = {'counts':{}} "
                         "= what these ckpts were trained on).")
    ap.add_argument("--judge_model", default="gpt-4o-mini")
    ap.add_argument("--judge_concurrency", type=int, default=16)
    # ── Direction-following inference experiments (bridge-only) ──
    ap.add_argument("--cot_tokens", type=int, default=0,
                    help="CoT-then-atoms: sample this many free-form LLM tokens before "
                         "the K=[atoms_i] block so the bridge reads reasoning context. "
                         "0 = off (deterministic).")
    ap.add_argument("--llm_temperature", type=float, default=1.0,
                    help="Sampling temperature for the CoT prefix (ignored if cot_tokens=0).")
    ap.add_argument("--cot_top_p", type=float, default=0.9,
                    help="Nucleus cutoff for the CoT prefix (ignored if cot_tokens=0).")
    ap.add_argument("--whiten_common_mode", action="store_true",
                    help="common-mode whitening (atomtxt): subtract the eval-set MEAN "
                         "bridge cond from every row before the consumer, leaving only the "
                         "directional residual. Two-pass.")
    ap.add_argument("--atoms_before_json", action="store_true",
                    help="For checkpoints whose model emits [atoms_i] BEFORE the {counts} "
                         "JSON: inference must build the assistant turn the same way, else "
                         "the bridge reads OOD positions.")
    args = ap.parse_args()

    if args.parquet is None:
        args.parquet = Path(DEFAULT_PARQUET[args.task])
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bridge-edit] task={args.task} writing → {args.out_dir}", flush=True)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Select rows (+ shard) ──
    rows = _select_rows(args.parquet, args.task, args.max_rows, args.seed,
                        args.num_shards, args.shard_idx, row_start=args.row_start,
                        atomtxt_property=args.atomtxt_property)
    print(f"[bridge-edit] {len(rows)} rows (task={args.task}, seed={args.seed}, "
          f"shard {args.shard_idx}/{args.num_shards}) from {args.parquet}", flush=True)
    if not rows:
        raise SystemExit(f"No rows selected for task={args.task} — check parquet/shard args.")

    # ── 2. Load ALM (planner + bridge) + bridged csp_backbone decoder ──
    # use_cached_embeddings=False: OrbV3 needed for live input-structure encoding (polymorph/doping).
    print(f"[bridge-edit] loading ALM + bridged csp_backbone decoder on {device} ...", flush=True)
    # Full-FT ckpts store the whole Qwen3 in llm_full_ft/ with no lora_adapter/ → load it directly, skip the overlay.
    _ckdir = Path(args.atoms_mapper).parent
    _is_full_ft = (_ckdir / "llm_full_ft" / "qwen3_state_dict.pt").exists()
    _alm_ckpt = str(_ckdir) if _is_full_ft else str(args.alm_checkpoint)
    if _is_full_ft:
        print(f"[bridge-edit] full-FT checkpoint detected → loading full Qwen3 from "
              f"{_ckdir}/llm_full_ft (LoRA overlay skipped)", flush=True)
    alm, tok, pl_module, K_tokens = load_alm_and_pl_module(
        alm_checkpoint=_alm_ckpt,
        atoms_mapper=str(args.atoms_mapper),
        use_cached_embeddings=False,
        device=device,
        model_path=args.mattergen_model_path,
    )
    # Two-stage load: fresh r=8 bridge LoRA on top of the Stage-2-merged ALM (load_alm alone uses the wrong LoRA).
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
        print("  [bridge-lora] SKIPPED (--bridge_lora_dir none): Stage-2 LoRA only", flush=True)
    alm.eval()
    pl_module.eval()

    cond_fields = pl_module.diffusion_module.model.cond_fields_model_was_trained_on
    has_alm = "alm_embedding" in cond_fields
    print(f"[bridge-edit] decoder cond_fields={cond_fields} has_alm_embedding={has_alm} "
          f"(t={time.time()-t0:.0f}s)", flush=True)
    if not has_alm:
        print("  [WARN] decoder has no alm_embedding cond_field — BRIDGE INERT. "
              "This measures observed-atom CSP only; the per-task edit signal is gone.",
              flush=True)

    from omegaconf import OmegaConf
    from mattergen.generator import draw_samples_from_sampler

    sampler = build_csp_sampler(pl_module, args.guidance_factor, args.diffusion_steps)

    # FK steering only where a diffused variable (pos+cell) maps to the gated metric:
    # atomtxt (signed property), polymorph (energy-lower), strain (volume target).
    # doping excluded: composition is observed, so a stoichiometry reward has nothing to steer.
    fk_enabled = bool(args.fk) and args.task in ("atomtxt", "polymorph", "strain")
    fk_reward_cache: dict = {}
    _fkst = None
    if fk_enabled:
        from generate_stage3 import (_ensure_fk_hook_installed, _install_fk_on_sampler,
                                       _fk_reset_per_prompt)
        from fk_rewards import parse_rewards as _parse_rewards
        _ensure_fk_hook_installed(pl_module)
        _fkst = pl_module._fk_state
        _fkst.enabled = True
        _fkst.n_particles = args.K
        _fkst.resample_every = args.fk_resample_every
        _fkst.t_start_frac = args.fk_t_start_frac
        _fkst.lambda_ = args.fk_lambda
        _fkst.potential = args.fk_potential
        _fkst.log_w_clip = args.fk_log_w_clip
        _fkst.ess_threshold_frac = args.fk_ess_threshold_frac
        _install_fk_on_sampler(sampler, _fkst)
        print(f"[bridge-edit:FK] installed N={args.K} λ={args.fk_lambda} "
              f"resample_every={args.fk_resample_every} t_start={args.fk_t_start_frac} "
              f"potential={args.fk_potential}", flush=True)

    def _fk_reward_for(prop, want):
        """Per-prompt signed-property reward (want +1=higher/-1=lower); cached per (property, direction)."""
        direction = "higher" if want > 0 else "lower"
        if prop == "polymorph":
            # Always lower-energy; `want` ignored (prompt is unconditionally lower-energy polymorph).
            spec = "mattersim_energy:1.0"
            rew_direction = "lower"
        elif prop == "formation_energy":
            spec = "mattersim_energy:1.0"
            rew_direction = direction
        elif prop == "volume":
            spec = "density_direction:1.0"
            # volume↑ ⇔ density↓ (and vice versa) at fixed composition.
            rew_direction = "lower" if want > 0 else "higher"
        else:  # density (and any other geometric axis): direction as requested.
            spec = "density_direction:1.0"
            rew_direction = direction
        key = (spec, rew_direction)
        if key not in fk_reward_cache:
            fk_reward_cache[key] = _parse_rewards(spec, direction=rew_direction)
        return fk_reward_cache[key]

    # ── 3. Generate per row (observed-atom CSP + stamped alm_embedding bridge) ──
    import random as _random
    gens_per_row: list[list] = []
    target_comp_per_row: list[dict | None] = []
    planner_text_per_row: list[str | None] = []
    n_planner_parse_fail = 0
    # n_gen_device_errors: rows that hit MatterGen's GemNet empty-graph cuda/cpu mismatch on every retry.
    n_gen_device_errors = 0
    n_gen_other_errors = 0
    last_gen_error = None

    def _is_device_mismatch(exc: Exception) -> bool:
        s = str(exc).lower()
        return ("on the same device" in s) or ("wrapper_cuda" in s) or \
               ("zero graph edges" in s) or ("expected all tensors to be on" in s)

    _cot_on = int(getattr(args, "cot_tokens", 0)) > 0

    def _cot_kw(i):
        # cot_seed = diffusion_seed + i so the whitening pre-pass and main loop draw the same CoT per row.
        return dict(cot_tokens=int(args.cot_tokens),
                    llm_temperature=float(args.llm_temperature),
                    cot_top_p=float(args.cot_top_p),
                    cot_seed=((int(args.diffusion_seed) + i) & 0x7FFFFFFF) if _cot_on else None)

    if _cot_on:
        print(f"[bridge-edit] CoT-then-atoms ON: cot_tokens={args.cot_tokens} "
              f"T={args.llm_temperature} top_p={args.cot_top_p}", flush=True)

    # precompute eval-set MEAN bridge cond (atomtxt) so the loop can subtract it (common-mode whitening).
    mean_emb = None
    if getattr(args, "whiten_common_mode", False) and args.task == "atomtxt" and has_alm:
        _embs = []
        for i, r in enumerate(rows):
            try:
                _ia = _ase_atoms_from_struct(r["input_atoms_struct"])
                _ae = _live_orbv3_features(alm, _ia, device)
                _is = _ase_to_struct(r["input_atoms_struct"])
                if _is is None:
                    continue
                _tc = _per_cell_counts(_is)
                _tdw = int(r.get("_direction", 0)) if args.handset_direction else None
                _e = get_alm_embedding(alm, tok, r["user_prompt"], device,
                                       atom_embed=_ae, wrap_user_template=False,
                                       json_counts=_tc, task_direction=_tdw, **_cot_kw(i))
                _embs.append(_e.detach().cpu())
            except Exception:
                continue
        if _embs:
            mean_emb = torch.stack(_embs, dim=0).mean(dim=0).to(device)
            print(f"[bridge-edit] common-mode whitening ON: subtracting mean cond over "
                  f"{len(_embs)}/{len(rows)} rows (||mean||={mean_emb.norm().item():.3f})", flush=True)
        else:
            print("[bridge-edit] common-mode whitening requested but 0 rows produced a cond "
                  "→ disabled (mean_emb=None)", flush=True)

    for i, r in enumerate(rows):
        _s = (int(args.diffusion_seed) + i) & 0x7FFFFFFF
        torch.manual_seed(_s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_s)
        np.random.seed(_s)
        _random.seed(_s)

        prompt = r["user_prompt"]
        plan_text = None

        # 3a. Observed composition (CSP-mode needs observed atoms) + bridge atom_embed.
        atom_embed = None
        if args.task in ("polymorph", "doping", "atomtxt", "strain"):
            try:
                input_atoms = _ase_atoms_from_struct(r["input_atoms_struct"])
                atom_embed = _live_orbv3_features(alm, input_atoms, device)
            except Exception as e:
                print(f"[bridge-edit] {r['row_id']}: live-encode failed "
                      f"({type(e).__name__}: {e}) → skip", flush=True)
                gens_per_row.append([]); target_comp_per_row.append(None)
                planner_text_per_row.append(None)
                continue
            input_struct = _ase_to_struct(r["input_atoms_struct"])
            if input_struct is None:
                gens_per_row.append([]); target_comp_per_row.append(None)
                planner_text_per_row.append(None)
                continue
            in_counts = _per_cell_counts(input_struct)
            if args.task in ("polymorph", "atomtxt"):
                target_comp = in_counts  # same composition, different structure
            elif args.doping_via_planner:
                # Planner->CSP (bridge off): LLM emits the substituted composition from text.
                instruction = re.sub(r"^\s*<atoms>\s*", "", str(prompt)).strip()
                dprompt = _DOPING_PLANNER_TMPL.format(counts=in_counts, instruction=instruction)
                plan_text, parsed = epc.llm_plan(dprompt, alm, tok,
                                                 prompt_version=args.prompt_version)
                target_comp, _fu = epc.comp_from_plan(parsed, args.prompt_version)
                if target_comp is None:
                    n_planner_parse_fail += 1
                    print(f"[bridge-edit] {r['row_id']}: doping planner parse fail "
                          f"→ skip  raw={(plan_text or '')[:80]!r}", flush=True)
                    gens_per_row.append([]); target_comp_per_row.append(None)
                    planner_text_per_row.append((plan_text or "")[:300])
                    continue
            else:  # doping (rule-based): apply X→Y to the input composition.
                donor, dopant = r["_donor"], r["_dopant"]
                target_comp = {}
                for el, n in in_counts.items():
                    key = dopant if el == donor else el
                    target_comp[key] = target_comp.get(key, 0) + n
                if donor not in in_counts:
                    # Donor element not present → can't apply the substitution; skip.
                    print(f"[bridge-edit] {r['row_id']}: donor {donor} absent in input "
                          f"{in_counts} → skip", flush=True)
                    gens_per_row.append([]); target_comp_per_row.append(None)
                    planner_text_per_row.append(None)
                    continue
        else:  # app: text-only; observed composition comes from the planner JSON.
            plan_text, parsed = epc.llm_plan(prompt, alm, tok,
                                             prompt_version=args.prompt_version)
            target_comp, _fu = epc.comp_from_plan(parsed, args.prompt_version)
            if target_comp is None:
                n_planner_parse_fail += 1
                gens_per_row.append([]); target_comp_per_row.append(None)
                planner_text_per_row.append((plan_text or "")[:300])
                continue

        target_comp_per_row.append(target_comp)
        planner_text_per_row.append((plan_text or "")[:300] if plan_text else None)

        # 3b. Bridge vector ([atoms_i] hidden states). json_counts=target_comp byte-matches
        # lm_loss_json training; omitting it puts the bridge vector off-distribution.
        alm_emb = None
        _td = int(r.get("_direction", 0)) if args.handset_direction else None
        if has_alm and not args.doping_via_planner:   # planner path is bridge-OFF
            if args.task in ("polymorph", "doping", "atomtxt", "strain"):
                alm_emb = get_alm_embedding(alm, tok, prompt, device,
                                            atom_embed=atom_embed,
                                            wrap_user_template=False,
                                            json_counts=target_comp,
                                            task_direction=_td,
                                            atoms_before_json=args.atoms_before_json,
                                            **_cot_kw(i))
            else:
                alm_emb = get_alm_embedding(alm, tok, prompt, device,
                                            json_counts=target_comp,
                                            task_direction=_td,
                                            atoms_before_json=args.atoms_before_json,
                                            **_cot_kw(i))
            if mean_emb is not None and alm_emb is not None:
                alm_emb = alm_emb - mean_emb

        # SEGA (atomtxt): compute the opposite-direction bridge vector; (asked - opp) applied at sampling.
        alm_emb_opp = None
        if (getattr(args, "sega_difference", False) and args.task == "atomtxt"
                and has_alm and not args.doping_via_planner):
            _askp, _oppp = _sega_prompts(r.get("_property", "formation_energy"),
                                         int(r.get("_direction", 0)))
            alm_emb = get_alm_embedding(alm, tok, _askp, device,
                                        atom_embed=atom_embed, wrap_user_template=False,
                                        json_counts=target_comp,
                                        atoms_before_json=args.atoms_before_json,
                                        **_cot_kw(i))
            alm_emb_opp = get_alm_embedding(alm, tok, _oppp, device,
                                            atom_embed=atom_embed, wrap_user_template=False,
                                            json_counts=target_comp,
                                            atoms_before_json=args.atoms_before_json,
                                            **_cot_kw(i))
            if i == 0:
                _cos = torch.nn.functional.cosine_similarity(
                    alm_emb.flatten().float(), alm_emb_opp.flatten().float(), dim=0).item()
                print(f"[bridge-edit] SEGA ON: sega_g={args.sega_g}  "
                      f"cos(asked,opp)={_cos:.4f}  ||asked-opp||="
                      f"{(alm_emb-alm_emb_opp).norm().item():.3f}", flush=True)

        if (i % 5) == 0:
            extra = f"  comp→{target_comp}" if args.task != "app" else f"  plan→{target_comp}"
            print(f"  [{i:3d}/{len(rows)}] {r['row_id']} {extra}  t={time.time()-t0:.0f}s",
                  flush=True)

        # 3b-FK. Per-prompt directional reward + reset cumulative weights.
        if fk_enabled:
            if args.task == "polymorph":
                _fkst.reward = _fk_reward_for("polymorph", -1)
            elif args.task == "strain":
                # Per-row target per-atom volume = vpa_input * (1 + target_dV%/100); not cached.
                _vpa_in = float(input_struct.volume) / max(len(input_struct), 1)
                _target_vpa = _vpa_in * (1.0 + float(r["_target_dv_pct"]) / 100.0)
                from fk_rewards import VolumeTargetReward as _VTR
                _fkst.reward = _VTR(target_vpa=_target_vpa)
            else:
                _fkst.reward = _fk_reward_for(r.get("_property", "formation_energy"),
                                              int(r.get("_direction", 0)))
            _fk_reset_per_prompt(_fkst, args.K, device)

        # 3c. Observed-atom CSP loader with alm_embedding stamped → sample.
        _sd = float(r.get("_direction", 0)) if args.scalar_direction else None
        samples = []
        gen_err = None
        _sega_restore = (_install_sega_on_sampler(sampler, alm_emb_opp, float(args.sega_g))
                         if alm_emb_opp is not None else None)
        # Retry the GemNet empty-graph device-mismatch with a bumped seed (perturbs the
        # trajectory, usually keeps the graph non-empty). Non-device errors are not retried.
        for _attempt in range(int(args.gen_retries) + 1):
            if _attempt > 0:
                _rs = (int(args.diffusion_seed) + i
                       + (_attempt * 100003)) & 0x7FFFFFFF
                torch.manual_seed(_rs)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(_rs)
                np.random.seed(_rs); _random.seed(_rs)
            try:
                loader = build_csp_condition_loader(target_comp, args.K, alm_emb,
                                                    task_direction=_sd)
                samples = draw_samples_from_sampler(
                    sampler=sampler,
                    condition_loader=loader,
                    properties_to_condition_on=None,  # alm_embedding stamped on chemgraphs
                    output_path=None,
                    cfg=OmegaConf.create({}),
                    record_trajectories=False,
                )
                gen_err = None
                if _attempt > 0:
                    print(f"    [{r['row_id']}] recovered on retry {_attempt} "
                          f"(seed-bump)", flush=True)
                break
            except Exception as e:
                gen_err = e
                if _is_device_mismatch(e) and _attempt < int(args.gen_retries):
                    print(f"    [{r['row_id']}] gen device-mismatch (empty-graph?) "
                          f"on attempt {_attempt} — re-seeding & retrying", flush=True)
                    samples = []
                    continue
                # Non-device error or retries exhausted: give up on this row.
                samples = []
                break
        if _sega_restore is not None:
            _sega_restore()
        if gen_err is not None:
            last_gen_error = gen_err
            if _is_device_mismatch(gen_err):
                n_gen_device_errors += 1
                print(f"    [{r['row_id']}] gen failed (device-mismatch, "
                      f"retries exhausted): {gen_err}", flush=True)
            else:
                n_gen_other_errors += 1
                print(f"    [{r['row_id']}] gen failed: {gen_err}", flush=True)
        gens_per_row.append(samples)
        if (i + 1) % 10 == 0:
            print(f"[bridge-edit] generated {i+1}/{len(rows)} prompts in {time.time()-t0:.0f}s",
                  flush=True)

    # ── 4. Score per task ──
    if args.task == "polymorph":
        headline, examples, n_scored = _score_polymorph(
            args, rows, gens_per_row, device)
    elif args.task == "atomtxt":
        headline, examples, n_scored = _score_atomtxt(
            args, rows, gens_per_row, device)
    elif args.task == "doping":
        headline, examples, n_scored = _score_doping(
            args, rows, gens_per_row)
    elif args.task == "strain":
        headline, examples, n_scored = _score_strain(
            args, rows, gens_per_row, device)
    elif args.task in ("describe", "ood"):
        headline, examples, n_scored = _score_text2struct(
            args, rows, gens_per_row)
    else:
        headline, examples, n_scored = _score_app(
            args, rows, gens_per_row, target_comp_per_row,
            planner_text_per_row, n_planner_parse_fail, device, t0)

    # ── 4b. Optional showcase CIF dump (read-only on scoring) ──
    if getattr(args, "save_cifs", None) is not None:
        try:
            _save_showcase_cifs(args, rows, gens_per_row, examples,
                                target_comp_per_row)
        except Exception as _e:  # never let the showcase dump break the eval
            print(f"[save_cifs] FAILED ({type(_e).__name__}: {_e}) — eval metrics unaffected",
                  flush=True)

    # ── 5. Write metrics + graceful-degradation n_scored gating ──
    headline["task"] = args.task
    headline["n_scored"] = n_scored
    headline.setdefault("n_prompts", len(rows))
    headline["K"] = args.K
    headline["guidance_factor"] = args.guidance_factor
    headline["has_alm_embedding_cond"] = has_alm
    headline["atoms_mapper"] = str(args.atoms_mapper)
    headline["mattergen_model_path"] = args.mattergen_model_path
    headline["cot_tokens"] = int(args.cot_tokens)
    headline["llm_temperature"] = float(args.llm_temperature)
    headline["whiten_common_mode"] = bool(getattr(args, "whiten_common_mode", False))
    headline["sega_difference"] = bool(getattr(args, "sega_difference", False))
    headline["sega_g"] = float(getattr(args, "sega_g", 0.0))
    headline["atoms_before_json"] = bool(getattr(args, "atoms_before_json", False))
    headline["wallclock_sec"] = time.time() - t0
    # Failure-class counters: distinguish "all rows hit the same fixable error" from a real empty result.
    headline["n_gen_device_errors"] = n_gen_device_errors
    headline["n_gen_other_errors"] = n_gen_other_errors
    headline["n_planner_parse_fail"] = n_planner_parse_fail
    if n_scored <= 0 and n_gen_device_errors > 0 and n_gen_other_errors == 0:
        headline["status"] = "device_mismatch_all_rows"
        headline["last_gen_error"] = str(last_gen_error)[:300]
    elif n_scored <= 0:
        headline["status"] = "zero_scored"
        if last_gen_error is not None:
            headline["last_gen_error"] = str(last_gen_error)[:300]
    else:
        headline["status"] = "ok"

    (args.out_dir / "metrics.json").write_text(json.dumps(headline, indent=2))
    with (args.out_dir / "predictions.jsonl").open("w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")

    print(f"\n[bridge-edit] HEADLINE task={args.task} (n_scored={n_scored}):", flush=True)
    for k, v in headline.items():
        if isinstance(v, float):
            print(f"  {k:32s} = {v:.4f}", flush=True)

    if n_scored > 0:
        print(f"[bridge-edit] DONE in {time.time()-t0:.0f}s", flush=True)
        return 0

    # n_scored == 0 below: distinguish the failure classes loudly.
    sharded = int(args.num_shards) > 1
    if n_gen_device_errors > 0 and n_gen_other_errors == 0:
        print(f"\n[bridge-edit] *** ALL {n_gen_device_errors} ROW(S) FAILED ON THE SAME "
              f"EXCEPTION: cuda/cpu device-mismatch from MatterGen's GemNet empty-graph "
              f"branch (zero edges → CPU tensor meets a cuda weight). This is a "
              f"PLUMBING/empty-graph failure, NOT a 0.0 result. Last error:\n"
              f"    {str(last_gen_error)[:300]}\n"
              f"[bridge-edit] Mitigations already applied this run: per-row seed-bump "
              f"retries (--gen_retries={args.gen_retries}) + on-device conditioning stamp. "
              f"If still 100% (planner mode-collapse to tiny binaries → all-out-of-cutoff "
              f"cells), the durable fix is the empty-graph device arg in "
              f"external/.../gemnet/layers/efficient.py (out of scope here). ***",
              flush=True)
    else:
        print("\n[bridge-edit] *** n_scored == 0 — PLUMBING FAILURE (no rows scored, and "
              f"NOT the all-device-mismatch signature: device_errors={n_gen_device_errors} "
              f"other_errors={n_gen_other_errors} planner_parse_fail={n_planner_parse_fail}). "
              "Check live-encode / planner-parse / generation logs above. ***", flush=True)

    if sharded and not args.strict_n_scored:
        # Graceful degradation: exit 0 so one shard scoring 0 cannot poison the aggregated run.
        print("[bridge-edit] sharded run (num_shards=%d): writing metrics.json with "
              "n_scored=0 and exiting 0 so surviving shards still aggregate "
              "(use --strict_n_scored to hard-fail per shard)." % int(args.num_shards),
              flush=True)
        return 0
    # Single-shard / manual / --strict_n_scored run: 0 truly is total failure.
    raise SystemExit(2)


# ─────────────────────────────────────────────────────────────────────────────
# Per-task scorers
# ─────────────────────────────────────────────────────────────────────────────
def _to_struct(g):
    if isinstance(g, Structure):
        return g
    try:
        return AseAtomsAdaptor.get_structure(g)
    except Exception:
        return None


def _score_polymorph(args, rows, gens_per_row, device):
    """PRIMARY: polymorph_lower_energy_rate (relaxed gen E/atom < relaxed input E/atom), plus distinctness/composition/validity."""
    from structure_metrics import relax_structures_mattersim, total_energy_per_atom
    matcher = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)

    # Flat relax batch: per row [input] + [gens], so input + gens share the same potential/settings.
    flat: list[Structure] = []
    flat_meta: list[tuple[int, int]] = []  # (row_i, gen_j); gen_j == -1 -> input
    input_struct_per_row: list[Structure | None] = []
    for i, gens in enumerate(gens_per_row):
        r = rows[i]
        input_struct = _ase_to_struct(r["input_atoms_struct"])
        input_struct_per_row.append(input_struct)
        if input_struct is None or not gens:
            continue
        zs = [int(s.specie.Z) for s in input_struct]
        if zs and all(1 <= z <= 94 for z in zs):
            flat.append(input_struct); flat_meta.append((i, -1))
        for j, g in enumerate(gens):
            s = _to_struct(g)
            if s is None:
                continue
            try:
                zs = [int(site.specie.Z) for site in s]
            except Exception:
                continue
            if not zs or any(z < 1 or z > 94 for z in zs):
                continue
            flat.append(s); flat_meta.append((i, j))

    energy_by_key: dict[tuple[int, int], float] = {}
    relaxed_struct_by_key: dict[tuple[int, int], Structure] = {}
    if not args.skip_relax and flat:
        print(f"[bridge-edit:polymorph] MatterSim relaxing {len(flat)} structures "
              f"(inputs + gens) ...", flush=True)
        relaxed_atoms, _ = relax_structures_mattersim(
            flat, device=str(device), potential_path=args.mattersim_potential_path,
            fmax=0.05, max_n_steps=500,
        )
        for (key, atoms) in zip(flat_meta, relaxed_atoms):
            energy_by_key[key] = total_energy_per_atom(atoms)
            try:
                relaxed_struct_by_key[key] = AseAtomsAdaptor.get_structure(atoms)
            except Exception:
                pass

    examples = []
    overall = Counter()
    n_scored = 0
    n_lower = 0
    n_lower_real = 0
    for i, gens in enumerate(gens_per_row):
        r = rows[i]
        input_struct = input_struct_per_row[i]
        if input_struct is None or not gens:
            continue
        e_input = energy_by_key.get((i, -1), float("nan"))
        per_candidate = []
        for j, g in enumerate(gens):
            s = _to_struct(g)
            if s is None:
                continue
            comp_match = (s.composition.reduced_formula ==
                          input_struct.composition.reduced_formula)
            valid_geom = validity_geom(s)
            valid_charge = validity_charge(s)
            valid = _structurally_valid(s)
            distinct = True
            if comp_match and valid:
                try:
                    distinct = not matcher.fit(input_struct, s)
                except Exception:
                    distinct = True
            e_gen = energy_by_key.get((i, j), float("nan"))
            lower_energy = bool(e_gen == e_gen and e_input == e_input and e_gen < e_input)
            # GATED real win: lower energy AND same composition AND distinct AND valid.
            real_lower = bool(lower_energy and comp_match and distinct and valid)
            sc = {
                "composition_preserved": comp_match,
                "structurally_distinct": distinct,
                "structurally_valid": valid,
                "validity_geom": valid_geom,
                "validity_charge": valid_charge,
                "validity_both": valid_geom and valid_charge,
                "e_gen_per_atom": e_gen if e_gen == e_gen else None,
                "e_input_per_atom": e_input if e_input == e_input else None,
                "lower_energy": lower_energy,
                "real_lower_energy": real_lower,
            }
            per_candidate.append(sc)
            n_scored += 1
            for k in ("composition_preserved", "structurally_distinct",
                      "structurally_valid", "validity_geom", "validity_charge",
                      "validity_both"):
                if sc[k]:
                    overall[k] += 1
            if lower_energy:
                n_lower += 1
                overall["lower_energy"] += 1
            if real_lower:
                n_lower_real += 1
        examples.append({
            "row_id": r["row_id"], "parent": r.get("parent"),
            "user_prompt": r["user_prompt"],
            "input_formula": str(input_struct.composition.reduced_formula),
            "e_input_per_atom": e_input if e_input == e_input else None,
            "n_candidates": len(per_candidate),
            "per_candidate_scores": per_candidate,
            "lower_energy_rate": float(np.mean([c["lower_energy"] for c in per_candidate]))
                                 if per_candidate else 0.0,
        })

    headline = {}
    if n_scored > 0:
        for k in ("composition_preserved", "structurally_distinct",
                  "structurally_valid", "validity_geom", "validity_charge",
                  "validity_both"):
            headline[k] = overall[k] / n_scored
        headline["polymorph_lower_energy_rate"] = n_lower_real / n_scored  # GATED PRIMARY
        headline["polymorph_lower_energy_rate_ungated"] = n_lower / n_scored  # diagnostic
    headline["per_prompt_mean_lower_energy"] = (
        float(np.mean([e["lower_energy_rate"] for e in examples])) if examples else 0.0)
    headline["skip_relax"] = bool(args.skip_relax)
    return headline, examples, n_scored


def _density_g_cm3(struct: Structure) -> float:
    """Geometric density (g/cm^3) from a raw structure."""
    from ase.data import atomic_masses, atomic_numbers
    try:
        mass = sum(atomic_masses[atomic_numbers[str(s.specie)]] for s in struct)
        return float(mass * 1.66054 / max(float(struct.volume), 1e-6))
    except Exception:
        return float("nan")


def _volume_per_atom(struct: Structure) -> float:
    """Cell volume per atom (A^3/atom) from a raw structure."""
    try:
        return float(struct.volume) / max(len(struct), 1)
    except Exception:
        return float("nan")


def _density_from_atoms(atoms) -> float:
    """Geometric density (g/cm^3) from a relaxed ASE Atoms."""
    try:
        return float(atoms.get_masses().sum() * 1.66054 / max(float(atoms.get_volume()), 1e-6))
    except Exception:
        return float("nan")


def _vol_per_atom_from_atoms(atoms) -> float:
    """Cell volume per atom (A^3/atom) from a relaxed ASE Atoms."""
    try:
        return float(atoms.get_volume()) / max(len(atoms), 1)
    except Exception:
        return float("nan")


def _score_atomtxt(args, rows, gens_per_row, device):
    """Directional property editing at fixed composition; direction_correct = sign(prop_gen - prop_input) == requested. Discriminating signal is the higher-direction rate."""
    from structure_metrics import relax_structures_mattersim, total_energy_per_atom

    input_struct_per_row: list[Structure | None] = []
    for i in range(len(gens_per_row)):
        input_struct_per_row.append(_ase_to_struct(rows[i]["input_atoms_struct"]))

    # density/volume are relaxed too (de-gamed): a pure lattice rescale relaxes back to the
    # input's equilibrium volume, so only a genuinely different material moves relaxed density.
    _RELAX_PROPS = ("formation_energy", "density", "volume")
    flat: list[Structure] = []
    flat_meta: list[tuple[int, int]] = []
    for i, gens in enumerate(gens_per_row):
        if rows[i].get("_property") not in _RELAX_PROPS:
            continue
        input_struct = input_struct_per_row[i]
        if input_struct is None or not gens:
            continue
        zs = [int(s.specie.Z) for s in input_struct]
        if zs and all(1 <= z <= 94 for z in zs):
            flat.append(input_struct); flat_meta.append((i, -1))
        for j, g in enumerate(gens):
            s = _to_struct(g)
            if s is None:
                continue
            try:
                zs = [int(site.specie.Z) for site in s]
            except Exception:
                continue
            if not zs or any(z < 1 or z > 94 for z in zs):
                continue
            flat.append(s); flat_meta.append((i, j))

    energy_by_key: dict[tuple[int, int], float] = {}
    relaxed_atoms_by_key: dict[tuple[int, int], object] = {}
    if not args.skip_relax and flat:
        # Force full relax whenever density/volume present (de-game needs equilibrium geometry);
        # --single_point_energy (max_n_steps=0) is honored only for FE-only runs.
        _props_present = {rows[i].get("_property") for i in range(len(gens_per_row))}
        _full_relax_needed = bool(_props_present & {"density", "volume"})
        _atx_steps = 0 if (args.single_point_energy and not _full_relax_needed) else 500
        print(f"[bridge-edit:atomtxt] MatterSim "
              f"{'SINGLE-POINT (max_n_steps=0, raw geometry)' if _atx_steps == 0 else 'full-relaxing'} "
              f"{len(flat)} structures (inputs + gens; props={sorted(p for p in _props_present if p)}) ...",
              flush=True)
        relaxed_atoms, _ = relax_structures_mattersim(
            flat, device=str(device), potential_path=args.mattersim_potential_path,
            fmax=0.05, max_n_steps=_atx_steps,
        )
        for key, atoms in zip(flat_meta, relaxed_atoms):
            energy_by_key[key] = total_energy_per_atom(atoms)
            relaxed_atoms_by_key[key] = atoms

    # De-game gate: a pure lattice rescale matches the input under the volume-normalized matcher.
    _distinct_matcher = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)

    examples = []
    overall = Counter()
    n_scored = 0
    _saved_gens = [] if getattr(args, "save_gens", None) else None
    by: dict[str, dict[str, int]] = {}  # (property, direction) -> counts, for cross-shard re-aggregation

    def _bump(prop, want, correct):
        d = by.setdefault(f"{prop}_{'higher' if want > 0 else 'lower'}",
                          {"n": 0, "correct": 0})
        d["n"] += 1
        if correct:
            d["correct"] += 1

    for i, gens in enumerate(gens_per_row):
        r = rows[i]
        input_struct = input_struct_per_row[i]
        if input_struct is None or not gens:
            continue
        prop = r.get("_property", "formation_energy")
        want = int(r.get("_direction", 0))
        if prop == "formation_energy":
            p_input = energy_by_key.get((i, -1), float("nan"))
        elif prop in ("density", "volume"):  # de-gamed: relaxed-equilibrium geometry
            _ria = relaxed_atoms_by_key.get((i, -1))
            if _ria is not None:
                p_input = (_density_from_atoms(_ria) if prop == "density"
                           else _vol_per_atom_from_atoms(_ria))
            else:  # --skip_relax fallback: raw geometric (gameable)
                p_input = (_density_g_cm3(input_struct) if prop == "density"
                           else _volume_per_atom(input_struct))
        else:
            p_input = float("nan")
        per_candidate = []
        for j, g in enumerate(gens):
            s = _to_struct(g)
            if s is None:
                continue
            comp_match = (s.composition.reduced_formula ==
                          input_struct.composition.reduced_formula)
            valid = _structurally_valid(s)
            if prop == "formation_energy":
                p_gen = energy_by_key.get((i, j), float("nan"))
            elif prop in ("density", "volume"):
                _rga = relaxed_atoms_by_key.get((i, j))
                if _rga is not None:
                    p_gen = (_density_from_atoms(_rga) if prop == "density"
                             else _vol_per_atom_from_atoms(_rga))
                else:
                    p_gen = (_density_g_cm3(s) if prop == "density"
                             else _volume_per_atom(s))
            else:
                p_gen = float("nan")
            dP = (p_gen - p_input) if (p_gen == p_gen and p_input == p_input) else float("nan")
            _dir_sign = bool(dP == dP and ((want > 0 and dP > 0) or (want < 0 and dP < 0)))
            # De-game (density/volume): reject pure lattice rescales via distinct_from_input.
            distinct_from_input = True
            if prop in ("density", "volume"):
                try:
                    distinct_from_input = not _distinct_matcher.fit(s, input_struct)
                except Exception:
                    distinct_from_input = False
                # Correct = same composition, distinct structure, valid, and moved the requested way.
                dir_correct = bool(_dir_sign and valid and comp_match and distinct_from_input)
            else:
                dir_correct = _dir_sign
            sc = {
                "property": prop,
                "composition_preserved": comp_match,
                "structurally_valid": valid,
                "distinct_from_input": distinct_from_input,
                "prop_gen": p_gen if p_gen == p_gen else None,
                "prop_input": p_input if p_input == p_input else None,
                "delta_prop": dP if dP == dP else None,
                "requested_direction": "higher" if want > 0 else "lower",
                "direction_correct": dir_correct,
            }
            per_candidate.append(sc)
            n_scored += 1
            if _saved_gens is not None:
                try:
                    # Structures as JSON strings (nested float lists trip pyarrow type inference).
                    _injson = json.dumps({
                        "elements": [str(sp) for sp in input_struct.species],
                        "frac_coords": input_struct.frac_coords.tolist(),
                        "lattice": input_struct.lattice.matrix.tolist()})
                    _genjson = json.dumps({
                        "elements": [str(sp) for sp in s.species],
                        "frac_coords": s.frac_coords.tolist(),
                        "lattice": s.lattice.matrix.tolist()})
                    _saved_gens.append({
                        "row_id": r["row_id"],
                        "user_prompt": r["user_prompt"],
                        "requested_direction": "higher" if want > 0 else "lower",
                        "direction_correct": bool(dir_correct),
                        "composition_preserved": bool(comp_match),
                        "structurally_valid": bool(valid),
                        "prop_gen": float(p_gen) if p_gen == p_gen else None,
                        "prop_input": float(p_input) if p_input == p_input else None,
                        "delta_prop": float(dP) if dP == dP else None,
                        "input_struct_json": _injson,
                        "gen_struct_json": _genjson,
                    })
                except Exception:
                    pass
            if comp_match:
                overall["composition_preserved"] += 1
            if valid:
                overall["structurally_valid"] += 1
            # Degenerate/failed gen (NaN property) is dir_correct=False, kept in the denominator
            # as wrong (never excluded); n_gen_failed is a diagnostic, not a gate.
            if dP != dP:
                overall["n_gen_failed"] += 1
            if dir_correct:
                overall["direction_correct"] += 1
                overall[f"correct_{prop}"] += 1
                if want > 0:
                    overall["correct_higher"] += 1
                else:
                    overall["correct_lower"] += 1
            overall[f"n_{prop}"] += 1
            overall["n_higher" if want > 0 else "n_lower"] += 1
            _bump(prop, want, dir_correct)
        examples.append({
            "row_id": r["row_id"], "parent": r.get("parent"), "property": prop,
            "user_prompt": r["user_prompt"],
            "requested_direction": "higher" if want > 0 else "lower",
            "input_formula": str(input_struct.composition.reduced_formula),
            "prop_input": p_input if p_input == p_input else None,
            "n_candidates": len(per_candidate),
            "per_candidate_scores": per_candidate,
            "direction_correct_rate": float(np.mean(
                [c["direction_correct"] for c in per_candidate])) if per_candidate else 0.0,
        })

    if _saved_gens is not None:
        import pandas as _pd
        args.save_gens.mkdir(parents=True, exist_ok=True)
        _gp = args.save_gens / f"gens_shard{getattr(args, 'shard_idx', 0)}.parquet"
        _pd.DataFrame(_saved_gens).to_parquet(_gp)
        _hc = sum(1 for g in _saved_gens
                  if g["requested_direction"] == "higher" and g["direction_correct"])
        print(f"[save_gens] wrote {len(_saved_gens)} gen records ({_hc} higher-correct) → {_gp}",
              flush=True)

    def _rate(num, den):
        return (overall[num] / overall[den]) if overall[den] else None

    headline = {}
    if n_scored > 0:
        # PRIMARY: direction-correct over ALL scored candidates (failed gens kept as wrong).
        headline["direction_correct_rate"] = overall["direction_correct"] / n_scored
        headline["composition_preserved"] = overall["composition_preserved"] / n_scored
        headline["structurally_valid"] = overall["structurally_valid"] / n_scored
        headline["gen_failed_rate"] = overall["n_gen_failed"] / n_scored  # diagnostic, not a gate
    # discriminating splits (None when a shard saw none of that bucket)
    headline["higher_direction_correct_rate"] = _rate("correct_higher", "n_higher")
    headline["lower_direction_correct_rate"] = _rate("correct_lower", "n_lower")
    headline["fe_direction_correct_rate"] = _rate("correct_formation_energy", "n_formation_energy")
    headline["density_direction_correct_rate"] = _rate("correct_density", "n_density")
    headline["volume_direction_correct_rate"] = _rate("correct_volume", "n_volume")
    # raw counts for exact cross-shard re-aggregation
    for key, d in sorted(by.items()):
        headline[f"cnt_{key}_n"] = d["n"]
        headline[f"cnt_{key}_correct"] = d["correct"]
    headline["n_higher_scored"] = overall["n_higher"]
    headline["n_lower_scored"] = overall["n_lower"]
    headline["n_formation_energy"] = overall["n_formation_energy"]
    headline["n_density"] = overall["n_density"]
    headline["n_volume"] = overall["n_volume"]
    headline["skip_relax"] = bool(args.skip_relax)
    return headline, examples, n_scored


def _score_doping(args, rows, gens_per_row):
    """correct_substitution_rate (Y present AND X removed AND ratio_match AND valid), plus components. No relax."""
    examples = []
    overall = Counter()
    n_scored = 0
    for i, gens in enumerate(gens_per_row):
        r = rows[i]
        donor, dopant = r["_donor"], r["_dopant"]
        input_struct = _ase_to_struct(r["input_atoms_struct"])
        if input_struct is None or not gens:
            continue
        in_elems = [str(s.specie.symbol) for s in input_struct]
        in_count = Counter(in_elems)
        n_in = max(1, len(in_elems))
        target_frac = in_count.get(donor, 0) / n_in
        # Naive relabel = input with donor sites renamed to dopant, identical coords/lattice;
        # a gen matching it is mere relabeling, not real doping, so it is rejected below.
        naive_relabel = None
        if donor in in_count:
            try:
                _nr = input_struct.copy()
                _nr.replace_species({donor: dopant})
                naive_relabel = _nr
            except Exception:
                naive_relabel = None
        _relabel_matcher = StructureMatcher(ltol=args.ltol, stol=args.stol,
                                            angle_tol=args.angle_tol)
        per_candidate = []
        for g in gens:
            s = _to_struct(g)
            if s is None:
                continue
            gen_elems = [str(site.specie.symbol) for site in s]
            gen_count = Counter(gen_elems)
            n_gen = max(1, len(gen_elems))
            dopant_present = gen_count.get(dopant, 0) > 0
            donor_removed = gen_count.get(donor, 0) == 0
            gen_frac = gen_count.get(dopant, 0) / n_gen
            ratio_match = abs(gen_frac - target_frac) <= 0.10
            valid = _structurally_valid(s)
            full_sub = dopant_present and donor_removed
            # Real-structure gate: distinct from the naive relabel (no relabel available -> distinct=True).
            is_relabel = False
            if naive_relabel is not None:
                try:
                    is_relabel = bool(_relabel_matcher.fit(s, naive_relabel))
                except Exception:
                    is_relabel = False
            distinct_from_relabel = not is_relabel
            correct = full_sub and ratio_match and valid and distinct_from_relabel
            sc = {
                "dopant_present": dopant_present,
                "donor_removed": donor_removed,
                "ratio_match": ratio_match,
                "structurally_valid": valid,
                "distinct_from_relabel": distinct_from_relabel,
                "full_substitution": full_sub,
                "correct_substitution": correct,
                "gen_formula": str(s.composition.reduced_formula),
            }
            per_candidate.append(sc)
            n_scored += 1
            for k in ("dopant_present", "donor_removed", "ratio_match",
                      "structurally_valid", "distinct_from_relabel",
                      "full_substitution", "correct_substitution"):
                if sc[k]:
                    overall[k] += 1
        examples.append({
            "row_id": r["row_id"], "parent": r.get("parent"),
            "user_prompt": r["user_prompt"], "donor": donor, "dopant": dopant,
            "input_formula": str(input_struct.composition.reduced_formula),
            "n_candidates": len(per_candidate),
            "per_candidate_scores": per_candidate,
            "correct_substitution_rate": float(np.mean(
                [c["correct_substitution"] for c in per_candidate])) if per_candidate else 0.0,
            "full_substitution_rate": float(np.mean(
                [c["full_substitution"] for c in per_candidate])) if per_candidate else 0.0,
        })

    headline = {}
    if n_scored > 0:
        for k in ("dopant_present", "donor_removed", "ratio_match",
                  "structurally_valid", "distinct_from_relabel",
                  "full_substitution", "correct_substitution"):
            headline[k] = overall[k] / n_scored
        headline["correct_substitution_rate"] = overall["correct_substitution"] / n_scored
    headline["per_prompt_mean_correct_sub"] = (
        float(np.mean([e["correct_substitution_rate"] for e in examples])) if examples else 0.0)
    return headline, examples, n_scored


def _score_strain(args, rows, gens_per_row, device):
    """strain_correct_rate: doping correct AND relaxed dV hits the row_id's target dV (within strain_tol_pct)."""
    from structure_metrics import relax_structures_mattersim

    # Flat relax batch: per row [input] + [gens], input relaxed with the same potential/settings.
    flat: list[Structure] = []
    flat_meta: list[tuple[int, int]] = []  # (row_i, gen_j); gen_j == -1 -> input
    input_struct_per_row: list[Structure | None] = []
    for i, gens in enumerate(gens_per_row):
        r = rows[i]
        input_struct = _ase_to_struct(r["input_atoms_struct"])
        input_struct_per_row.append(input_struct)
        if input_struct is None or not gens:
            continue
        zs = [int(s.specie.Z) for s in input_struct]
        if zs and all(1 <= z <= 94 for z in zs):
            flat.append(input_struct); flat_meta.append((i, -1))
        for j, g in enumerate(gens):
            s = _to_struct(g)
            if s is None:
                continue
            try:
                zs = [int(site.specie.Z) for site in s]
            except Exception:
                continue
            if not zs or any(z < 1 or z > 94 for z in zs):
                continue
            flat.append(s); flat_meta.append((i, j))

    vpa_by_key: dict[tuple[int, int], float] = {}
    if not args.skip_relax and flat:
        print(f"[bridge-edit:strain] MatterSim relaxing {len(flat)} structures "
              f"(inputs + gens) ...", flush=True)
        relaxed_atoms, _ = relax_structures_mattersim(
            flat, device=str(device), potential_path=args.mattersim_potential_path,
            fmax=0.05, max_n_steps=500,
        )
        for key, atoms in zip(flat_meta, relaxed_atoms):
            vpa_by_key[key] = _vol_per_atom_from_atoms(atoms)

    examples = []
    overall = Counter()
    n_scored = 0
    realized_dv = []
    for i, gens in enumerate(gens_per_row):
        r = rows[i]
        donor, dopant = r["_donor"], r["_dopant"]
        target_dv = r["_target_dv_pct"]
        input_struct = input_struct_per_row[i]
        if input_struct is None or not gens:
            continue
        in_elems = [str(s.specie.symbol) for s in input_struct]
        in_count = Counter(in_elems)
        n_in = max(1, len(in_elems))
        target_frac = in_count.get(donor, 0) / n_in
        # naive (coords-identical) relabel, disallowed (same realness gate as doping).
        naive_relabel = None
        if donor in in_count:
            try:
                _nr = input_struct.copy()
                _nr.replace_species({donor: dopant})
                naive_relabel = _nr
            except Exception:
                naive_relabel = None
        _relabel_matcher = StructureMatcher(ltol=args.ltol, stol=args.stol,
                                            angle_tol=args.angle_tol)
        vpa_input = vpa_by_key.get((i, -1), float("nan"))
        per_candidate = []
        for j, g in enumerate(gens):
            s = _to_struct(g)
            if s is None:
                continue
            gen_count = Counter(str(site.specie.symbol) for site in s)
            n_gen = max(1, sum(gen_count.values()))
            dopant_present = gen_count.get(dopant, 0) > 0
            donor_removed = gen_count.get(donor, 0) == 0
            gen_frac = gen_count.get(dopant, 0) / n_gen
            ratio_match = abs(gen_frac - target_frac) <= 0.10
            valid = _structurally_valid(s)
            full_sub = dopant_present and donor_removed
            is_relabel = False
            if naive_relabel is not None:
                try:
                    is_relabel = bool(_relabel_matcher.fit(s, naive_relabel))
                except Exception:
                    is_relabel = False
            distinct_from_relabel = not is_relabel
            doping_correct = full_sub and ratio_match and valid and distinct_from_relabel
            # Relaxed per-atom volume change vs relaxed input.
            vpa_gen = vpa_by_key.get((i, j), float("nan"))
            realized_dv_pct = float("nan")
            if vpa_gen == vpa_gen and vpa_input == vpa_input and vpa_input > 0:
                realized_dv_pct = 100.0 * (vpa_gen - vpa_input) / vpa_input
            volume_match = bool(realized_dv_pct == realized_dv_pct
                                and abs(realized_dv_pct - target_dv) < args.strain_tol_pct)
            strain_correct = bool(doping_correct and volume_match)
            sc = {
                "doping_correct": doping_correct,
                "volume_match": volume_match,
                "strain_correct": strain_correct,
                "target_dv_pct": target_dv,
                "realized_dv_pct": realized_dv_pct if realized_dv_pct == realized_dv_pct else None,
                "dopant_present": dopant_present, "donor_removed": donor_removed,
                "ratio_match": ratio_match, "structurally_valid": valid,
                "distinct_from_relabel": distinct_from_relabel,
                "gen_formula": str(s.composition.reduced_formula),
            }
            per_candidate.append(sc)
            n_scored += 1
            for k in ("doping_correct", "volume_match", "strain_correct",
                      "dopant_present", "donor_removed", "ratio_match",
                      "structurally_valid", "distinct_from_relabel"):
                if sc[k]:
                    overall[k] += 1
            if realized_dv_pct == realized_dv_pct:
                realized_dv.append((realized_dv_pct, target_dv))
        examples.append({
            "row_id": r["row_id"], "parent": r.get("parent"),
            "user_prompt": r["user_prompt"], "donor": donor, "dopant": dopant,
            "target_dv_pct": target_dv,
            "input_formula": str(input_struct.composition.reduced_formula),
            "n_candidates": len(per_candidate),
            "per_candidate_scores": per_candidate,
            "strain_correct_rate": float(np.mean(
                [c["strain_correct"] for c in per_candidate])) if per_candidate else 0.0,
        })

    headline = {}
    if n_scored > 0:
        for k in ("doping_correct", "volume_match", "dopant_present", "donor_removed",
                  "ratio_match", "structurally_valid", "distinct_from_relabel"):
            headline[k] = overall[k] / n_scored
        headline["doping_correct_rate"] = overall["doping_correct"] / n_scored
        headline["volume_match_rate"] = overall["volume_match"] / n_scored
        headline["strain_correct_rate"] = overall["strain_correct"] / n_scored  # GATED PRIMARY
    if realized_dv:
        errs = [abs(rd - td) for rd, td in realized_dv]
        headline["mean_abs_dv_err_pct"] = float(np.mean(errs))
        headline["median_abs_dv_err_pct"] = float(np.median(errs))
    headline["strain_tol_pct"] = float(args.strain_tol_pct)
    headline["skip_relax"] = bool(args.skip_relax)
    return headline, examples, n_scored


def _score_text2struct(args, rows, gens_per_row):
    """describe/ood text->structure recovery vs GT: comp_match_rate + struct_match_rate (rough disordered matcher), no relax."""
    try:
        from mattergen.evaluation.utils.structure_matcher import DisorderedStructureMatcher
        rough = DisorderedStructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)
    except Exception:
        rough = StructureMatcher(ltol=0.3, stol=0.5, angle_tol=10.0)
    examples = []
    overall = Counter()
    n_scored = 0
    n_prompts = 0
    n_gen_failed = 0
    comp_any_k = 0
    struct_any_k = 0
    for i, gens in enumerate(gens_per_row):
        r = rows[i]
        gt = _ase_to_struct(r.get("atoms_struct"))
        if gt is None:
            continue  # GT missing -> cannot score
        gt_formula = gt.composition.reduced_formula
        if not gens:
            # No structure produced: count as one scored-WRONG candidate (kept in denominator).
            n_scored += 1
            n_gen_failed += 1
            n_prompts += 1
            examples.append({"row_id": r["row_id"], "gt_formula": gt_formula,
                             "gen_failed": True, "n_candidates": 0})
            continue
        per_candidate = []
        row_comp_hit = False
        row_struct_hit = False
        for g in gens:
            s = _to_struct(g)
            if s is None:
                continue
            comp_match = (s.composition.reduced_formula == gt_formula)
            struct_match = False
            if comp_match and _structurally_valid(s):
                try:
                    struct_match = bool(rough.fit(gt, s))
                except Exception:
                    struct_match = False
            sc = {"comp_match": comp_match, "struct_match": struct_match,
                  "gen_formula": str(s.composition.reduced_formula)}
            per_candidate.append(sc)
            n_scored += 1
            if comp_match:
                overall["comp_match"] += 1; row_comp_hit = True
            if struct_match:
                overall["struct_match"] += 1; row_struct_hit = True
        if not per_candidate:
            continue
        n_prompts += 1
        comp_any_k += int(row_comp_hit)
        struct_any_k += int(row_struct_hit)
        examples.append({
            "row_id": r["row_id"], "parent": r.get("parent"),
            "user_prompt": str(r["user_prompt"])[:200],
            "gt_formula": gt_formula, "n_candidates": len(per_candidate),
            "per_candidate_scores": per_candidate,
            "comp_match_rate": float(np.mean([c["comp_match"] for c in per_candidate])),
            "struct_match_rate": float(np.mean([c["struct_match"] for c in per_candidate])),
        })

    headline = {}
    if n_scored > 0:
        headline["comp_match_rate"] = overall["comp_match"] / n_scored
        headline["struct_match_rate"] = overall["struct_match"] / n_scored
    if n_prompts > 0:
        headline["comp_match_at_k"] = comp_any_k / n_prompts      # any-of-K diagnostic
        headline["struct_match_at_k"] = struct_any_k / n_prompts
        headline["n_prompts_scored"] = n_prompts
    if n_scored > 0:
        headline["gen_failed_rate"] = n_gen_failed / n_scored      # diagnostic, not a gate
    headline["n_gen_failed"] = n_gen_failed
    headline["matcher"] = "DisorderedStructureMatcher(0.3/0.5/10)"
    return headline, examples, n_scored


def _score_app(args, rows, gens_per_row, target_comp_per_row, planner_text_per_row,
               n_planner_parse_fail, device, t0):
    """App-consistency via the eval_app_consistency LLM judge on relaxed gens."""
    import asyncio
    from ase.data import atomic_masses, atomic_numbers
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from structure_metrics import relax_structures_mattersim, total_energy_per_atom
    from llm_judge import (batch_judge, build_app_consistency_messages, parse_score,
                                reset_failure_counts, get_failure_counts)

    def _summary(struct: Structure) -> dict:
        elements_set = sorted({str(e) for e in struct.composition.elements})
        try:
            sg = SpacegroupAnalyzer(struct, symprec=0.1).get_space_group_symbol()
        except Exception:
            sg = "?"
        n_atoms = int(struct.num_sites)
        vol = float(struct.volume)
        vpa = vol / max(n_atoms, 1)
        mass = sum(atomic_masses[atomic_numbers[str(s.specie)]] for s in struct)
        density = float(mass * 1.66054 / max(vol, 1e-3))
        return {"formula": struct.composition.reduced_formula, "space_group": sg,
                "n_atoms": n_atoms, "elements": elements_set,
                "density": density, "volume_per_atom": vpa}

    # Flatten gens; filter out-of-range Z (device-side asserts in MatterSim).
    flat: list[Structure] = []
    flat_meta: list[int] = []
    n_filtered_z = 0
    for i, gens in enumerate(gens_per_row):
        for g in gens:
            s = _to_struct(g)
            if s is None:
                continue
            try:
                zs = [int(site.specie.Z) for site in s]
            except Exception:
                continue
            if not zs or any(z < 1 or z > 94 for z in zs):
                n_filtered_z += 1
                continue
            flat.append(s); flat_meta.append(i)
    if n_filtered_z:
        print(f"[bridge-edit:app] pre-filtered {n_filtered_z} structures with out-of-range Z",
              flush=True)

    if args.skip_relax:
        relaxed_atoms = [AseAtomsAdaptor.get_atoms(s) for s in flat]
    elif flat:
        print(f"[bridge-edit:app] MatterSim relaxing {len(flat)} structures ...", flush=True)
        relaxed_atoms, _ = relax_structures_mattersim(
            flat, device=str(device), potential_path=args.mattersim_potential_path,
            fmax=0.05, max_n_steps=500)
    else:
        relaxed_atoms = []

    judge_items: list[dict] = []
    judge_back_row: list[int] = []
    for row_i, raw_s, atoms in zip(flat_meta, flat, relaxed_atoms):
        try:
            struct = AseAtomsAdaptor.get_structure(atoms)
            summary = _summary(struct)
            fe = total_energy_per_atom(atoms)
            if not (fe == fe):
                fe = 0.0
            judge_items.append({
                "row_id": rows[row_i]["row_id"],
                "prompt": rows[row_i]["user_prompt"],
                "formation_energy_per_atom": fe,
                # Realness gate: invalid raw output is forced to score 0 below (judge can't see broken crystals).
                "_valid": bool(_structurally_valid(raw_s)),
                **summary,
            })
            judge_back_row.append(row_i)
        except Exception:
            pass

    reset_failure_counts()
    print(f"[bridge-edit:app] dispatching {len(judge_items)} judge calls "
          f"(model={args.judge_model}, concurrency={args.judge_concurrency}) ...", flush=True)
    verdicts = asyncio.run(batch_judge(
        items=judge_items, build_messages_fn=build_app_consistency_messages,
        model=args.judge_model, concurrency=args.judge_concurrency)) if judge_items else []
    fc = get_failure_counts()
    if fc:
        print(f"[bridge-edit:app] judge failures: {fc}", flush=True)

    from collections import defaultdict
    judged: dict[str, list[int]] = defaultdict(list)
    examples = []
    n_gated_invalid = 0
    for item, verdict in zip(judge_items, verdicts):
        score = parse_score(verdict, default=0)
        if not item.get("_valid", True):          # REALNESS GATE: invalid structure → 0
            if score:
                n_gated_invalid += 1
            score = 0
        judged[item["row_id"]].append(score)
        examples.append({**item, "judge_score": score,
                         "judge_verdict": verdict.get("verdict") if verdict else None,
                         "judge_reason": verdict.get("reason") if verdict else None,
                         "extracted_application": verdict.get("extracted_application") if verdict else None})

    # Honest denominator: every prompt contributes; one with no judged structure scores 0.
    per_prompt = {r["row_id"]: judged.get(r["row_id"], [0]) for r in rows}
    per_prompt_mean = {rid: float(np.mean(s)) for rid, s in per_prompt.items()}
    overall_mean = float(np.mean(list(per_prompt_mean.values()))) if per_prompt_mean else 0.0
    score_2_rate = float(np.mean([s == 2 for ss in per_prompt.values() for s in ss])) \
        if per_prompt else 0.0
    n_scored = len(judge_items)

    headline = {
        "overall_consistency_mean_per_prompt": overall_mean,   # ∈ [0,2]  PRIMARY (validity-gated)
        "fraction_score_2": score_2_rate,
        "n_prompts_scored": len(per_prompt),
        "n_gated_invalid": n_gated_invalid,
        "n_judge_calls": len(judge_items),
        "n_judge_failures": int(sum(fc.values())) if fc else 0,
        "n_planner_parse_fail": n_planner_parse_fail,
        "judge_model": args.judge_model,
        "skip_relax": bool(args.skip_relax),
        "per_prompt_mean": per_prompt_mean,
        "prompt_version": args.prompt_version,
    }
    return headline, examples, n_scored


def _save_showcase_cifs(args, rows, gens_per_row, examples, target_comp_per_row):
    """Showcase dump: write input + first successful gen as CIF for the first --save_cifs_max successful rows. Read-only on metrics."""
    out = Path(args.save_cifs)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "saved_meta.jsonl"
    saved = 0
    task = args.task
    success_key = {
        "polymorph": "real_lower_energy",
        "atomtxt": "direction_correct",
        "doping": "correct_substitution",
        "strain": "strain_correct",
    }.get(task)

    # app: text-only, judge-scored. Dump rows with the highest per-prompt judge score.
    if task == "app":
        # Best judge score per row (_score_app emits one example per judged candidate).
        from collections import defaultdict as _dd
        best = _dd(lambda: -1)
        verdict_for = {}
        for e in examples:
            sc = int(e.get("judge_score", 0) or 0)
            if sc > best[e["row_id"]]:
                best[e["row_id"]] = sc
                verdict_for[e["row_id"]] = e
        ridx = {r["row_id"]: i for i, r in enumerate(rows)}
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        with meta_path.open("a") as mf:
            for rid, sc in ranked:
                if saved >= args.save_cifs_max or sc < 2:
                    break
                i = ridx.get(rid)
                if i is None or not gens_per_row[i]:
                    continue
                s = _to_struct(gens_per_row[i][0])
                if s is None or not _structurally_valid(s):
                    continue
                tag = f"almedit_app_{saved+1:02d}_{s.composition.reduced_formula}"
                gp = out / f"{tag}_gen.cif"
                s.to(filename=str(gp))
                v = verdict_for.get(rid, {})
                mf.write(json.dumps({
                    "file": gp.name, "task": "app", "row_id": rid,
                    "prompt": rows[i]["user_prompt"],
                    "planner_composition": target_comp_per_row[i],
                    "gen_formula": str(s.composition.reduced_formula),
                    "judge_score": sc,
                    "judge_application": v.get("extracted_application"),
                    "judge_reason": v.get("judge_reason"),
                    "why_success": "The gpt-4o-mini app-consistency judge scored this "
                                   "prompt's generations 2/2 (fully consistent with the "
                                   "requested application) on the relaxed structures. The "
                                   "saved CIF is a representative raw generated candidate "
                                   "for the prompt (same planner-fixed composition).",
                }) + "\n")
                saved += 1
        print(f"[save_cifs] app: wrote {saved} showcase example(s) → {out}", flush=True)
        return

    # structured tasks: per_candidate boolean success flag
    for e in examples:
        if saved >= args.save_cifs_max:
            break
        rid = e["row_id"]
        i = next((k for k, r in enumerate(rows) if r["row_id"] == rid), None)
        if i is None:
            continue
        gens = gens_per_row[i]
        if not gens:
            continue
        pcs = e.get("per_candidate_scores", [])
        # re-walk gens in the same skip-None order the scorers use so the index aligns with pcs
        cand_structs = []
        for g in gens:
            s = _to_struct(g)
            if s is None:
                continue
            cand_structs.append(s)
        win_j = None
        if task in ("describe", "ood"):  # prefer struct_match, else comp_match
            for j, sc in enumerate(pcs):
                if j < len(cand_structs) and sc.get("struct_match"):
                    win_j = j; break
            if win_j is None:
                for j, sc in enumerate(pcs):
                    if j < len(cand_structs) and sc.get("comp_match"):
                        win_j = j; break
        else:
            for j, sc in enumerate(pcs):
                if j < len(cand_structs) and sc.get(success_key):
                    win_j = j; break
        if win_j is None:
            continue
        gen_s = cand_structs[win_j]
        if not _structurally_valid(gen_s):
            continue

        tag = f"almedit_{task}_{saved+1:02d}_{gen_s.composition.reduced_formula}"
        meta = {"task": task, "row_id": rid, "prompt": rows[i]["user_prompt"],
                "gen_formula": str(gen_s.composition.reduced_formula)}
        input_struct = None
        if "input_atoms_struct" in rows[i]:
            input_struct = _ase_to_struct(rows[i]["input_atoms_struct"])
        if input_struct is not None:
            ip = out / f"{tag}_input.cif"
            input_struct.to(filename=str(ip))
            meta["input_file"] = ip.name
            meta["input_formula"] = str(input_struct.composition.reduced_formula)
        gp = out / f"{tag}_output.cif"
        gen_s.to(filename=str(gp))
        meta["output_file"] = gp.name
        meta["target_composition"] = target_comp_per_row[i] if i < len(target_comp_per_row) else None

        sc = pcs[win_j]
        if task == "polymorph":
            meta["why_success"] = (
                "Same composition as the input (composition_preserved), a genuinely "
                "DISTINCT structure (StructureMatcher != input), structurally valid, and "
                "MatterSim-relaxed energy/atom BELOW the relaxed input → a real "
                "lower-energy polymorph.")
            meta.update({k: sc.get(k) for k in
                         ("e_gen_per_atom", "e_input_per_atom",
                          "composition_preserved", "structurally_distinct")})
        elif task == "atomtxt":
            meta["why_success"] = (
                f"Requested to move {sc.get('property')} {sc.get('requested_direction')}; the "
                f"relaxed generated structure moved it the requested way "
                f"(direction_correct) at fixed composition.")
            meta.update({k: sc.get(k) for k in
                         ("property", "requested_direction", "prop_input", "prop_gen",
                          "delta_prop", "composition_preserved")})
        elif task == "doping":
            meta["why_success"] = (
                f"Dopant {rows[i].get('_dopant')} present, donor {rows[i].get('_donor')} "
                f"removed, ratio matches, valid, and distinct from a naive coords-identical "
                f"relabel → a real substitution.")
            meta.update({k: sc.get(k) for k in
                         ("dopant_present", "donor_removed", "ratio_match",
                          "distinct_from_relabel", "gen_formula")})
            meta["donor"] = rows[i].get("_donor"); meta["dopant"] = rows[i].get("_dopant")
        elif task == "strain":
            meta["why_success"] = (
                f"Doping correct AND the relaxed-equilibrium volume change hit the target "
                f"dV={sc.get('target_dv_pct')}% (realized {sc.get('realized_dv_pct')}%, "
                f"within tol).")
            meta.update({k: sc.get(k) for k in
                         ("target_dv_pct", "realized_dv_pct", "doping_correct",
                          "volume_match", "gen_formula")})
            meta["donor"] = rows[i].get("_donor"); meta["dopant"] = rows[i].get("_dopant")
        else:  # describe / ood
            meta["why_success"] = (
                "Text→structure recovery: generated reduced formula matches the GT and "
                + ("the geometry matches under DisorderedStructureMatcher (struct_match)."
                   if sc.get("struct_match") else "the composition matches (comp_match).") )
            meta["gt_formula"] = e.get("gt_formula")
            meta.update({k: sc.get(k) for k in ("comp_match", "struct_match")})

        with meta_path.open("a") as mf:
            mf.write(json.dumps(meta) + "\n")
        saved += 1
        print(f"[save_cifs] {task}: saved {tag} (row {rid})", flush=True)

    print(f"[save_cifs] {task}: wrote {saved} showcase example(s) → {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
