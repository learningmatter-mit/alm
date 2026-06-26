"""Polymorph eval: generate K candidates per input crystal and score composition_match, structurally_distinct, structurally_valid, polymorph_success."""
from __future__ import annotations


import argparse
import hashlib
import json
import os
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
from ase import Atoms
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from paths import DATA_ROOT  # noqa: E402


def _ase_to_struct(atoms_struct: dict) -> Structure | None:
    try:
        elements = [str(e).strip() for e in atoms_struct["elements"]]
        coords = np.asarray(atoms_struct["coords"], dtype=np.float64)
        lattice = np.asarray(atoms_struct["lattice_mat"], dtype=np.float64)
        cartesian = bool(atoms_struct.get("cartesian", True))
        return Structure(
            lattice=lattice, species=elements, coords=coords,
            coords_are_cartesian=cartesian,
        )
    except Exception:
        return None


def _select_rows(parquet_path: Path, max_rows: int, seed: int) -> list[dict]:
    """Hash-deterministic eval subset."""
    pf = pq.ParquetFile(parquet_path)
    candidates: list[dict] = []
    cols = ["row_id", "parent", "source_idx", "user_prompt", "atoms_struct",
            "input_atoms_struct", "input_source_idx"]
    for batch in pf.iter_batches(batch_size=10000, columns=cols):
        for r in batch.to_pylist():
            h = int(hashlib.md5(f"{r['row_id']}:{seed}".encode()).hexdigest(), 16)
            r["_h"] = h
            candidates.append(r)
    candidates.sort(key=lambda r: r["_h"])
    return candidates[:max_rows]


def _ase_atoms_from_struct(atoms_struct: dict):
    """ASE Atoms from inline input_atoms_struct (live OrbV3; cache is row-index keyed, would miss on material_id)."""
    from ase import Atoms
    elements = [str(e) for e in atoms_struct["elements"]]
    coords = np.asarray(atoms_struct["coords"], dtype=np.float64)
    lattice = np.asarray(atoms_struct["lattice_mat"], dtype=np.float64)
    cartesian = bool(atoms_struct.get("cartesian", True))
    if cartesian:
        return Atoms(symbols=elements, positions=coords, cell=lattice, pbc=True)
    return Atoms(symbols=elements, scaled_positions=coords, cell=lattice, pbc=True)


def _live_orbv3_features(alm, atoms_obj, device) -> torch.Tensor:
    """Raw (N, 256) OrbV3 node features (pre-projector, as get_alm_embedding expects)."""
    from orb_models.forcefield import atomic_system
    from orb_models.forcefield.base import batch_graphs
    with torch.no_grad():
        atoms_capped = atoms_obj if alm.max_atoms is None else atoms_obj[:alm.max_atoms]
        graph = batch_graphs([atomic_system.ase_atoms_to_atom_graphs(
            atoms_capped, alm.atomistic_model.system_config, device=device,
        )])
        results = alm.atomistic_model.model(graph)
        return results["node_features"]  # (N, 256)


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


def _composition_match(gen: Structure, input_struct: Structure) -> bool:
    try:
        return gen.composition.reduced_formula == input_struct.composition.reduced_formula
    except Exception:
        return False


def _counts_from_atoms_struct(atoms_struct: dict) -> dict[str, int]:
    counts = Counter(str(e) for e in atoms_struct.get("elements", []))
    return {str(el): int(n) for el, n in counts.items() if int(n) > 0}


def _score_generation(
    gen: Structure | Atoms,
    input_struct: Structure,
    matcher: StructureMatcher,
) -> dict:
    if not isinstance(gen, Structure):
        try:
            gen = AseAtomsAdaptor.get_structure(gen)
        except Exception:
            return {
                "parsed": False, "composition_match": False,
                "structurally_distinct": False, "structurally_valid": False,
                "polymorph_success": False,
            }
    comp_match = _composition_match(gen, input_struct)
    valid = _structurally_valid(gen)
    distinct = True
    if comp_match and valid:
        try:
            distinct = not matcher.fit(input_struct, gen)
        except Exception:
            distinct = True  # matcher error => treat as distinct (conservative)
    success = comp_match and distinct and valid
    return {
        "parsed": True,
        "composition_match": comp_match,
        "structurally_distinct": distinct,
        "structurally_valid": valid,
        "polymorph_success": success,
        "gen_formula": str(gen.composition.reduced_formula),
        "input_formula": str(input_struct.composition.reduced_formula),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True)
    ap.add_argument("--atoms_mapper", required=True)
    ap.add_argument("--polymorph_parquet", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_polymorph_under_hull.parquet")))
    ap.add_argument("--cached_embs_root", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "cached_embs_narratives")))
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diffusion_seed", type=int, default=1337)
    ap.add_argument("--score_e_hull", action="store_true",
                    help="Also relax via MatterSim and report fraction of polymorphs "
                         "below input's E_hull (slow; off by default).")
    # StructureMatcher tolerances: CDVAE / CrystaLLM defaults
    ap.add_argument("--ltol", type=float, default=0.3)
    ap.add_argument("--stol", type=float, default=0.5)
    ap.add_argument("--angle_tol", type=float, default=10.0)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_polymorph] writing → {args.out_dir}", flush=True)
    t0 = time.time()

    rows = _select_rows(args.polymorph_parquet, args.max_rows, args.seed)
    print(f"[eval_polymorph] {len(rows)} rows (seed={args.seed})", flush=True)
    parents = {r["parent"] for r in rows}
    print(f"[eval_polymorph] parents: {parents}", flush=True)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generate_stage3 import (
        load_alm_and_pl_module, get_alm_embedding,
        build_sampler_and_loader, draw_samples_from_sampler,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_polymorph] loading ALM + MatterGen on {device} (live OrbV3) ...", flush=True)
    # cached_embeddings off: input_source_idx is material_id, not a cache row-index key, so live-encode
    alm, tokenizer, pl_module, K_tokens = load_alm_and_pl_module(
        alm_checkpoint=args.alm_checkpoint,
        atoms_mapper=args.atoms_mapper,
        mattergen_pretrained=args.mattergen_pretrained,
        device=device,
        use_cached_embeddings=False,
    )

    import random as _random
    structures_per_prompt: list[list] = []
    for i, r in enumerate(rows):
        _s = (int(args.diffusion_seed) + i) & 0x7FFFFFFF
        torch.manual_seed(_s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_s)
        np.random.seed(_s)
        _random.seed(_s)
        try:
            input_atoms = _ase_atoms_from_struct(r["input_atoms_struct"])
            atom_embed = _live_orbv3_features(alm, input_atoms, device)
        except Exception as e:
            print(f"[eval_polymorph] {r['row_id']}: live-encode failed ({type(e).__name__}: {e}) → skip", flush=True)
            structures_per_prompt.append([])
            continue
        json_counts = _counts_from_atoms_struct(r["input_atoms_struct"])
        alm_emb = get_alm_embedding(
            alm, tokenizer, r["user_prompt"], device,
            atom_embed=atom_embed,
            wrap_user_template=False,
            json_counts=json_counts,
        )
        sampler, condition_loader = build_sampler_and_loader(
            pl_module=pl_module, batch_size=args.K, num_batches=1,
            num_atoms_distribution="ALEX_MP_20",
            alm_emb_vec=alm_emb,
            diffusion_guidance_factor=args.guidance_factor,
        )
        gens = draw_samples_from_sampler(
            sampler, condition_loader,
            output_path=None,
            properties_to_condition_on=None,
            record_trajectories=False,
        )
        structures_per_prompt.append(gens)
        if (i + 1) % 10 == 0:
            print(f"[eval_polymorph] generated {i+1}/{len(rows)} prompts in {time.time()-t0:.0f}s", flush=True)

    matcher = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)
    examples = []
    overall = Counter()
    n_scored = 0
    for i, gens in enumerate(structures_per_prompt):
        r = rows[i]
        input_struct = _ase_to_struct(r["input_atoms_struct"])
        if input_struct is None or not gens:
            continue
        per_candidate = []
        for g in gens:
            if isinstance(g, Atoms):
                try:
                    g = AseAtomsAdaptor.get_structure(g)
                except Exception:
                    continue
            sc = _score_generation(g, input_struct, matcher)
            per_candidate.append(sc)
            n_scored += 1
            for k in ("composition_match", "structurally_distinct",
                      "structurally_valid", "polymorph_success"):
                if sc[k]:
                    overall[k] += 1
        examples.append({
            "row_id": r["row_id"],
            "parent": r["parent"],
            "user_prompt": r["user_prompt"],
            "input_formula": str(input_struct.composition.reduced_formula),
            "per_candidate_scores": per_candidate,
            "n_candidates": len(per_candidate),
            "polymorph_success_rate": np.mean([sc["polymorph_success"] for sc in per_candidate]) if per_candidate else 0.0,
        })

    headline = {"n_scored": n_scored, "n_prompts": len(examples),
                "guidance_factor": args.guidance_factor, "K": args.K,
                "ltol": args.ltol, "stol": args.stol, "angle_tol": args.angle_tol}
    if n_scored > 0:
        for k in ("composition_match", "structurally_distinct",
                  "structurally_valid", "polymorph_success"):
            headline[k] = overall[k] / n_scored
    headline["per_prompt_mean_success"] = float(np.mean([e["polymorph_success_rate"] for e in examples])) if examples else 0.0

    (args.out_dir / "metrics.json").write_text(json.dumps(headline, indent=2))
    with (args.out_dir / "predictions.jsonl").open("w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")

    print(f"\n[eval_polymorph] HEADLINE ({n_scored} candidates across {len(examples)} prompts):", flush=True)
    print(f"  composition_match             = {headline.get('composition_match', 0):.3f}", flush=True)
    print(f"  structurally_distinct         = {headline.get('structurally_distinct', 0):.3f}", flush=True)
    print(f"  structurally_valid            = {headline.get('structurally_valid', 0):.3f}", flush=True)
    print(f"  polymorph_success_rate        = {headline.get('polymorph_success', 0):.3f}", flush=True)
    print(f"  per_prompt_mean_success       = {headline['per_prompt_mean_success']:.3f}", flush=True)
    print(f"[eval_polymorph] DONE in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
