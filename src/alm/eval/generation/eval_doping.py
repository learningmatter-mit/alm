"""Doping/substitution eval for the stage3a doping_strain bucket: generate K candidates per X->Y prompt and score substitution success."""
from __future__ import annotations


import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Pauling electronegativity.*")
warnings.filterwarnings("ignore", message=".*fractional coordinates.*")

import numpy as np
import pyarrow.parquet as pq
import torch
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from paths import DATA_ROOT  # noqa: E402

_PATTERNS = [
    r"replacing all ([A-Z][a-z]?) atoms with ([A-Z][a-z]?)",
    r"substitute ([A-Z][a-z]?) sites with ([A-Z][a-z]?)",
    r"Replace the ([A-Z][a-z]?) sublattice with ([A-Z][a-z]?)",
    r"Apply a ([A-Z][a-z]?)->\s*([A-Z][a-z]?) substitution",
    r"replacing all ([A-Z][a-z]?) sites with ([A-Z][a-z]?)",
]
_SUB_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))


def _parse_substitution(prompt: str) -> tuple[str, str] | None:
    """Return (donor, dopant) = (X, Y) from a "replace X with Y"-style prompt."""
    m = _SUB_RE.search(prompt)
    if not m:
        return None
    groups = m.groups()
    for i in range(0, len(groups), 2):
        if groups[i] is not None and groups[i + 1] is not None:
            return groups[i], groups[i + 1]
    return None


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
    """Hash-deterministic eval subset; keep only rows with parseable X->Y prompts."""
    pf = pq.ParquetFile(parquet_path)
    candidates: list[dict] = []
    cols = ["row_id", "parent", "source_idx", "user_prompt", "atoms_struct",
            "input_atoms_struct", "input_source_idx"]
    for batch in pf.iter_batches(batch_size=10000, columns=cols):
        for r in batch.to_pylist():
            sub = _parse_substitution(r["user_prompt"])
            if sub is None:
                continue
            r["_donor"], r["_dopant"] = sub
            h = int(hashlib.md5(f"{r['row_id']}:{seed}".encode()).hexdigest(), 16)
            r["_h"] = h
            candidates.append(r)
    candidates.sort(key=lambda r: r["_h"])
    return candidates[:max_rows]


def _load_orbv3_cache(cached_embs_root: Path, parents: set[str]) -> dict:
    out = {"memmap": {}, "idx": {}}
    for parent in parents:
        bin_p = cached_embs_root / parent / "embeddings" / "orb_v3_direct_20_omat_atom.flat.bin"
        idx_p = cached_embs_root / parent / "embeddings" / "orb_v3_direct_20_omat_atom.flat.idx.json"
        if not bin_p.exists() or not idx_p.exists():
            print(f"[eval_doping] WARN no cache for {parent} at {bin_p}", flush=True)
            continue
        with open(idx_p) as f:
            out["idx"][parent] = json.load(f)
        out["memmap"][parent] = np.memmap(bin_p, dtype=np.float32, mode="r").reshape(-1, 256)
    return out


def _input_atom_embed(cache, parent: str, input_idx, device) -> torch.Tensor | None:
    idx_map = cache["idx"].get(parent)
    mm = cache["memmap"].get(parent)
    if idx_map is None or mm is None:
        return None
    ent = idx_map.get(str(input_idx))
    if ent is None:
        return None
    off, length = int(ent[0]), int(ent[1])
    arr = np.asarray(mm[off:off + length], dtype=np.float32).copy()
    return torch.from_numpy(arr).to(device)


def _ase_atoms_from_struct(atoms_struct: dict):
    """Build ASE Atoms from inline input_atoms_struct for live OrbV3 (the disk cache keys by row-index but input_source_idx is a material_id, so it always missed)."""
    from ase import Atoms
    elements = [str(e) for e in atoms_struct["elements"]]
    coords = np.asarray(atoms_struct["coords"], dtype=np.float64)
    lattice = np.asarray(atoms_struct["lattice_mat"], dtype=np.float64)
    cartesian = bool(atoms_struct.get("cartesian", True))
    if cartesian:
        return Atoms(symbols=elements, positions=coords, cell=lattice, pbc=True)
    return Atoms(symbols=elements, scaled_positions=coords, cell=lattice, pbc=True)


def _live_orbv3_features(alm, atoms_obj, device) -> torch.Tensor:
    """Raw (N, 256) OrbV3 node features, stopping before the projector (get_alm_embedding expects pre-projector features)."""
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
    """Lite validity check: shortest interatomic distance > min_dist (Angstrom)."""
    try:
        if len(struct) < 1:
            return False
        d = struct.distance_matrix
        n = d.shape[0]
        d = d + np.eye(n) * 1e6
        return float(d.min()) > min_dist
    except Exception:
        return False


def _score_generation(
    gen: Structure | Atoms,
    donor: str,
    dopant: str,
    input_struct: Structure,
) -> dict:
    if not isinstance(gen, Structure):
        try:
            gen = AseAtomsAdaptor.get_structure(gen)
        except Exception:
            return {
                "parsed": False, "dopant_present": False, "donor_removed": False,
                "ratio_match": False, "structurally_valid": False,
                "full_substitution": False, "correct_substitution": False,
            }
    gen_elems = [str(s.specie.symbol) for s in gen]
    in_elems = [str(s.specie.symbol) for s in input_struct]
    n_gen = max(1, len(gen_elems))
    n_in = max(1, len(in_elems))
    gen_count = Counter(gen_elems)
    in_count = Counter(in_elems)
    dopant_present = gen_count.get(dopant, 0) > 0
    donor_removed = gen_count.get(donor, 0) == 0
    # match if dopant fraction in gen is within 0.1 abs of donor fraction in input
    target_frac = in_count.get(donor, 0) / n_in
    gen_frac = gen_count.get(dopant, 0) / n_gen
    ratio_match = abs(gen_frac - target_frac) <= 0.10
    valid = _structurally_valid(gen)
    full_sub = dopant_present and donor_removed
    correct = full_sub and ratio_match and valid
    return {
        "parsed": True,
        "dopant_present": dopant_present,
        "donor_removed": donor_removed,
        "ratio_match": ratio_match,
        "structurally_valid": valid,
        "full_substitution": full_sub,
        "correct_substitution": correct,
        "gen_formula": str(gen.composition.reduced_formula),
        "input_formula": str(input_struct.composition.reduced_formula),
    }


def _target_counts_after_substitution(input_atoms_struct: dict, donor: str, dopant: str) -> dict[str, int]:
    counts = Counter(str(e) for e in input_atoms_struct.get("elements", []))
    if donor in counts:
        counts[dopant] += counts.pop(donor)
    return {str(el): int(n) for el, n in counts.items() if int(n) > 0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True)
    ap.add_argument("--atoms_mapper", required=True)
    ap.add_argument("--doping_parquet", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_doping_strain_sub1M.parquet")))
    ap.add_argument("--cached_embs_root", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "cached_embs_narratives")))
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_rows", type=int, default=100)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="Eval-set hash seed; pick a value never used in training.")
    ap.add_argument("--diffusion_seed", type=int, default=1337)
    args = ap.parse_args()

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_doping] writing → {args.out_dir}", flush=True)
    t0 = time.time()

    rows = _select_rows(args.doping_parquet, args.max_rows, args.seed)
    print(f"[eval_doping] {len(rows)} rows (seed={args.seed})", flush=True)
    by_pair = Counter((r["_donor"], r["_dopant"]) for r in rows)
    print(f"[eval_doping] top substitutions: {by_pair.most_common(8)}", flush=True)
    parents = {r["parent"] for r in rows}
    print(f"[eval_doping] loading OrbV3 caches for parents: {parents}", flush=True)
    cache = _load_orbv3_cache(args.cached_embs_root, parents)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from generate_stage3 import (
        load_alm_and_pl_module, get_alm_embedding,
        build_sampler_and_loader, draw_samples_from_sampler,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_doping] loading ALM + MatterGen on {device} ...", flush=True)
    alm, tokenizer, pl_module, K_tokens = load_alm_and_pl_module(
        alm_checkpoint=args.alm_checkpoint,
        atoms_mapper=args.atoms_mapper,
        mattergen_pretrained=args.mattergen_pretrained,
        device=device,
        use_cached_embeddings=False,  # live-encode input from input_atoms_struct
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
            print(f"[eval_doping] {r['row_id']}: live-encode failed ({type(e).__name__}: {e}) → skip", flush=True)
            structures_per_prompt.append([])
            continue
        json_counts = _target_counts_after_substitution(
            r["input_atoms_struct"], r["_donor"], r["_dopant"]
        )
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
            print(f"[eval_doping] generated {i+1}/{len(rows)} prompts in {time.time()-t0:.0f}s", flush=True)

    examples = []
    per_prompt_rates: dict[str, list[float]] = defaultdict(list)
    overall = Counter()
    n_scored = 0
    for i, gens in enumerate(structures_per_prompt):
        r = rows[i]
        donor, dopant = r["_donor"], r["_dopant"]
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
            sc = _score_generation(g, donor, dopant, input_struct)
            per_candidate.append(sc)
            n_scored += 1
            for k in ("dopant_present", "donor_removed", "ratio_match",
                      "structurally_valid", "full_substitution", "correct_substitution"):
                if sc[k]:
                    overall[k] += 1
        examples.append({
            "row_id": r["row_id"],
            "parent": r["parent"],
            "user_prompt": r["user_prompt"],
            "donor": donor,
            "dopant": dopant,
            "input_formula": str(input_struct.composition.reduced_formula),
            "per_candidate_scores": per_candidate,
            "n_candidates": len(per_candidate),
            "full_substitution_rate": np.mean([sc["full_substitution"] for sc in per_candidate]) if per_candidate else 0.0,
            "correct_substitution_rate": np.mean([sc["correct_substitution"] for sc in per_candidate]) if per_candidate else 0.0,
        })

    headline = {}
    if n_scored > 0:
        for k in ("dopant_present", "donor_removed", "ratio_match",
                  "structurally_valid", "full_substitution", "correct_substitution"):
            headline[k] = overall[k] / n_scored
    headline["n_scored"] = n_scored
    headline["n_prompts"] = len(examples)
    headline["guidance_factor"] = args.guidance_factor
    headline["K"] = args.K

    headline["per_prompt_mean_full_sub"] = float(np.mean([e["full_substitution_rate"] for e in examples])) if examples else 0.0
    headline["per_prompt_mean_correct_sub"] = float(np.mean([e["correct_substitution_rate"] for e in examples])) if examples else 0.0

    (args.out_dir / "metrics.json").write_text(json.dumps(headline, indent=2))
    with (args.out_dir / "predictions.jsonl").open("w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")

    print(f"\n[eval_doping] HEADLINE ({n_scored} candidates across {len(examples)} prompts):", flush=True)
    print(f"  dopant_present                = {headline.get('dopant_present', 0):.3f}", flush=True)
    print(f"  donor_removed                 = {headline.get('donor_removed', 0):.3f}", flush=True)
    print(f"  ratio_match                   = {headline.get('ratio_match', 0):.3f}", flush=True)
    print(f"  structurally_valid            = {headline.get('structurally_valid', 0):.3f}", flush=True)
    print(f"  full_substitution_rate        = {headline.get('full_substitution', 0):.3f}", flush=True)
    print(f"  correct_substitution_rate     = {headline.get('correct_substitution', 0):.3f}", flush=True)
    print(f"  per_prompt_mean_full_sub      = {headline['per_prompt_mean_full_sub']:.3f}", flush=True)
    print(f"  per_prompt_mean_correct_sub   = {headline['per_prompt_mean_correct_sub']:.3f}", flush=True)
    print(f"[eval_doping] DONE in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
