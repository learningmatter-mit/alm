"""Score a directory of generated CIFs against the MP-2020 hull (CrystalReasoner / Crys-JEPA conventions).

Rates divide by every CIF in the directory, or by the number of requested generations when it is
larger, so unparseable CIFs and generations that never produced a CIF count as invalid, not unique,
not novel and not stable. The requested count is --n_expected when given; otherwise it is the sum of
n_expected over the summary_shard*.json files that gen_dng_bridge / gen_dng_native write (looked up
in --cif_dir, then its parent).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

from pymatgen.core import Structure  # noqa: E402

from structure_metrics import (  # noqa: E402
    validity_geom,
    validity_charge,
    unique_indices,
    relax_structures_mattersim,
    e_above_hull_per_atom,
    load_hull_reference,
)

# Disordered matcher for U+N; CDVAE Ordered is used for CSP M@K, not DNG.
from mattergen.evaluation.utils.structure_matcher import (  # noqa: E402
    DefaultDisorderedStructureMatcher,
)
from paths import DATA_ROOT  # noqa: E402


def mg_eval_matcher():
    return DefaultDisorderedStructureMatcher()


def load_train_formulas(train_csv: Path) -> set[str]:
    import csv
    formulas = set()
    if not train_csv.exists():
        raise FileNotFoundError(f"novelty reference not found: {train_csv} (pass --train_csv)")
    with open(train_csv) as f:
        for row in csv.DictReader(f):
            cif = row.get("cif")
            if cif:
                try:
                    s = Structure.from_str(cif, fmt="cif")
                    formulas.add(s.composition.reduced_formula)
                except Exception:
                    pass
    return formulas


def n_expected_from_summaries(cif_dir: Path) -> tuple[int, list[Path]]:
    """Sum n_expected over summary_shard*.json in cif_dir, else in its parent (the generators' --out_dir, whose cifs/ subdir holds the CIFs); (0, []) when none are found."""
    for d in (cif_dir, cif_dir.parent):
        files = sorted(d.glob("summary_shard*.json"))
        if files:
            total = 0
            for fp in files:
                total += int(json.loads(fp.read_text()).get("n_expected", 0))
            return total, files
    return 0, []


def dng_rates(n_total: int, valid_full: list[bool], is_unique: list[bool],
              is_novel: list[bool], e_above: list[float | None], threshold: float) -> tuple[float, float]:
    """(stable rate, SUN rate) at an E_hull threshold over n_total CIFs; missing E_hull and unparseable CIFs count as failures."""
    if not n_total:
        return 0.0, 0.0
    stab = [v is not None and v <= threshold for v in e_above]
    sun = [s and u and n and v for s, u, n, v in zip(stab, is_unique, is_novel, valid_full)]
    return sum(stab) / n_total, sum(sun) / n_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif_dir", type=Path, required=True)
    ap.add_argument("--out_path", type=Path, required=True)
    ap.add_argument("--hull_dir", type=Path, default=None,
                    help="MP-2020 hull dir (default: MatterGen bundled).")
    ap.add_argument("--train_csv", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "eval_data/csp/mp_20/train.csv")))
    ap.add_argument("--mattersim_device", default="cuda")
    ap.add_argument("--max_structures", type=int, default=-1,
                    help="Maximum number of structures to score (default: all in cif_dir).")
    ap.add_argument("--n_expected", type=int, default=None,
                    help="Number of requested generations; missing CIFs count as failures (default: the summed "
                         "n_expected of summary_shard*.json in --cif_dir or its parent, else the number of CIFs found).")
    args = ap.parse_args()
    if args.n_expected is None:
        args.n_expected, summary_files = n_expected_from_summaries(args.cif_dir)
        if summary_files:
            print(f"[dng-hull] n_expected={args.n_expected} from {len(summary_files)} summary file(s) "
                  f"in {summary_files[0].parent}", flush=True)

    cif_files = sorted(args.cif_dir.glob("*.cif"))
    if args.max_structures > 0:
        cif_files = cif_files[: args.max_structures]
    print(f"[dng-hull] loading {len(cif_files)} CIFs ...", flush=True)
    structs = []
    sids = []
    for fp in cif_files:
        try:
            s = Structure.from_str(fp.read_text(), fmt="cif")
            structs.append(s)
            sids.append(fp.stem)
        except Exception:
            pass
    n_unparsed = len(cif_files) - len(structs)
    print(f"[dng-hull] {len(structs)} structures parsed ({n_unparsed} unparseable CIFs)", flush=True)

    valid_geom = [validity_geom(s) for s in structs]
    valid_charge = []
    for s in structs:
        try:
            valid_charge.append(validity_charge(s))
        except Exception:
            valid_charge.append(False)
    valid_full = [g and c for g, c in zip(valid_geom, valid_charge)]
    # Denominator for every rate below (--n_expected is ignored when --max_structures caps the set).
    n_total = len(cif_files) if args.max_structures > 0 else max(len(cif_files), args.n_expected)
    n_missing = n_total - len(cif_files)
    if n_total == 0:
        raise SystemExit(f"no CIFs found in {args.cif_dir}")
    print(f"[dng-hull] validity_geom={sum(valid_geom)/n_total:.3f}  "
          f"validity_charge={sum(valid_charge)/n_total:.3f}  "
          f"valid_full={sum(valid_full)/n_total:.3f}", flush=True)

    print(f"[dng-hull] uniqueness via DisorderedStructureMatcher (mg-eval) ...", flush=True)
    matcher = mg_eval_matcher()
    uniq_idx = unique_indices(structs, matcher=matcher)
    is_unique = [False] * len(structs)
    for i in uniq_idx:
        is_unique[i] = True
    uniq_rate = sum(is_unique) / n_total
    print(f"[dng-hull] uniqueness={uniq_rate:.3f}", flush=True)

    train_formulas = load_train_formulas(args.train_csv)
    is_novel = [s.composition.reduced_formula not in train_formulas for s in structs]
    novel_rate = sum(is_novel) / n_total
    print(f"[dng-hull] novelty (formula vs MP-20 train, {len(train_formulas)} ref): "
          f"{novel_rate:.3f}", flush=True)

    print(f"[dng-hull] loading hull reference ...", flush=True)
    if args.hull_dir is not None:
        reference = load_hull_reference(args.hull_dir)
    else:
        reference = load_hull_reference()

    print(f"[dng-hull] relaxing {len(structs)} via MatterSim ...", flush=True)
    # A relaxation failure raises: writing a result with every E_h missing would look like a real 0.0.
    relaxed_atoms_list, energies_arr = relax_structures_mattersim(
        structs,
        device=args.mattersim_device,
    )

    from pymatgen.io.ase import AseAtomsAdaptor
    print(f"[dng-hull] computing E_h vs MP-2020 hull ...", flush=True)
    e_above = []
    n_err_logged = 0
    # energies_arr: total energy in eV, per structure
    energies = energies_arr.tolist()
    for s_init, ase_atoms, e_total in zip(structs, relaxed_atoms_list, energies):
        if ase_atoms is None or e_total is None:
            e_above.append(None)
            continue
        try:
            s_relaxed = AseAtomsAdaptor.get_structure(ase_atoms) if not isinstance(ase_atoms, Structure) else ase_atoms
            eh = e_above_hull_per_atom(structure=s_relaxed,
                                       total_energy_eV=float(e_total),
                                       hull_reference=reference)
            # NaN: chemical system missing from the hull or phase diagram failed; counts as not stable.
            if math.isnan(eh):
                if n_err_logged < 3:
                    print(f"  [hull-err] NaN for {s_relaxed.composition.reduced_formula}", flush=True)
                    n_err_logged += 1
                e_above.append(None)
            else:
                e_above.append(eh)
        except Exception as ex:
            if n_err_logged < 3:
                print(f"  [hull-err] {type(ex).__name__}: {ex}", flush=True)
                n_err_logged += 1
            e_above.append(None)
    valid_eh = [v for v in e_above if v is not None]
    print(f"[dng-hull] {len(valid_eh)}/{len(e_above)} E_h computed; "
          f"mean={np.mean(valid_eh):.4f} eV/atom" if valid_eh else
          "[dng-hull] no E_h computed", flush=True)

    def buckets(threshold):
        return dng_rates(n_total, valid_full, is_unique, is_novel, e_above, threshold)

    s0, sun0 = buckets(0.0)       # on hull
    s016, sun016 = buckets(0.016)  # stable (SUN)
    s100, sun100 = buckets(0.100)  # metastable (MSUN)

    result = {
        "n_structures": len(structs),
        "n_cifs": len(cif_files),
        "n_expected": n_total,
        "n_cif_unparsed": n_unparsed,
        "n_missing_generations": n_missing,
        "gen_failed_rate": (n_unparsed + n_missing) / n_total,
        "validity": {
            "geom_rate": sum(valid_geom) / n_total,
            "charge_rate": sum(valid_charge) / n_total,
            "full_rate": sum(valid_full) / n_total,
        },
        "uniqueness": uniq_rate,
        "novelty_by_formula_vs_mp20_train": novel_rate,
        "novelty_ref_size": len(train_formulas),
        "n_eh_computed": len(valid_eh),
        "mean_e_above_hull": float(np.mean(valid_eh)) if valid_eh else None,
        "median_e_above_hull": float(np.median(valid_eh)) if valid_eh else None,
        "stability_rates": {
            "stable_at_0":     s0,
            "stable_at_0.016": s016,
            "stable_at_0.1":   s100,
        },
        "sun_rates": {
            "sun_at_0":     sun0,
            "sun_at_0.016": sun016,   # SUN
            "sun_at_0.1":   sun100,   # MSUN
        },
        "hull": "MP-2020 (MatterGen-bundled)",
        "relax_mlip": "MatterSim",
        "matcher": "DisorderedStructureMatcher (mg-eval default)",
        "novelty_method": "formula-level vs MP-20 train CSV",
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[dng-hull] wrote {args.out_path}")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
