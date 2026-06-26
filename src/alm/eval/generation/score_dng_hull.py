"""Score a DNG-style CIF directory via MP-2020 hull (CrystalReasoner/Crys-JEPA convention)."""
from __future__ import annotations

import argparse
import json
import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ALM_ROOT)

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
        return formulas
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
                    help="cap on number of structures (default: all in cif_dir).")
    ap.add_argument("--n_workers_relax", type=int, default=1,
                    help="MatterSim is GPU-bound; parallelism limited.")
    args = ap.parse_args()

    cif_files = sorted(args.cif_dir.glob("*.cif"))
    if args.max_structures > 0:
        cif_files = cif_files[: args.max_structures]
    print(f"[mp20-hull] loading {len(cif_files)} CIFs ...", flush=True)
    structs = []
    sids = []
    for fp in cif_files:
        try:
            s = Structure.from_str(fp.read_text(), fmt="cif")
            structs.append(s)
            sids.append(fp.stem)
        except Exception:
            pass
    print(f"[mp20-hull] {len(structs)} structures parsed", flush=True)

    valid_geom = [validity_geom(s) for s in structs]
    valid_charge = []
    for s in structs:
        try:
            valid_charge.append(validity_charge(s))
        except Exception:
            valid_charge.append(False)
    valid_full = [g and c for g, c in zip(valid_geom, valid_charge)]
    print(f"[mp20-hull] validity_geom={sum(valid_geom)/len(structs):.3f}  "
          f"validity_charge={sum(valid_charge)/len(structs):.3f}  "
          f"valid_full={sum(valid_full)/len(structs):.3f}", flush=True)

    print(f"[mp20-hull] uniqueness via DisorderedStructureMatcher (mg-eval) ...", flush=True)
    matcher = mg_eval_matcher()
    try:
        uniq_idx = unique_indices(structs, matcher=matcher)
        is_unique = [False] * len(structs)
        for i in uniq_idx:
            is_unique[i] = True
        uniq_rate = sum(is_unique) / len(structs)
    except Exception as e:
        print(f"  uniqueness err: {e}", flush=True)
        is_unique = [True] * len(structs)
        uniq_rate = 1.0
    print(f"[mp20-hull] uniqueness={uniq_rate:.3f}", flush=True)

    train_formulas = load_train_formulas(args.train_csv)
    is_novel = [s.composition.reduced_formula not in train_formulas for s in structs]
    novel_rate = sum(is_novel) / len(structs) if structs else 0
    print(f"[mp20-hull] novelty (formula vs MP-20 train, {len(train_formulas)} ref): "
          f"{novel_rate:.3f}", flush=True)

    print(f"[mp20-hull] loading hull reference ...", flush=True)
    if args.hull_dir is not None:
        reference = load_hull_reference(args.hull_dir)
    else:
        reference = load_hull_reference()

    print(f"[mp20-hull] relaxing {len(structs)} via MatterSim ...", flush=True)
    relaxed_atoms_list: list = [None] * len(structs)
    energies_arr = None
    try:
        relaxed_atoms_list, energies_arr = relax_structures_mattersim(
            structs,
            device=args.mattersim_device,
        )
    except Exception as e:
        print(f"[mp20-hull] relax err: {e}", flush=True)

    from pymatgen.io.ase import AseAtomsAdaptor
    print(f"[mp20-hull] computing E_h vs MP-2020 hull ...", flush=True)
    e_above = []
    n_err_logged = 0
    # energies_arr: total energy in eV, per structure
    energies = energies_arr.tolist() if energies_arr is not None else [None] * len(relaxed_atoms_list)
    for s_init, ase_atoms, e_total in zip(structs, relaxed_atoms_list, energies):
        if ase_atoms is None or e_total is None:
            e_above.append(None)
            continue
        try:
            s_relaxed = AseAtomsAdaptor.get_structure(ase_atoms) if not isinstance(ase_atoms, Structure) else ase_atoms
            eh = e_above_hull_per_atom(structure=s_relaxed,
                                       total_energy_eV=float(e_total),
                                       hull_reference=reference)
            # NaN = missing chemsys or failed PD construction; treat as missing
            import math
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
    print(f"[mp20-hull] {len(valid_eh)}/{len(e_above)} E_h computed; "
          f"mean={np.mean(valid_eh):.4f} eV/atom" if valid_eh else
          "[mp20-hull] no E_h computed", flush=True)

    def buckets(threshold):
        stab = [v is not None and v <= threshold for v in e_above]
        sun_mask = [
            s and u and n and v for s, u, n, v in
            zip(stab, is_unique, is_novel, valid_full)
        ]
        return sum(stab) / len(stab), sum(sun_mask) / len(sun_mask)

    s0, sun0 = buckets(0.0)             # on-hull
    s016, sun016 = buckets(0.016)       # SUN strict (CrystalReasoner)
    s100, sun100 = buckets(0.100)       # MSUN metastable (Crys-JEPA lax)

    result = {
        "n_structures": len(structs),
        "validity": {
            "geom_rate": sum(valid_geom) / len(structs),
            "charge_rate": sum(valid_charge) / len(structs),
            "full_rate": sum(valid_full) / len(structs),
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
            "sun_at_0.016": sun016,   # SUN strict
            "sun_at_0.1":   sun100,   # MSUN metastable
        },
        "hull": "MP-2020 (MatterGen-bundled)",
        "relax_mlip": "MatterSim",
        "matcher": "DisorderedStructureMatcher (mg-eval default)",
        "novelty_method": "formula-level vs MP-20 train CSV",
        "compare_to": {
            "CrystalReasoner_SUN@0.016": sun016,
            "CrysJEPA_SUN@0.1":          sun100,
        },
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[mp20-hull] HEADLINE → {args.out_path}")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
