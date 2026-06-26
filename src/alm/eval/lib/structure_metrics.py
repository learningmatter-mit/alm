"""Structure-metric helpers wrapping MatterGen's evaluation submodule, pinned to CDVAE/CrystaLLM tolerances."""
from __future__ import annotations


import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mattergen.evaluation.utils.structure_matcher import OrderedStructureMatcher  # noqa: E402
from mattergen.evaluation.utils.dataset_matcher import (  # noqa: E402
    get_matches,
    get_unique,
)

from paths import DATA_ROOT  # noqa: E402


# Matcher: CDVAE/CrystaLLM tolerances (looser than MatterGen's 0.2/0.3/5).
CDVAE_TOLS = dict(ltol=0.3, stol=0.5, angle_tol=10)


def cdvae_matcher() -> OrderedStructureMatcher:
    return OrderedStructureMatcher(**CDVAE_TOLS)


_DEFAULT = None


def _default_matcher() -> OrderedStructureMatcher:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = cdvae_matcher()
    return _DEFAULT


def validity_geom(s: Structure, min_dist: float = 0.5) -> bool:
    """True iff geometrically valid: no atoms closer than min_dist, positive volume, sane angles."""
    try:
        if s.volume <= 0:
            return False
        if not s.is_valid(tol=min_dist):
            return False
        a, b, c = s.lattice.angles
        if not all(0 < ang < 180 for ang in (a, b, c)):
            return False
        return True
    except Exception:
        return False


def validity_charge(s: Structure) -> bool:
    """True iff a charge-neutral oxidation-state assignment exists (smact); matches CDVAE."""
    try:
        import smact  # noqa: F401
        from smact.screening import pauling_test
    except Exception:
        return False
    try:
        comp = Composition(s.composition.reduced_formula)
        symbols = [str(el) for el in comp.elements]
        counts = [int(comp[el]) for el in comp.elements]
        from smact import element_dictionary, neutral_ratios
        elem_objs = [element_dictionary().get(sym) for sym in symbols]
        if any(e is None for e in elem_objs):
            return False
        ox_combos = [e.oxidation_states for e in elem_objs]
        from itertools import product
        for ox_states in product(*ox_combos):
            if neutral_ratios(ox_states, stoichs=[(c,) for c in counts])[0]:
                electronegs = [e.pauling_eneg for e in elem_objs if e.pauling_eneg is not None]
                if len(electronegs) != len(elem_objs):
                    return True  # missing eneg: accept charge-balance alone
                if pauling_test(ox_states, electronegs, symbols):
                    return True
        return False
    except Exception:
        return False


def validity_full(s: Structure) -> dict[str, bool]:
    g = validity_geom(s)
    c = validity_charge(s)
    return {"geom": g, "charge": c, "both": g and c}


def match_one(
    gen: Structure, ref: Structure,
    matcher: OrderedStructureMatcher | None = None,
) -> tuple[bool, float | None]:
    """Returns (matched, rmse); rmse is None if not matched."""
    m = matcher or _default_matcher()
    try:
        if not m.fit(gen, ref):
            return False, None
        rms = m.get_rms_dist(gen, ref)
        if rms is None:
            return False, None
        return True, float(rms[0])  # pymatgen returns (rms, max_dist); report rms
    except Exception:
        return False, None


def match_many(
    gens: Sequence[Structure], ref: Structure,
    matcher: OrderedStructureMatcher | None = None,
) -> dict:
    """n=1 (first gen) and n=K (any gen, min rmse) match aggregates against one reference."""
    m = matcher or _default_matcher()
    out = {
        "n": len(gens),
        "matched_n1": False, "rmse_n1": None,
        "matched_nK": False, "rmse_nK": None,
        "match_idx": [],
    }
    rmses = []
    for i, g in enumerate(gens):
        matched, rmse = match_one(g, ref, m)
        if matched:
            out["match_idx"].append(i)
            rmses.append(rmse)
            if i == 0:
                out["matched_n1"] = True
                out["rmse_n1"] = rmse
    if rmses:
        out["matched_nK"] = True
        out["rmse_nK"] = float(min(rmses))
    return out


def composition_set(s: Structure) -> set[str]:
    return {str(el) for el in s.composition.elements}


def composition_match_ratio(s: Structure, target_elements: Iterable[str]) -> float:
    """Fraction of target elements present in the generated structure."""
    target = {str(t) for t in target_elements}
    if not target:
        return 0.0
    have = composition_set(s)
    return len(target & have) / len(target)


def density_g_per_cm3(s: Structure) -> float:
    """Mass density in g/cm^3 (pymatgen's built-in)."""
    try:
        return float(s.density)
    except Exception:
        return float("nan")


def unique_indices(
    structures: Sequence[Structure],
    matcher: OrderedStructureMatcher | None = None,
) -> list[int]:
    """Indices of unique structures in a batch via MatterGen's get_unique (O(N^2))."""
    m = matcher or _default_matcher()
    return get_unique(m, list(structures))


def novel_mask(
    structures: Sequence[Structure],
    reference_structures: Sequence[Structure],
    matcher: OrderedStructureMatcher | None = None,
) -> np.ndarray:
    """Boolean mask: True for structures NOT matched in the reference dataset (O(N x M))."""
    m = matcher or _default_matcher()
    matches = get_matches(m, list(structures), list(reference_structures))
    novel = np.ones(len(structures), dtype=bool)
    for idx, ref_hits in matches.items():
        if ref_hits:
            novel[idx] = False
    return novel


def novel_mask_by_formula(
    structures: Sequence[Structure],
    reference_structures: Sequence[Structure],
    matcher: OrderedStructureMatcher | None = None,
) -> np.ndarray:
    """Faster novelty: only compare against references with matching reduced formula."""
    m = matcher or _default_matcher()
    by_formula: dict[str, list[Structure]] = {}
    for r in reference_structures:
        rf = r.composition.reduced_formula
        by_formula.setdefault(rf, []).append(r)
    novel = np.ones(len(structures), dtype=bool)
    for i, s in enumerate(structures):
        rf = s.composition.reduced_formula
        candidates = by_formula.get(rf, [])
        for r in candidates:
            try:
                if m.fit(s, r):
                    novel[i] = False
                    break
            except Exception:
                continue
    return novel


def relax_structures_mattersim(
    inputs: Sequence[Structure | Atoms],
    device: str = "cuda",
    potential_path: str | None = None,
    fmax: float = 0.05,
    max_n_steps: int = 500,
    output_extxyz: str | Path | None = None,
) -> tuple[list[Atoms], np.ndarray]:
    """Relax a batch with MatterSim; returns (relaxed_atoms, total_energies in eV/cell)."""
    from mattergen.evaluation.utils.relaxation import relax_atoms
    atoms_list: list[Atoms] = []
    for x in inputs:
        if isinstance(x, Structure):
            atoms_list.append(AseAtomsAdaptor.get_atoms(x))
        elif isinstance(x, Atoms):
            atoms_list.append(x)
        else:
            raise TypeError(f"unsupported input type: {type(x)}")
    # Skip degenerate cells: a near-zero lattice blows up pymatgen's neighbor list to
    # ~1e17 bytes (MemoryError) and crashes the whole batch; return NaN placeholders instead.
    def _degenerate(a) -> bool:
        try:
            if not np.isfinite(a.get_positions()).all():
                return True
            cell = np.asarray(a.get_cell(), dtype=float)
            if not np.isfinite(cell).all():
                return True
            vol = abs(float(a.get_volume()))
            n = max(1, len(a))
            lengths = np.asarray(a.cell.lengths(), dtype=float)
            if vol <= 1e-3 or (vol / n) < 0.5 or (vol / n) > 1.0e4:
                return True
            if (lengths < 0.5).any() or (lengths > 500.0).any():
                return True
        except Exception:
            return True  # unreadable -> treat as degenerate
        return False

    ok_idx = [i for i, a in enumerate(atoms_list) if not _degenerate(a)]
    n_bad = len(atoms_list) - len(ok_idx)
    if n_bad:
        print(f"[relax] skipping {n_bad}/{len(atoms_list)} geometrically-degenerate "
              f"cell(s) (NaN energy; would MemoryError pymatgen's neighbor list)", flush=True)

    relaxed_full: list[Atoms] = list(atoms_list)
    energies_full = np.full(len(atoms_list), float("nan"), dtype=float)

    if ok_idx:
        sub = [atoms_list[i] for i in ok_idx]
        try:
            r, e = relax_atoms(
                sub, device=device, potential_load_path=potential_path,
                fmax=fmax, max_n_steps=max_n_steps,
                output_path=str(output_extxyz) if output_extxyz else None,
            )
        except Exception as batch_exc:
            # Batch poisoned by a bad cell: fall back to per-structure relax, loading the
            # Potential ONCE (per-row from_checkpoint would be a multi-minute reload each).
            print(f"[relax] batch relax failed ({type(batch_exc).__name__}: {batch_exc}); "
                  f"retrying per-structure with a SHARED potential (no per-row reload)", flush=True)
            from mattersim.applications.batch_relax import BatchRelaxer
            from mattersim.forcefield.potential import Potential
            _pot = Potential.from_checkpoint(
                device=device, load_path=potential_path, load_training_state=False)
            r, e = [], []
            for a in sub:
                try:
                    _relaxer = BatchRelaxer(
                        potential=_pot, filter="EXPCELLFILTER",
                        fmax=fmax, max_n_steps=max_n_steps)
                    trajs = _relaxer.relax([a])
                    ra = [t[-1] for t in trajs.values()][0]
                    r.append(ra); e.append(float(ra.info["total_energy"]))
                except Exception:
                    r.append(a); e.append(float("nan"))
            e = np.asarray(e, dtype=float)
        for j, i in enumerate(ok_idx):
            relaxed_full[i] = r[j]
            energies_full[i] = float(e[j])

    return relaxed_full, energies_full


def total_energy_per_atom(atoms: Atoms) -> float:
    """Per-atom total energy from atoms.info['total_energy']; NaN if missing."""
    e = atoms.info.get("total_energy")
    if e is None:
        return float("nan")
    return float(e) / max(1, len(atoms))


DEFAULT_HULL_DIR = Path(os.path.join(DATA_ROOT, "eval_data/mp_hull"))


def load_hull_reference(hull_dir: Path | str = DEFAULT_HULL_DIR):
    """Load the convex-hull reference for E_hull scoring: a ReferenceDataset (.gz) or list of ComputedStructureEntry (.pkl)."""
    hull_dir = Path(hull_dir)
    if not hull_dir.exists():
        raise FileNotFoundError(
            f"hull dir {hull_dir} does not exist — run scripts/fetch_mp_hull.py"
        )

    preferred = hull_dir / "preferred.txt"
    if preferred.exists():
        fname = preferred.read_text().strip()
        path = hull_dir / fname
    else:
        for candidate in ("reference_MP2020correction.gz",
                          "reference_TRI2024correction.gz",
                          "reference_mp_api.pkl"):
            if (hull_dir / candidate).exists():
                path = hull_dir / candidate
                break
        else:
            raise FileNotFoundError(
                f"no hull reference found in {hull_dir} — run scripts/fetch_mp_hull.py"
            )

    if path.suffix == ".gz":
        # Detect a Git-LFS pointer left by a clone without `git lfs pull`.
        with open(path, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            raise FileNotFoundError(
                f"{path} is a Git LFS pointer, not the actual gzipped reference. "
                f"Run `cd external/mattergen && git lfs install --local && git lfs pull` "
                f"to materialize it."
            )
        from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
        return LMDBGZSerializer().deserialize(str(path))
    if path.suffix == ".pkl":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"unknown hull reference format: {path}")


def hull_entries_for_chemsys(reference, chemsys_elements: Iterable[str]) -> list:
    """ComputedStructureEntry list covering all sub-chemsystems of the requested elements, deduped on entry_id."""
    elems = sorted({str(e) for e in chemsys_elements})
    if hasattr(reference, "entries_by_chemsys"):
        # entries_by_chemsys key is a "-"-joined SORTED element list.
        out = []
        seen_ids = set()
        from itertools import combinations
        for k in range(1, len(elems) + 1):
            for combo in combinations(elems, k):
                key = "-".join(combo)
                bucket = reference.entries_by_chemsys.get(key, [])
                for e in bucket:
                    eid = getattr(e, "entry_id", None) or id(e)
                    if eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                    out.append(e)
        return out
    return [e for e in reference
            if set(str(el) for el in e.composition.elements).issubset(set(elems))]


def e_above_hull_per_atom(
    structure: Structure,
    total_energy_eV: float,
    hull_reference,
) -> float:
    """E_hull (eV/atom) for one structure; NaN if the chemsys is missing or the hull can't be built."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    pd_entry = PDEntry(structure.composition, total_energy_eV)
    if hasattr(hull_reference, "get_e_above_hull"):
        try:
            return float(hull_reference.get_e_above_hull(pd_entry))
        except Exception:
            return float("nan")
    elems = [str(el) for el in structure.composition.elements]
    relevant = hull_entries_for_chemsys(hull_reference, elems)
    if not relevant:
        return float("nan")
    try:
        pd = PhaseDiagram(list(relevant) + [pd_entry])
        return float(pd.get_e_above_hull(pd_entry))
    except Exception:
        return float("nan")


def _smoke():
    from pymatgen.core import Lattice
    s = Structure(
        Lattice.cubic(4.2),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    print("validity_geom:", validity_geom(s))
    print("validity_charge:", validity_charge(s))
    print("composition_set:", composition_set(s))
    print("density:", density_g_per_cm3(s))
    print("match_one self vs self:", match_one(s, s))


if __name__ == "__main__":
    _smoke()
