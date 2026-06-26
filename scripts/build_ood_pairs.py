"""Build pairs_ood.parquet by filling LLM-generated templates with per-row metadata."""
from __future__ import annotations


import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from paths import DATA_ROOT


PLACEHOLDERS_ALLOWED = {
    "formula", "sg_symbol", "crystal_system", "density", "n_atoms",
    "n_elements", "first_element", "elements_csv", "volume", "n_atoms_int",
}

# Extract the space group symbol from narrative prose; handles common SG-symbol shapes.
SG_PATTERN = re.compile(
    r"space\s*group\s*(?:symbol\s*(?:of|is)?|is)?\s*([A-Z][\w\-/_]+)",
    re.IGNORECASE,
)

CRYSTAL_SYS_PATTERN = re.compile(
    r"\b(triclinic|monoclinic|orthorhombic|tetragonal|trigonal|hexagonal|cubic|"
    r"rhombohedral)\b",
    re.IGNORECASE,
)


def _formula_from_elements(elements: list[str]) -> str:
    counts = Counter(elements)
    parts = []
    for el in sorted(counts.keys()):
        n = counts[el]
        parts.append(f"{el}{n if n > 1 else ''}")
    return "".join(parts)


def _meta_from_row(atoms_struct: dict, narrative: str) -> dict:
    elements = [str(e).strip() for e in atoms_struct["elements"]]
    if not elements:
        return {}
    counts = Counter(elements)
    n_atoms = len(elements)
    unique_els = sorted(counts.keys())
    formula = _formula_from_elements(elements)
    elements_csv = ", ".join(unique_els)
    first_element = unique_els[0]
    lattice = np.asarray(atoms_struct["lattice_mat"], dtype=np.float64)
    try:
        volume = float(abs(np.linalg.det(lattice)))
    except Exception:
        volume = None
    if volume and volume > 0:
        # density g/cm^3: sum of atomic masses (amu) * 1.66054 / volume (A^3)
        from ase.data import atomic_masses, atomic_numbers
        try:
            mass_amu = sum(atomic_masses[atomic_numbers[el]] for el in elements)
            density = mass_amu * 1.66054 / volume
        except Exception:
            density = None
    else:
        density = None
    sg_match = SG_PATTERN.search(narrative or "")
    sg_symbol = sg_match.group(1).strip().rstrip(".,;") if sg_match else None
    sys_match = CRYSTAL_SYS_PATTERN.search(narrative or "")
    crystal_system = sys_match.group(1).lower() if sys_match else None
    return {
        "formula": formula,
        "sg_symbol": sg_symbol,
        "crystal_system": crystal_system,
        "density": f"{density:.2f}" if density is not None else None,
        "n_atoms": str(n_atoms),
        "n_atoms_int": n_atoms,
        "n_elements": str(len(unique_els)),
        "first_element": first_element,
        "elements_csv": elements_csv,
        "volume": f"{volume:.2f}" if volume is not None else None,
    }


def _template_placeholders(t: str) -> set[str]:
    return set(re.findall(r"\{([a-z_]+)\}", t))


def _pick_template_for_row(row_id: str, templates_by_req: dict[frozenset, list[str]],
                           available: set[str]) -> str | None:
    """Deterministically pick a template whose required placeholders are all available."""
    h = int(hashlib.md5(row_id.encode()).hexdigest(), 16)
    compat = [reqs for reqs in templates_by_req if reqs.issubset(available)]
    if not compat:
        return None
    bucket = compat[h % len(compat)]
    pool = templates_by_req[bucket]
    return pool[(h >> 8) % len(pool)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs_parquet", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs.parquet")))
    ap.add_argument("--templates_jsonl", type=Path,
                    default=Path("helper_scripts/eval_prompts/ood_templates.jsonl"))
    ap.add_argument("--out_path", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_ood.parquet")))
    ap.add_argument("--batch_size", type=int, default=10000)
    args = ap.parse_args()

    print(f"[build] loading templates from {args.templates_jsonl}")
    templates: list[str] = []
    with open(args.templates_jsonl) as f:
        for line in f:
            try:
                obj = json.loads(line)
                t = obj.get("template")
                if t and isinstance(t, str):
                    templates.append(t)
            except Exception:
                pass
    print(f"[build] {len(templates)} templates loaded")
    if not templates:
        raise SystemExit("no templates")

    templates_by_req: dict[frozenset, list[str]] = {}
    bucket_sizes = Counter()
    for t in templates:
        reqs = frozenset(_template_placeholders(t))
        templates_by_req.setdefault(reqs, []).append(t)
        bucket_sizes[reqs] += 1
    print(f"[build] {len(templates_by_req)} distinct placeholder-set buckets")
    print(f"[build] top buckets:")
    for reqs, n in sorted(bucket_sizes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n:6d}  required={set(reqs)}")

    pf = pq.ParquetFile(args.pairs_parquet)
    print(f"[build] reading {pf.metadata.num_rows:,} rows from {args.pairs_parquet}")

    schema = pf.schema_arrow
    writer = pq.ParquetWriter(args.out_path, schema, compression="snappy")

    n_kept = 0
    n_dropped_no_template = 0
    n_dropped_unfillable = 0
    n_seen = 0
    for batch in tqdm(pf.iter_batches(batch_size=args.batch_size), total=(pf.metadata.num_rows + args.batch_size - 1)//args.batch_size, desc="rows"):
        batch_dict = batch.to_pydict()
        new_prompts = []
        keep_idx = []
        for i, row_id in enumerate(batch_dict["row_id"]):
            n_seen += 1
            atoms_struct = batch_dict["atoms_struct"][i]
            narrative = batch_dict["narrative"][i]
            meta = _meta_from_row(atoms_struct, narrative)
            available = {k for k, v in meta.items() if v is not None}
            template = _pick_template_for_row(row_id, templates_by_req, available)
            if template is None:
                n_dropped_no_template += 1
                continue
            try:
                filled = template.format(**meta)
            except Exception:
                n_dropped_unfillable += 1
                continue
            new_prompts.append(filled)
            keep_idx.append(i)
            n_kept += 1

        if not keep_idx:
            continue
        out_dict = {col: [batch_dict[col][i] for i in keep_idx] for col in schema.names}
        out_dict["user_prompt"] = new_prompts
        writer.write_table(pa.Table.from_pydict(out_dict, schema=schema))

    writer.close()
    print(f"\n[build] kept   : {n_kept:,}")
    print(f"[build] dropped (no compatible template): {n_dropped_no_template:,}")
    print(f"[build] dropped (template format error) : {n_dropped_unfillable:,}")
    print(f"[build] wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
