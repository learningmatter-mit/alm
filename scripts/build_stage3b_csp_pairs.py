"""Rewrite a Stage 3a pairs.parquet to CSP-style (composition, space group) prompts."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm


PROMPT_FLAVORS = ("minimal", "rich_v1", "rich_v2")


def _det_pick(row_id: str, options: tuple[str, ...]) -> str:
    h = int(hashlib.sha1(row_id.encode()).hexdigest(), 16)
    return options[h % len(options)]


def _struct_from_atoms(a: dict) -> Structure | None:
    elements = [s.strip() for s in a["elements"]]
    coords = np.asarray(a["coords"], dtype=float)
    lattice_mat = np.asarray(a["lattice_mat"], dtype=float)
    if coords.size == 0 or lattice_mat.shape != (3, 3):
        return None
    return Structure(
        lattice=Lattice(lattice_mat),
        species=elements,
        coords=coords,
        coords_are_cartesian=bool(a.get("cartesian", False)),
    )


def _sg_info(struct: Structure) -> tuple[str, str]:
    try:
        sga = SpacegroupAnalyzer(struct, symprec=0.01)
        return sga.get_space_group_symbol(), sga.get_crystal_system()
    except Exception:
        return "P1", "triclinic"


def _make_prompt(flavor: str, formula: str, sg_symbol: str, crystal_system: str) -> str:
    if flavor == "minimal":
        return (
            f"Generate a crystal structure with formula {formula}, "
            f"space group {sg_symbol}."
        )
    if flavor == "rich_v1":
        return (
            f"The material with the formula {formula} has a "
            f"{crystal_system} crystal structure with the space group symbol "
            f"{sg_symbol}."
        )
    if flavor == "rich_v2":
        return (
            f"{formula} is a {crystal_system} crystalline material "
            f"with a space group symbol {sg_symbol}."
        )
    raise ValueError(f"unknown flavor {flavor!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", required=True,
                   help="Input pairs.parquet (Stage 3a build).")
    p.add_argument("--out_path", required=True,
                   help="Output pairs_csp.parquet — same schema, CSP-style prompts.")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--symprec", type=float, default=0.01,
                   help="Pymatgen SpacegroupAnalyzer symprec (matches eval_csp.py).")
    args = p.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.in_path)
    schema = pf.schema_arrow
    writer = pq.ParquetWriter(out_path, schema, compression="zstd")

    n_rewrite = 0
    n_p1_fallback = 0
    n_rows = pf.metadata.num_rows
    n_batches = (n_rows + args.batch_size - 1) // args.batch_size
    for batch in tqdm(pf.iter_batches(batch_size=args.batch_size), total=n_batches, desc="CSP pairs"):
        cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        new_prompts = []
        for i in range(batch.num_rows):
            row_id = cols["row_id"][i]
            atoms = cols["atoms_struct"][i]
            struct = _struct_from_atoms(atoms)
            if struct is None:
                # Rare: keep original prompt when atoms_struct won't parse.
                new_prompts.append(cols["user_prompt"][i])
                continue
            formula = struct.composition.reduced_formula
            sg_symbol, crystal_system = _sg_info(struct)
            if sg_symbol == "P1":
                n_p1_fallback += 1
            flavor = _det_pick(row_id, PROMPT_FLAVORS)
            new_prompts.append(_make_prompt(flavor, formula, sg_symbol, crystal_system))
            n_rewrite += 1
        cols["user_prompt"] = new_prompts
        out_batch = pa.RecordBatch.from_pydict(cols, schema=schema)
        writer.write_batch(out_batch)

    writer.close()
    print(f"[csp-pairs] wrote {out_path}: {n_rewrite}/{n_rows} rows rewritten "
          f"({n_p1_fallback} P1-fallback on SG analysis), schema preserved.")


if __name__ == "__main__":
    main()
