"""Build pairs_atomtxt.parquet: atom-input + text-prompt -> structure-target directional pairs."""
from __future__ import annotations


import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from paths import DATA_ROOT


# Per-parent source column -> canonical property name (oqmd has no volume).
PARENT_PROPS = {
    "dft_3d": {
        "formation_energy": "formation energy per atom (eV/atom)",
        "band_gap": "band gap (eV)",
        "density": "density (g/cm³)",
        "volume": "volume (Å³)",
    },
    "mp_3d_2020": {
        "formation_energy": "formation energy per atom (eV/atom)",
        "band_gap": "band gap (eV)",
        "density": "density (g/cm³)",
        "volume": "volume (Å³)",
    },
    "aflow2": {
        "formation_energy": "formation energy per atom (eV/atom)",
        "band_gap": "band gap (eV)",
        "density": "density (g/cm³)",
        "volume": "volume (Å³)",
    },
    "oqmd": {
        "formation_energy": "_oqmd_delta_e",
        "band_gap": "_oqmd_band_gap",
        "density": "density (g/cm3)",
    },
}


# One template per (property, direction) slot is picked per pair via deterministic hash.
PROMPT_TEMPLATES = {
    ("formation_energy", "lower"): [
        "Generate a more thermodynamically stable version of this material.",
        "Design a structure with the same elements but lower formation energy.",
        "Build a more stable polymorph of this material.",
        "Make a version of this with stronger atomic binding (lower formation energy per atom).",
    ],
    ("formation_energy", "higher"): [
        "Generate a metastable variant of this material with higher formation energy.",
        "Design a less thermodynamically favored polymorph of this material.",
    ],
    ("band_gap", "higher"): [
        "Generate a wider-bandgap version of this material.",
        "Design a more electronically insulating variant with the same elements.",
        "Build a version of this material suitable as a wider-bandgap semiconductor.",
        "Make a polymorph with a larger electronic band gap.",
    ],
    ("band_gap", "lower"): [
        "Generate a narrower-bandgap version of this material.",
        "Design a more electronically conductive variant with the same elements.",
        "Build a version of this with reduced band gap, closer to metallic behavior.",
        "Make a polymorph with a smaller electronic band gap.",
    ],
    ("density", "higher"): [
        "Generate a denser version of this material.",
        "Design a more close-packed polymorph with the same elements.",
        "Build a higher-density variant of this material.",
    ],
    ("density", "lower"): [
        "Generate a less dense version of this material.",
        "Design a more open-framework polymorph with the same elements.",
        "Build a lower-density variant of this material.",
    ],
    ("volume", "larger"): [
        "Generate a version of this material with a larger unit-cell volume.",
        "Design a polymorph with expanded unit cell.",
    ],
    ("volume", "smaller"): [
        "Generate a version of this material with a smaller unit-cell volume.",
        "Design a polymorph with a more compact unit cell.",
    ],
}


# Minimum relative change for a property delta to count as meaningful.
DELTA_THRESHOLDS = {
    "formation_energy": 0.05,
    "band_gap": 0.20,           # plus an absolute >=0.3 eV floor enforced below
    "density": 0.10,
    "volume": 0.10,
}


def _is_meaningful(prop: str, val_a: float, val_b: float) -> tuple[bool, str]:
    """Return (is_meaningful, direction) where direction is higher/lower/larger/smaller."""
    if val_a is None or val_b is None:
        return False, ""
    if not (val_a == val_a) or not (val_b == val_b):  # NaN
        return False, ""
    delta = val_b - val_a
    if abs(val_a) < 1e-6:
        rel = abs(delta) > DELTA_THRESHOLDS[prop]  # absolute fallback when val_a near 0
    else:
        rel = abs(delta) / abs(val_a) > DELTA_THRESHOLDS[prop]
    if prop == "band_gap":
        rel = rel and abs(delta) >= 0.3
    if not rel:
        return False, ""
    if prop == "volume":
        return True, "larger" if delta > 0 else "smaller"
    return True, "higher" if delta > 0 else "lower"


def _load_per_parent_metadata(narratives_root: Path) -> dict:
    """Build parent -> list of per-row dicts with source_idx, elements_key, and property values."""
    index: dict[str, list] = {}
    for parent, prop_map in PARENT_PROPS.items():
        p = narratives_root / f"{parent}_gpt_narratives.parquet"
        if not p.exists():
            print(f"[load] skip {parent} (not found at {p})")
            continue
        cols_present = ["atoms"] + list(prop_map.values())
        pf = pq.ParquetFile(p)
        actual_cols = [c for c in cols_present if c in pf.schema_arrow.names]
        rows = []
        idx = 0
        for batch in tqdm(pf.iter_batches(batch_size=10000, columns=actual_cols),
                          total=(pf.metadata.num_rows + 9999) // 10000,
                          desc=f"scan-{parent}"):
            data = batch.to_pylist()
            for r in data:
                a = r.get("atoms")
                if a is None or a.get("elements") is None:
                    rows.append(None)
                    idx += 1
                    continue
                elements = sorted({str(e).strip() for e in a["elements"]})
                if not elements:
                    rows.append(None)
                    idx += 1
                    continue
                props = {}
                for canonical, source_col in prop_map.items():
                    v = r.get(source_col) if source_col in actual_cols else None
                    try:
                        props[canonical] = float(v) if v is not None else None
                    except (TypeError, ValueError):
                        props[canonical] = None
                rows.append({
                    "source_idx": idx,
                    "elements_key": tuple(elements),
                    "props": props,
                })
                idx += 1
        index[parent] = rows
        print(f"[load] {parent}: {sum(1 for r in rows if r is not None):,} rows with metadata")
    return index


def _formula_from_atoms_struct(atoms_struct: dict) -> str:
    elements = [str(e).strip() for e in atoms_struct.get("elements", [])]
    if not elements:
        return ""
    counts = Counter(elements)
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs_parquet", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs.parquet")))
    ap.add_argument("--narratives_root", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "GPT-Narratives-for-Materials")))
    ap.add_argument("--out_path", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_atomtxt.parquet")))
    ap.add_argument("--max_pairs_per_cluster", type=int, default=10,
                    help="Cap pairs per (parent, element-set) cluster; k(k-1) grows fast.")
    ap.add_argument("--max_total_pairs", type=int, default=300000,
                    help="Cap on total pairs written. Subsampled randomly if needed.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[main] loading metadata from {args.narratives_root}")
    parent_index = _load_per_parent_metadata(args.narratives_root)

    print(f"[main] clustering by (parent, element-set)")
    clusters: dict[tuple, list] = defaultdict(list)
    for parent, rows in parent_index.items():
        for r in rows:
            if r is None:
                continue
            clusters[(parent, r["elements_key"])].append(r)
    cluster_sizes = [len(v) for v in clusters.values()]
    print(f"[main] {len(clusters):,} clusters total")
    print(f"[main]   cluster size distribution: "
          f"min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
          f"mean={sum(cluster_sizes)/len(cluster_sizes):.2f}")
    print(f"[main]   clusters with size>=2 (paired-eligible): "
          f"{sum(1 for s in cluster_sizes if s>=2):,}")

    rng = random.Random(args.seed)
    print(f"[main] generating directional pairs (cap {args.max_pairs_per_cluster}/cluster)")
    raw_pairs: list[dict] = []
    for (parent, elements_key), members in tqdm(clusters.items(), desc="pairs"):
        if len(members) < 2:
            continue
        local: list[dict] = []
        for i in range(len(members)):
            for j in range(len(members)):
                if i == j:
                    continue
                a = members[i]
                b = members[j]
                for prop in PARENT_PROPS[parent].keys():
                    val_a = a["props"].get(prop)
                    val_b = b["props"].get(prop)
                    ok, direction = _is_meaningful(prop, val_a, val_b)
                    if not ok:
                        continue
                    local.append({
                        "parent": parent,
                        "input_source_idx": a["source_idx"],
                        "target_source_idx": b["source_idx"],
                        "elements_key": list(elements_key),
                        "prop": prop,
                        "direction": direction,
                        "val_a": val_a,
                        "val_b": val_b,
                    })
        if len(local) > args.max_pairs_per_cluster:
            local = rng.sample(local, args.max_pairs_per_cluster)
        raw_pairs.extend(local)
    print(f"[main] raw pair candidates: {len(raw_pairs):,}")

    if args.max_total_pairs > 0 and len(raw_pairs) > args.max_total_pairs:
        rng.shuffle(raw_pairs)
        raw_pairs = raw_pairs[:args.max_total_pairs]
        print(f"[main] capped to {args.max_total_pairs:,} (random subsample)")

    # Surface (prop, direction) distribution to flag mode-collapse risk.
    dist = Counter((r["prop"], r["direction"]) for r in raw_pairs)
    print(f"[main] (prop, direction) distribution:")
    for (prop, direction), n in dist.most_common():
        print(f"  {prop:20s} {direction:8s}  {n:>8,}")

    # Build (parent, source_idx) -> row lookup for atoms_struct.
    print(f"[main] loading pairs.parquet for atoms_struct lookup")
    pf_pairs = pq.ParquetFile(args.pairs_parquet)
    pair_lookup: dict[tuple, dict] = {}
    for batch in tqdm(pf_pairs.iter_batches(batch_size=10000),
                      total=(pf_pairs.metadata.num_rows + 9999) // 10000,
                      desc="lookup"):
        for r in batch.to_pylist():
            pair_lookup[(r["parent"], r["source_idx"])] = r
    print(f"[main] lookup table size: {len(pair_lookup):,}")

    # Output schema = pairs.parquet schema plus input_atoms_struct and input_source_idx.
    print(f"[main] building output rows")
    pairs_schema = pf_pairs.schema_arrow
    out_schema = pa.schema(
        list(pairs_schema) + [
            pa.field("input_atoms_struct", pairs_schema.field("atoms_struct").type),
            pa.field("input_source_idx", pa.int64()),
        ]
    )

    output_rows = []
    for p in raw_pairs:
        in_row = pair_lookup.get((p["parent"], p["input_source_idx"]))
        out_row = pair_lookup.get((p["parent"], p["target_source_idx"]))
        if in_row is None or out_row is None:
            continue
        key = (p["prop"], p["direction"])
        if key not in PROMPT_TEMPLATES:
            continue
        templates = PROMPT_TEMPLATES[key]
        h = hash((p["parent"], p["input_source_idx"], p["target_source_idx"]))
        prompt = templates[h % len(templates)]

        # Row is the target (out_row); prepend <atoms> so the input-side OrbV3 splice lands.
        new_row = dict(out_row)
        new_row["row_id"] = f"atomtxt-{p['parent']}-{p['input_source_idx']}-to-{p['target_source_idx']}-{p['prop']}-{p['direction']}"
        new_row["user_prompt"] = "<atoms>\n" + prompt
        new_row["input_atoms_struct"] = in_row["atoms_struct"]
        new_row["input_source_idx"] = int(p["input_source_idx"])
        new_row["narrative"] = ""  # unused downstream for this bucket
        output_rows.append(new_row)

    print(f"[main] resolved output rows: {len(output_rows):,}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out_path, out_schema, compression="snappy")
    BATCH = 5000
    for i in tqdm(range(0, len(output_rows), BATCH), desc="write"):
        chunk = output_rows[i:i + BATCH]
        out_dict = {col: [] for col in out_schema.names}
        for r in chunk:
            for col in out_schema.names:
                out_dict[col].append(r[col])
        writer.write_table(pa.Table.from_pydict(out_dict, schema=out_schema))
    writer.close()
    print(f"[main] wrote {len(output_rows):,} rows to {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
