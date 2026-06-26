"""Resolve the MP convex-hull reference for E_hull scoring (MatterGen-bundled, else mp-api fallback)."""
from __future__ import annotations


import argparse
import os
import pickle
import sys
from pathlib import Path

from paths import DATA_ROOT

REPO_ROOT = Path(__file__).resolve().parent.parent
MATTERGEN_BUNDLED = REPO_ROOT / "external" / "mattergen" / "data-release" / "alex-mp"
DEST_ROOT = Path(os.path.join(DATA_ROOT, "eval_data/mp_hull"))


def resolve_bundled(variant: str) -> Path | None:
    fname = f"reference_{variant}correction.gz"
    candidate = MATTERGEN_BUNDLED / fname
    if not candidate.exists():
        return None
    # On-disk file may still be a ~130-byte Git LFS pointer, not the real data.
    try:
        with open(candidate, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            print(
                f"[fetch] {candidate} is a Git LFS pointer, not the actual data.\n"
                f"        Run:\n"
                f"          cd <repo>/external/mattergen && \\\n"
                f"            git lfs install --local && git lfs pull\n"
                f"        Then re-run this script.",
                file=sys.stderr,
            )
            return None
    except Exception:
        pass
    return candidate


def link_bundled(src: Path, dest_dir: Path, force: bool) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists() and not force:
        print(f"[fetch] already present: {dest}")
        return dest
    if dest.exists():
        dest.unlink()
    dest.symlink_to(src.resolve())
    print(f"[fetch] linked {src} → {dest}")
    return dest


def fetch_via_mp_api(out_path: Path) -> Path:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print(
            "[fetch] MP_API_KEY not set and no MatterGen-bundled reference found. "
            "Either set MP_API_KEY (https://next-gen.materialsproject.org/api) or "
            "run `bash external/setup_mattergen.sh` to ensure the submodule is "
            "checked out with its data-release/ directory.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception:
        print(
            "[fetch] mp-api not installed. Install via `pip install mp-api>=0.4` "
            "or use the MatterGen-bundled file (preferred).",
            file=sys.stderr,
        )
        sys.exit(3)
    print("[fetch] querying Materials Project for thermo entries...")
    with MPRester(api_key) as mpr:
        entries = mpr.thermo.search(thermo_types=["GGA_GGA+U"])
        # ThermoDoc -> ComputedStructureEntry.
        cse_list = []
        for d in entries:
            cse = d.entries.get("GGA_GGA+U")
            if cse is None:
                cse = next(iter(d.entries.values()))
            cse_list.append(cse)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(cse_list, f)
    print(f"[fetch] wrote {len(cse_list)} ComputedStructureEntry objects to {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["MP2020", "TRI2024"], default="MP2020",
                   help="Which MatterGen-bundled hull to use. MP2020 matches the "
                        "Materials Project default and is what most papers use.")
    p.add_argument("--force", action="store_true",
                   help="Re-link/re-fetch even if destination already exists.")
    p.add_argument("--out_dir", type=Path, default=DEST_ROOT,
                   help="Where to place the resolved reference.")
    args = p.parse_args()

    bundled = resolve_bundled(args.variant)
    if bundled is not None:
        print(f"[fetch] using MatterGen-bundled {args.variant}: {bundled}")
        link_bundled(bundled, args.out_dir, force=args.force)
        # Sentinel so structure_metrics.load_hull_reference() picks the right path.
        marker = args.out_dir / "preferred.txt"
        marker.write_text(f"reference_{args.variant}correction.gz\n")
        print(f"[fetch] preferred marker written: {marker}")
        return 0

    print(f"[fetch] no bundled reference at {MATTERGEN_BUNDLED} — trying mp-api...")
    out_pkl = args.out_dir / "reference_mp_api.pkl"
    if out_pkl.exists() and not args.force:
        print(f"[fetch] already present: {out_pkl}")
        return 0
    fetch_via_mp_api(out_pkl)
    marker = args.out_dir / "preferred.txt"
    marker.write_text("reference_mp_api.pkl\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
