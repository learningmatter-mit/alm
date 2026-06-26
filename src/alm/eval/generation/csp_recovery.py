"""CSP eval: direct CrystaLLM head-to-head on MP-20 and MPTS-52 via CDVAE matcher."""
from __future__ import annotations


import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # alm/

from runs import run_dir, write_run  # noqa: E402
from structure_metrics import (  # noqa: E402
    cdvae_matcher,
    match_many,
    validity_full,
)
from paths import DATA_ROOT  # noqa: E402


CSV_COLUMN_CANDIDATES = ("cif_string", "cif", "structure", "structure_cif")
DEFAULT_BENCH_ROOT = Path(os.path.join(DATA_ROOT, "eval_data/csp"))


def _resolve_sg_numbers(sg_symbols: list[str]) -> list[int]:
    """SG symbol to integer 1..230 via pymatgen; failures map to 0 (unconditional)."""
    from pymatgen.symmetry.groups import SpaceGroup
    out = []
    for sym in sg_symbols:
        try:
            n = SpaceGroup(sym).int_number
            out.append(int(n) if 1 <= n <= 230 else 0)
        except Exception:
            out.append(0)
    return out


def _counts_from_structure(struct: Structure) -> dict[str, int]:
    comp = struct.composition
    return {str(el): int(round(comp[el])) for el in comp.elements}


def _composition_count_vec_from_counts(counts: dict[str, int]) -> torch.Tensor:
    from ase.data import atomic_numbers as _ase_z
    v = torch.zeros(101, dtype=torch.float32)
    for sym, n in counts.items():
        z = int(_ase_z.get(str(sym), 0))
        if 1 <= z < v.numel():
            v[z] = float(n)
    return v


def _reduced_z_counts(counts: dict[str, int]) -> dict[int, int]:
    from ase.data import atomic_numbers as _ase_z
    from functools import reduce as _reduce
    from math import gcd

    clean = {str(sym): int(n) for sym, n in counts.items() if int(n) > 0}
    if not clean:
        return {}
    g = max(_reduce(gcd, clean.values()), 1)
    return {int(_ase_z[sym]): int(n // g) for sym, n in clean.items()}


def _safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_test_rows(benchmark: str, max_rows: int = -1, bench_root: Path | None = None):
    """Yield per-row dicts from a CDVAE-format benchmark's test split."""
    bench_root = bench_root or DEFAULT_BENCH_ROOT
    test_csv = bench_root / benchmark / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"missing {test_csv} — run scripts/download_csp_benchmarks.sh"
        )
    with open(test_csv) as f:
        reader = csv.DictReader(f)
        cif_col = next((c for c in CSV_COLUMN_CANDIDATES if c in reader.fieldnames), None)
        if cif_col is None:
            raise ValueError(
                f"no recognized CIF column in {test_csv}; "
                f"have {reader.fieldnames}; expected one of {CSV_COLUMN_CANDIDATES}"
            )
        for i, row in enumerate(reader):
            if max_rows > 0 and i >= max_rows:
                break
            cif_str = row[cif_col]
            row_id = row.get("material_id") or row.get("id") or f"{benchmark}-{i}"
            try:
                struct = CifParser.from_str(cif_str).parse_structures(primitive=False)[0]
            except Exception as exc:
                print(f"[csp] skip {row_id}: CIF parse failed ({exc})")
                continue
            try:
                sg_analyzer = SpacegroupAnalyzer(struct)
                sg_symbol = sg_analyzer.get_space_group_symbol()
                crystal_system = sg_analyzer.get_crystal_system()
            except Exception:
                sg_symbol = "P1"
                crystal_system = "triclinic"
            formula = row.get("pretty_formula") or struct.composition.reduced_formula
            elements = sorted({str(el) for el in struct.composition.elements})
            yield {
                "row_id": row_id,
                "formula": formula,
                "sg_symbol": sg_symbol,
                "crystal_system": crystal_system,
                "ref_structure": struct,
                "formation_energy_per_atom": _safe_float(row.get("formation_energy_per_atom")),
                "band_gap": _safe_float(row.get("band_gap")),
                "e_above_hull": _safe_float(row.get("e_above_hull")),
                "density": float(struct.density),
                "elements": elements,
            }


# Prompt templates: `minimal` is CrystaLLM-style; rich_v1/v2/v3 mirror the
# dft_3d / mp_3d_2020 / oqmd GPT-Narratives prose patterns seen at training time.

def _band_gap_descriptor(bg) -> str:
    if bg is None:
        return ""
    if bg <= 0.05:
        return "indicating that it is a metal"
    if bg < 1.0:
        return "indicating that it is a narrow-gap semiconductor"
    if bg < 3.0:
        return "indicating that it is a semiconductor"
    return "indicating that it is a wide-gap insulator"


def _stability_sentence(e_hull) -> str:
    if e_hull is None:
        return ""
    if e_hull <= 0.001:
        return "The material is considered stable."
    if e_hull <= 0.1:
        return "The material is considered metastable."
    return "The material is considered unstable."


def make_prompt_minimal(row: dict) -> str:
    """CrystaLLM-style baseline: terse formula + space group only."""
    return (
        f"Generate a crystal structure with formula {row['formula']}, "
        f"space group {row['sg_symbol']}."
    )


def make_prompt_rich_v1(row: dict) -> str:
    """dft_3d-style: single rich paragraph, formula + sg + properties woven in."""
    parts = [
        f"The material with the formula {row['formula']} has a "
        f"{row['crystal_system']} crystal system with a space group symbol of "
        f"{row['sg_symbol']}."
    ]
    if row["density"] is not None:
        parts.append(f"It has a density of {row['density']:.3f} g/cm³.")
    if row["e_above_hull"] is not None:
        parts.append(
            f"The energy above the hull is {row['e_above_hull']:.4f} eV/atom."
        )
    if row["formation_energy_per_atom"] is not None:
        parts.append(
            f"The formation energy per atom is {row['formation_energy_per_atom']:.4f} eV/atom."
        )
    if row["band_gap"] is not None:
        bg_desc = _band_gap_descriptor(row["band_gap"])
        if bg_desc:
            parts.append(f"It has a band gap of {row['band_gap']:.4f} eV, {bg_desc}.")
        else:
            parts.append(f"It has a band gap of {row['band_gap']:.4f} eV.")
    stab = _stability_sentence(row.get("e_above_hull"))
    if stab:
        parts.append(stab)
    return " ".join(parts)


def make_prompt_rich_v2(row: dict) -> str:
    """mp_3d_2020-style: formula-first, compact property prose."""
    parts = [
        f"{row['formula']} is a {row['crystal_system']} crystalline material "
        f"with a space group symbol {row['sg_symbol']}."
    ]
    if row["formation_energy_per_atom"] is not None:
        parts.append(
            f"Its formation energy per atom is {row['formation_energy_per_atom']:.4f} eV."
        )
    if row["e_above_hull"] is not None:
        parts.append(
            f"The energy above the hull is {row['e_above_hull']:.4f} eV/atom."
        )
    if row["band_gap"] is not None:
        bg_desc = _band_gap_descriptor(row["band_gap"])
        if bg_desc:
            parts.append(
                f"The band gap of the material is {row['band_gap']:.4f} eV, {bg_desc}."
            )
        else:
            parts.append(f"The band gap of the material is {row['band_gap']:.4f} eV.")
    if row["density"] is not None:
        parts.append(
            f"The density of the material is {row['density']:.3f} grams per cubic centimeter."
        )
    return " ".join(parts)


def make_prompt_rich_v3(row: dict) -> str:
    """oqmd-style: header + bulleted Key Properties block."""
    elements_phrase = ", ".join(row["elements"][:-1]) + (
        f" and {row['elements'][-1]}" if len(row["elements"]) > 1 else row["elements"][0]
    )
    header = (
        f"The material under consideration is a {row['crystal_system']} compound "
        f"with the chemical formula {row['formula']}, space group {row['sg_symbol']}.\n\n"
        f"Key Properties:"
    )
    bullets = []
    if row["formation_energy_per_atom"] is not None:
        bullets.append(
            f"- Formation energy per atom: {row['formation_energy_per_atom']:.4f} eV/atom."
        )
    if row["band_gap"] is not None:
        bullets.append(f"- Band gap: {row['band_gap']:.4f} eV.")
    if row["e_above_hull"] is not None:
        bullets.append(
            f"- Energy above hull per atom: {row['e_above_hull']:.4f} eV/atom."
        )
    if row["density"] is not None:
        bullets.append(f"- Density: {row['density']:.3f} g/cm³.")
    bullets.append(f"- Elements in this compound: {elements_phrase}.")
    return header + "\n" + "\n".join(bullets)


PROMPT_TEMPLATES = {
    "minimal": make_prompt_minimal,
    "rich_v1": make_prompt_rich_v1,
    "rich_v2": make_prompt_rich_v2,
    "rich_v3": make_prompt_rich_v3,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True,
                    help="Stage 3b step= dir (with lora_adapter/ + projector_and_state.pt).")
    ap.add_argument("--atoms_mapper", required=True,
                    help="Path to atoms_mapper.pt produced by the Stage-3 trainer")
    ap.add_argument("--benchmark", required=True, choices=["mp_20", "mpts_52", "perov_5", "carbon_24"],
                    help="CDVAE/CrystaLLM-format benchmark to evaluate against.")
    ap.add_argument("--n", type=int, default=20,
                    help="Number of generations per composition (CrystaLLM reports 1 and 20; "
                         "this CLI accepts any positive int — useful for resample-many test-time "
                         "compute experiments where you crank N up to see if the upper-bound "
                         "match rate climbs).")
    ap.add_argument("--prompt_template", default="rich_v1",
                    choices=list(PROMPT_TEMPLATES.keys()),
                    help="Prompt format. `minimal` matches CrystaLLM's terse 'formula + sg' "
                         "prompt. `rich_v1/v2/v3` mirror GPT-Narratives prose patterns the "
                         "model saw at training time (dft_3d / mp_3d_2020 / oqmd respectively). "
                         "Default `rich_v1` is the closest training-distribution match.")
    ap.add_argument("--mask_elements", action=argparse.BooleanOptionalAction, default=True,
                    help="Hard-mask atomic-number logits at sample time so generations only "
                         "use elements that appear in the target row's `elements` field. "
                         "Default ON. Disable with --no-mask_elements for the unrestricted "
                         "baseline.")
    ap.add_argument("--max_rows", type=int, default=-1,
                    help="Cap test rows for smoke runs; -1 = full set.")
    ap.add_argument("--guidance_factor", type=float, default=1.0,
                    help="CFG guidance scale.")
    ap.add_argument("--diffusion_snr", type=float, default=None,
                    help="Sampling temperature analog (scales Langevin-corrector "
                         "SNR vs defaults pos=0.4 / cell=0.2). <1.0 = warmer, "
                         ">1.0 = cooler, None = default. Same semantics as in "
                         "eval_dng.py.")
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--num_atoms_distribution", default="ALEX_MP_20")
    ap.add_argument("--composition_source", choices=["teacher", "planner"], default="teacher",
                    help="teacher (default) uses ground-truth benchmark composition for the "
                         "JSON prefix / composition cond fields. planner has the LLM emit "
                         "JSON composition first, then conditions on those predicted counts.")
    ap.add_argument("--planner_alm_checkpoint", type=Path, default=None,
                    help="Optional separate clean ALM checkpoint for composition planning. "
                         "Default: use the same ALM loaded for the bridge.")
    ap.add_argument("--planner_prompt_version", default="v5",
                    choices=["v1", "v2", "v3", "v4", "v5"],
                    help="Planner JSON format; v5 asks for per-formula-unit counts and "
                         "the parser scales to the benchmark cell atom count.")
    ap.add_argument("--planner_fallback_teacher", action="store_true",
                    help="If --composition_source planner fails to parse a row, fall back "
                         "to teacher-forced ground-truth counts instead of skipping it. "
                         "Useful for plumbing smokes; leave off for honest planner evals.")
    ap.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    ap.add_argument("--out_root", type=Path, default=None,
                    help="Override $ALM_EVAL_RESULTS_ROOT.")
    ap.add_argument("--run_id", type=str, default=None)
    # Feynman-Kac steering (off by default).
    ap.add_argument("--fk_n_particles", type=int, default=0,
                    help="0 = no FK (default). When > 0, replaces --n: each prompt produces "
                         "fk_n_particles structures via FK steering. target_counts auto-extracted "
                         "from each row's pretty_formula.")
    ap.add_argument("--fk_rewards", type=str,
                    default="stoich_match:1.0;count_l1:1.0;ratio_kl:1.0",
                    help="Reward composition for FK.")
    ap.add_argument("--fk_resample_every", type=int, default=10)
    ap.add_argument("--fk_t_start_frac", type=float, default=0.5)
    ap.add_argument("--fk_lambda", type=float, default=0.5)
    ap.add_argument("--fk_potential", type=str, default="sum",
                    choices=["diff", "sum", "max"],
                    help="Default 'sum' (suited to stationary count rewards).")
    ap.add_argument("--fk_ess_threshold_frac", type=float, default=0.5)
    ap.add_argument("--fk_keep_top_k", type=int, default=-1)
    ap.add_argument("--fk_log_w_clip", type=float, default=10.0)
    ap.add_argument("--fk_constrain_n_atoms_to_target_multiple", action="store_true",
                    help="Restrict per-particle N_p to multiples of sum(target_counts) for "
                         "the row. Recommended on.")
    ap.add_argument("--fk_stratify_resample_by_n_atoms", action="store_true")
    ap.add_argument("--fk_n_atoms_exact_sum_target", action="store_true",
                    help="Force N_p = sum(target_counts) exactly (no multiples). "
                         "For SrTiO3 → only 5 atoms/cell. "
                         "Pair with --fk_enforce_target_counts to guarantee exact stoichiometry.")
    ap.add_argument("--fk_physical_bounds_path", type=Path, default=None,
                    help="Path to JSON of empirical physical-prior bounds. Required if "
                         "--fk_rewards includes 'physical_sanity'.")
    ap.add_argument("--fk_enforce_target_counts", action="store_true",
                    help="Post-hoc Hungarian Z-override at end of denoising — forces exact "
                         "target_counts on every particle by reassigning atom labels via "
                         "Hungarian on the model's final probs. Positions/lattice untouched.")
    # MatterSim relax before matching (geometry-rescue lever).
    ap.add_argument("--relax_before_match", action="store_true",
                    help="Run MatterSim relax on every generation before passing to the "
                         "structure matcher. Reports BOTH unrelaxed (standard convention) "
                         "and relaxed match rates side-by-side. Relaxation can rescue "
                         "right-formula-wrong-geometry rows (unit-cell doublings, rotations, "
                         "near-correct lattices).")
    ap.add_argument("--mattersim_potential_path", type=str, default=None,
                    help="Optional MatterSim checkpoint path; default uses MatterSim's "
                         "bundled mattersim-v1.0.0-1M.")
    # Row sharding for parallel multi-GPU runs.
    ap.add_argument("--shard_idx", type=int, default=0,
                    help="0-indexed shard of rows this process handles. With "
                         "--num_shards N, processes rows where (row_idx %% N) == shard_idx. "
                         "Default 0 = no sharding.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Total shards (1 = no sharding). Set N when launching N parallel "
                         "processes on N GPUs; auto-suffixes --run_id with '_shard{i}of{N}' "
                         "so per-GPU output dirs don't collide.")
    ap.add_argument("--row_start", type=int, default=0,
                    help="Inclusive row index where this process starts. Composes with "
                         "bash ranges for explicit per-GPU partitioning (alternative to "
                         "stride sharding via --shard_idx/--num_shards). Default 0.")
    ap.add_argument("--row_end", type=int, default=-1,
                    help="Exclusive row index where this process stops. Default -1 = end "
                         "of test set. Auto-suffixes --run_id with '_rows{start}-{end}' so "
                         "concurrent ranges write to disjoint dirs.")
    ap.add_argument("--diffusion_seed", type=int, default=1337,
                    help="Seed for diffusion noise. Per-prompt offset added so "
                         "reordering doesn't change individual outputs.")
    # Peaked atomic_numbers init at sampler t=T.
    ap.add_argument("--init_types_at_target", action="store_true",
                    help="Override atomic_numbers immediately after _sample_prior with "
                         "a permuted target multiset (Z values, integer). Lets the "
                         "denoiser see real types from t=T → positions can co-evolve. "
                         "Requires --fk_n_atoms_exact_sum_target. Independent of FK; "
                         "you can run with --fk_n_particles 0 (vanilla diffusion) to "
                         "isolate the init-prior effect, or compose with FK.")
    args = ap.parse_args()
    # argparse required=True allows empty strings (unset shell vars); fast-fail clearly.
    if not str(args.alm_checkpoint).strip():
        ap.error("--alm_checkpoint is empty (likely an unset shell variable like $ALM_CKPT)")
    if not str(args.atoms_mapper).strip():
        ap.error("--atoms_mapper is empty (likely an unset shell variable)")
    ckpt_path = Path(args.alm_checkpoint)
    # Full-FT ckpts store Qwen3 in llm_full_ft/qwen3_state_dict.pt and have no lora_adapter/.
    _is_full_ft = (ckpt_path / "llm_full_ft" / "qwen3_state_dict.pt").is_file()
    if not _is_full_ft and not (ckpt_path / "lora_adapter").is_dir():
        ap.error(f"--alm_checkpoint {ckpt_path} contains neither a lora_adapter/ subdir "
                 "nor llm_full_ft/qwen3_state_dict.pt; check that the path points at a "
                 "Stage 3b step= directory.")
    if not Path(args.atoms_mapper).is_file():
        ap.error(f"--atoms_mapper {args.atoms_mapper} is not a file; expected the "
                 "atoms_mapper.pt under the step= dir.")
    if args.num_shards < 1 or not (0 <= args.shard_idx < args.num_shards):
        ap.error(f"shard_idx ({args.shard_idx}) must be in [0, num_shards={args.num_shards})")
    if args.row_start < 0:
        ap.error(f"--row_start must be >= 0 (got {args.row_start})")
    if args.row_end != -1 and args.row_end <= args.row_start:
        ap.error(f"--row_end ({args.row_end}) must be > --row_start ({args.row_start}) or -1")
    if args.init_types_at_target and not args.fk_n_atoms_exact_sum_target:
        ap.error("--init_types_at_target requires --fk_n_atoms_exact_sum_target so each "
                 "particle has exactly sum(target_counts) atoms to populate from the "
                 "target multiset.")
    if args.init_types_at_target and args.fk_enforce_target_counts:
        print("[init_types] WARNING: --fk_enforce_target_counts is redundant with "
              "--init_types_at_target (Hungarian post-hoc would overwrite an already "
              "count-correct trajectory). Keeping it on is harmless but does extra work.",
              flush=True)
    if args.num_shards > 1:
        suffix = f"_shard{args.shard_idx}of{args.num_shards}"
        args.run_id = (args.run_id + suffix) if args.run_id else f"shard{args.shard_idx}of{args.num_shards}"
    if args.row_start != 0 or args.row_end != -1:
        end_tag = "end" if args.row_end == -1 else str(args.row_end)
        suffix = f"_rows{args.row_start}-{end_tag}"
        args.run_id = (args.run_id + suffix) if args.run_id else f"rows{args.row_start}-{end_tag}"

    # Timestamped progress logging: the "looks hung" stalls are silent ckpt load + first diffusion.
    import time as _time, datetime as _dt
    _t0 = _time.time()
    def _stage(msg):
        elapsed = _time.time() - _t0
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        print(f"[csp][{ts}][+{elapsed:6.1f}s] {msg}", flush=True)

    _stage("args validated; starting eval pipeline")
    _stage(f"  benchmark={args.benchmark}  max_rows={args.max_rows}  n={args.n}  g={args.guidance_factor}")
    _stage(f"  fk_n_particles={args.fk_n_particles}  fk_potential={args.fk_potential}  "
           f"fk_lambda={args.fk_lambda}  fk_resample_every={args.fk_resample_every}")
    _stage(f"  fk_n_atoms_exact_sum_target={args.fk_n_atoms_exact_sum_target}  "
           f"fk_enforce_target_counts={args.fk_enforce_target_counts}  "
           f"relax_before_match={args.relax_before_match}")

    # Lazy import: pulls in MatterGen + ALM (heavy).
    _stage("importing generate_stage3 (loads MatterGen + ALM modules) ...")
    from generate_stage3 import generate_for_prompts, load_alm_and_pl_module  # noqa: E402
    _stage("imports done")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _stage(f"device={device}; loading ALM checkpoint + MatterGen pl_module (this is the slow step, ~30s) ...")
    alm, tokenizer, pl_module, K = load_alm_and_pl_module(
        alm_checkpoint=args.alm_checkpoint,
        atoms_mapper=args.atoms_mapper,
        mattergen_pretrained=args.mattergen_pretrained,
        device=device,
    )
    _stage(f"ALM + pl_module loaded; K={K} output atom tokens")
    planner_alm, planner_tokenizer = alm, tokenizer
    if args.composition_source == "planner" and args.planner_alm_checkpoint is not None:
        from loader import load_alm as _load_alm
        _stage(f"loading separate composition planner from {args.planner_alm_checkpoint} ...")
        planner_alm, planner_tokenizer = _load_alm(
            checkpoint=str(args.planner_alm_checkpoint),
            use_cached_embeddings=True,
            merge_lora=True,
            is_trainable=False,
            device=device,
        )
        planner_alm.eval()
        _stage("separate composition planner loaded")

    # Append prompt template to benchmark name so prompt-format experiments don't collide.
    tmpl_tag = "" if args.prompt_template == "minimal" else f"_{args.prompt_template}"
    benchmark_name = (
        f"stage3b_csp_{args.benchmark}_n{args.n}_g{int(args.guidance_factor*10):02d}"
        f"{tmpl_tag}"
    )
    if args.out_root is not None:
        os.environ["ALM_EVAL_RESULTS_ROOT"] = str(args.out_root)
    rd = run_dir(benchmark_name, args.alm_checkpoint, run_id=args.run_id)
    _stage(f"writing results to {rd}")
    _stage(f"prompt_template={args.prompt_template}")
    _stage(f"composition_source={args.composition_source} "
           f"planner_prompt_version={args.planner_prompt_version} "
           f"planner_fallback_teacher={args.planner_fallback_teacher}")

    _stage(f"reading test CSV (this parses CIFs row-by-row, can take ~30-60s for 1000 rows) ...")
    rows = list(read_test_rows(args.benchmark, max_rows=args.max_rows, bench_root=args.bench_root))
    _stage(f"loaded {len(rows)} test rows from {args.benchmark}")
    # Row range slice [start, end) applied to the post-max_rows subset, before stride sharding.
    if args.row_start != 0 or args.row_end != -1:
        end_idx = len(rows) if args.row_end == -1 else min(args.row_end, len(rows))
        rows = rows[args.row_start:end_idx]
        print(f"[csp] row range: rows[{args.row_start}:{end_idx}] kept {len(rows)} rows", flush=True)
    # Stride sharding (vs contiguous block) is robust to per-row wallclock variance.
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if (i % args.num_shards) == args.shard_idx]
        print(f"[csp] shard {args.shard_idx}/{args.num_shards}: kept {len(rows)} rows "
              f"(stride pattern row_idx % {args.num_shards} == {args.shard_idx})", flush=True)

    if not rows:
        print("[csp] no rows — nothing to do", flush=True)
        return 0

    prompt_fn = PROMPT_TEMPLATES[args.prompt_template]
    prompts = [prompt_fn(r) for r in rows]
    prompt_ids = [r["row_id"] for r in rows]
    if rows:
        print(f"[csp] example prompt for {prompt_ids[0]}:")
        for line in prompts[0].splitlines():
            print(f"      {line}")

    # Resolve the conditioning composition: teacher = GT counts, planner = LLM-emitted counts.
    pre_predictions = []
    resolved_rows = []
    resolved_prompts = []
    resolved_prompt_ids = []
    json_counts_per_prompt = []
    gt_counts_per_prompt = []
    planner_text_per_prompt = []
    planner_correct_per_prompt = []
    n_planner_parse_fail = 0
    n_planner_correct = 0
    if args.composition_source == "planner":
        import eval_planner_csp as epc

    for r, prompt, pid in zip(rows, prompts, prompt_ids):
        gt_counts = _counts_from_structure(r["ref_structure"])
        target_atoms = int(sum(gt_counts.values()))
        plan_text = None
        target_counts = gt_counts
        planner_correct = True
        planner_parse_fail = False

        if args.composition_source == "planner":
            import eval_planner_csp as epc
            plan_prompt = epc.planner_prompt(str(r["formula"]), target_atoms)
            plan_text, parsed = epc.llm_plan(
                plan_prompt, planner_alm, planner_tokenizer,
                prompt_version=args.planner_prompt_version,
            )
            planned_counts, _fu = epc.comp_from_plan(
                parsed, args.planner_prompt_version, target_atoms=target_atoms,
            )
            if planned_counts is None or int(sum(planned_counts.values())) != target_atoms:
                planner_parse_fail = True
                n_planner_parse_fail += 1
                if args.planner_fallback_teacher:
                    target_counts = gt_counts
                    planner_correct = False
                else:
                    pre_predictions.append({
                        "row_id": pid,
                        "formula": r["formula"],
                        "space_group": r["sg_symbol"],
                        "n_gen": 0,
                        "matched_n1": False,
                        "matched_nK": False,
                        "rmse_n1": None,
                        "rmse_nK": None,
                        "validity": {"geom_pct": 0.0, "charge_pct": 0.0},
                        "skipped": True,
                        "planner_parse_fail": True,
                        "planner_text": (plan_text or "")[:300],
                        "gt_counts": gt_counts,
                        "planner_counts": None,
                    })
                    continue
            else:
                target_counts = planned_counts
                planner_correct = (target_counts == gt_counts)
                if planner_correct:
                    n_planner_correct += 1

        resolved_rows.append(r)
        resolved_prompts.append(prompt)
        resolved_prompt_ids.append(pid)
        json_counts_per_prompt.append(target_counts)
        gt_counts_per_prompt.append(gt_counts)
        planner_text_per_prompt.append((plan_text or "")[:300] if plan_text else None)
        planner_correct_per_prompt.append(planner_correct)

    rows, prompts, prompt_ids = resolved_rows, resolved_prompts, resolved_prompt_ids
    if not rows:
        metrics = {
            "benchmark": args.benchmark,
            "n_test": len(pre_predictions),
            "n_scored": 0,
            "n_invalid_geom_skipped": 0,
            "n": args.n,
            "guidance_factor": args.guidance_factor,
            "match_rate_n1": 0.0,
            "match_rate_nK": 0.0,
            "rmse_n1": None,
            "rmse_nK": None,
            "composition_source": args.composition_source,
            "planner_prompt_version": args.planner_prompt_version,
            "planner_parse_fail": n_planner_parse_fail,
            "planner_correct_rate": 0.0,
            "planner_fallback_teacher": bool(args.planner_fallback_teacher),
            "alm_checkpoint": str(args.alm_checkpoint),
            "atoms_mapper": str(args.atoms_mapper),
            "relax_before_match": bool(args.relax_before_match),
        }
        write_run(rd, metrics, pre_predictions)
        _stage("all rows failed planner composition parsing; wrote empty metrics")
        return 0

    if args.composition_source == "planner":
        denom = max(1, len(rows))
        _stage(f"planner composition: parse_fail={n_planner_parse_fail} "
               f"correct={n_planner_correct}/{denom} "
               f"fallback_teacher={args.planner_fallback_teacher}")

    gen_root = rd / "generations"
    gen_root.mkdir(parents=True, exist_ok=True)
    allowed_elements_per_prompt = (
        [list(c.keys()) for c in json_counts_per_prompt] if args.mask_elements else None
    )
    if args.mask_elements:
        print(f"[csp] hard-masking atomic-number logits to resolved composition elements "
              f"(e.g. row 0 → {allowed_elements_per_prompt[0]})", flush=True)

    fk_active = args.fk_n_particles > 0
    composition_count_per_prompt = [
        _composition_count_vec_from_counts(c) for c in json_counts_per_prompt
    ]
    # init_types_at_target needs per-row target_counts even when FK is off.
    need_target_counts = fk_active or args.init_types_at_target
    fk_target_counts_per_prompt = None
    if need_target_counts:
        fk_target_counts_per_prompt = [
            _reduced_z_counts(c) for c in json_counts_per_prompt
        ]
        _stage(f"FK target_counts (reduced) per row — example "
               f"row 0 ({rows[0]['formula']}): {fk_target_counts_per_prompt[0]}")

    physical_bounds = None
    if args.fk_physical_bounds_path is not None:
        import json as _json
        with open(args.fk_physical_bounds_path) as _f:
            _pb = _json.load(_f)
        physical_bounds = _pb.get("bounds", _pb)
        _stage(f"loaded physical-prior bounds from {args.fk_physical_bounds_path}")

    n_rows_to_gen = len(prompts)
    n_per_row = args.fk_n_particles if fk_active else args.n
    _stage(f"starting generation: {n_rows_to_gen} rows × {n_per_row} structures/row "
           f"= {n_rows_to_gen * n_per_row} total. "
           f"At ~30-60s/row this is ~{(n_rows_to_gen * 45)/60:.0f} min total.")
    _stage("generate_for_prompts emits a [gen-batch] line per row → use those for live progress")
    structures_per_prompt = generate_for_prompts(
        prompts=prompts,
        alm=alm, tokenizer=tokenizer, pl_module=pl_module,
        out_root=gen_root,
        batch_size=args.n,
        num_batches=1,
        diffusion_guidance_factor=args.guidance_factor,
        diffusion_snr=args.diffusion_snr,
        num_atoms_distribution=args.num_atoms_distribution,
        prompt_ids=prompt_ids,
        save_meta=False,
        allowed_elements_per_prompt=allowed_elements_per_prompt,
        json_counts_per_prompt=json_counts_per_prompt,
        fk_n_particles=args.fk_n_particles,
        fk_rewards=args.fk_rewards,
        fk_target_counts_per_prompt=fk_target_counts_per_prompt,
        fk_resample_every=args.fk_resample_every,
        fk_t_start_frac=args.fk_t_start_frac,
        fk_lambda=args.fk_lambda,
        fk_potential=args.fk_potential,
        fk_ess_threshold_frac=args.fk_ess_threshold_frac,
        fk_keep_top_k=args.fk_keep_top_k,
        fk_log_w_clip=args.fk_log_w_clip,
        fk_constrain_n_atoms_to_target_multiple=args.fk_constrain_n_atoms_to_target_multiple,
        fk_stratify_resample_by_n_atoms=args.fk_stratify_resample_by_n_atoms,
        fk_n_atoms_exact_sum_target=args.fk_n_atoms_exact_sum_target,
        fk_enforce_target_counts=args.fk_enforce_target_counts,
        fk_physical_bounds=physical_bounds,
        fk_target_sg_per_prompt=[r["sg_symbol"] for r in rows] if fk_active else None,
        diffusion_seed=args.diffusion_seed,
        init_types_at_target=args.init_types_at_target,
        init_types_target_counts_per_prompt=(
            fk_target_counts_per_prompt if args.init_types_at_target else None
        ),
        # Pass the RESOLVED composition (not benchmark truth) to avoid hidden teacher forcing in planner mode.
        chemical_system_per_prompt=[list(c.keys()) for c in json_counts_per_prompt],
        composition_count_per_prompt=composition_count_per_prompt,
        space_group_per_prompt=_resolve_sg_numbers([r["sg_symbol"] for r in rows]),
    )
    _stage(f"generation done; collected structures for {len(structures_per_prompt)} prompts")

    # Optional relax pass: score both raw (standard CSP convention) and relaxed (upper bound).
    relaxed_per_prompt = None
    if args.relax_before_match:
        _stage("starting MatterSim relax pass (flattening structures across prompts) ...")
        from structure_metrics import relax_structures_mattersim
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Flatten, relax one batch, unflatten by per-prompt counts.
        flat = []
        per_prompt_counts = []
        for gens in structures_per_prompt:
            structs_only = []
            for g in gens:
                if isinstance(g, Structure):
                    structs_only.append(g)
                else:
                    try: structs_only.append(AseAtomsAdaptor.get_structure(g))
                    except Exception: pass
            per_prompt_counts.append(len(structs_only))
            flat.extend(structs_only)
        if flat:
            _stage(f"running MatterSim relax on {len(flat)} structures "
                   f"(across {len(per_prompt_counts)} rows; ~30s/struct → "
                   f"~{(len(flat) * 30)/60:.0f} min) ...")
            relaxed_atoms, _ = relax_structures_mattersim(
                flat, device=device,
                potential_path=args.mattersim_potential_path,
            )
            _stage(f"MatterSim relax done; got {len(relaxed_atoms)} relaxed atoms")
            relaxed_structs = [AseAtomsAdaptor.get_structure(a) for a in relaxed_atoms]
            relaxed_per_prompt = []
            cursor = 0
            for k in per_prompt_counts:
                relaxed_per_prompt.append(relaxed_structs[cursor:cursor + k])
                cursor += k
        else:
            _stage("[relax] no structures to relax — skipping")
            relaxed_per_prompt = [[] for _ in per_prompt_counts]

    _stage(f"starting per-row scoring loop (CDVAE matcher: ltol=0.3, stol=0.5, angle_tol=10 deg) ...")
    matcher = cdvae_matcher()
    predictions = list(pre_predictions)
    n_matched_n1 = 0
    n_matched_nK = 0
    rmses_n1 = []
    rmses_nK = []
    # Relaxed counterparts (only populated when --relax_before_match).
    n_matched_n1_r = 0
    n_matched_nK_r = 0
    rmses_n1_r = []
    rmses_nK_r = []
    n_invalid_geom = 0
    _last_heartbeat = _time.time()
    for ridx, (r, gens) in enumerate(zip(rows, structures_per_prompt)):
        rid = r["row_id"]
        formula = r["formula"]
        sg = r["sg_symbol"]
        ref = r["ref_structure"]
        gen_structs = []
        for g in gens:
            if isinstance(g, Structure):
                gen_structs.append(g)
            else:
                try:
                    gen_structs.append(AseAtomsAdaptor.get_structure(g))
                except Exception:
                    n_invalid_geom += 1
        if not gen_structs:
            predictions.append({
                "row_id": rid, "formula": formula, "space_group": sg,
                "n_gen": 0,
                "matched_n1": False, "matched_nK": False,
                "rmse_n1": None, "rmse_nK": None,
                "validity": {"geom_pct": 0.0, "charge_pct": 0.0},
                "gt_counts": gt_counts_per_prompt[ridx],
                "planner_counts": json_counts_per_prompt[ridx],
                "planner_text": planner_text_per_prompt[ridx],
                "planner_correct": bool(planner_correct_per_prompt[ridx]),
                "planner_parse_fail": False,
                "skipped": True,
            })
            continue
        mm = match_many(gen_structs, ref, matcher=matcher)
        v_geom = sum(validity_full(s)["geom"] for s in gen_structs) / len(gen_structs)
        v_charge = sum(validity_full(s)["charge"] for s in gen_structs) / len(gen_structs)
        if mm["matched_n1"]:
            n_matched_n1 += 1
            rmses_n1.append(mm["rmse_n1"])
        if mm["matched_nK"]:
            n_matched_nK += 1
            rmses_nK.append(mm["rmse_nK"])
        mm_r = None
        if relaxed_per_prompt is not None and relaxed_per_prompt[ridx]:
            mm_r = match_many(relaxed_per_prompt[ridx], ref, matcher=matcher)
            if mm_r["matched_n1"]:
                n_matched_n1_r += 1
                rmses_n1_r.append(mm_r["rmse_n1"])
            if mm_r["matched_nK"]:
                n_matched_nK_r += 1
                rmses_nK_r.append(mm_r["rmse_nK"])
        pred = {
            "row_id": rid, "formula": formula, "space_group": sg,
            "n_gen": len(gen_structs),
            "matched_n1": mm["matched_n1"], "rmse_n1": mm["rmse_n1"],
            "matched_nK": mm["matched_nK"], "rmse_nK": mm["rmse_nK"],
            "match_idx": mm["match_idx"],
            "validity": {"geom_pct": v_geom, "charge_pct": v_charge},
            "skipped": False,
            "gt_counts": gt_counts_per_prompt[ridx],
            "planner_counts": json_counts_per_prompt[ridx],
            "planner_text": planner_text_per_prompt[ridx],
            "planner_correct": bool(planner_correct_per_prompt[ridx]),
            "planner_parse_fail": False,
        }
        if mm_r is not None:
            pred["matched_n1_relaxed"] = mm_r["matched_n1"]
            pred["matched_nK_relaxed"] = mm_r["matched_nK"]
            pred["rmse_n1_relaxed"] = mm_r["rmse_n1"]
            pred["rmse_nK_relaxed"] = mm_r["rmse_nK"]
            pred["match_idx_relaxed"] = mm_r["match_idx"]
        predictions.append(pred)
        # Heartbeat every 25 rows or every 60s.
        if (ridx + 1) % 25 == 0 or (_time.time() - _last_heartbeat) > 60:
            n_done = ridx + 1
            cur_match_n1 = sum(1 for p in predictions if not p.get('skipped') and p['matched_n1'])
            cur_match_nK = sum(1 for p in predictions if not p.get('skipped') and p['matched_nK'])
            extra = ""
            if relaxed_per_prompt is not None:
                cur_r1 = sum(1 for p in predictions if p.get('matched_n1_relaxed'))
                cur_rK = sum(1 for p in predictions if p.get('matched_nK_relaxed'))
                extra = f", relaxed: {cur_r1}@1 / {cur_rK}@K"
            _stage(f"scored {n_done}/{len(rows)} rows so far  →  "
                   f"raw: {cur_match_n1}@1 / {cur_match_nK}@K{extra}")
            _last_heartbeat = _time.time()

    n = len(predictions)
    n_scored = sum(1 for p in predictions if not p["skipped"])
    metrics = {
        "benchmark": args.benchmark,
        "n_test": n,
        "n_scored": n_scored,
        "n_invalid_geom_skipped": n_invalid_geom,
        "n": args.n,
        "guidance_factor": args.guidance_factor,
        "match_rate_n1": n_matched_n1 / n_scored if n_scored else 0.0,
        "match_rate_nK": n_matched_nK / n_scored if n_scored else 0.0,
        "rmse_n1": float(sum(rmses_n1) / len(rmses_n1)) if rmses_n1 else None,
        "rmse_nK": float(sum(rmses_nK) / len(rmses_nK)) if rmses_nK else None,
        "rmse_n1_min": float(min(rmses_n1)) if rmses_n1 else None,
        "rmse_nK_min": float(min(rmses_nK)) if rmses_nK else None,
        "alm_checkpoint": str(args.alm_checkpoint),
        "atoms_mapper": str(args.atoms_mapper),
        "relax_before_match": bool(args.relax_before_match),
        "composition_source": args.composition_source,
        "planner_prompt_version": args.planner_prompt_version,
        "planner_parse_fail": n_planner_parse_fail,
        "planner_correct_rate": (
            n_planner_correct / max(1, len(rows))
            if args.composition_source == "planner" else 1.0
        ),
        "planner_fallback_teacher": bool(args.planner_fallback_teacher),
    }
    if args.relax_before_match:
        metrics["match_rate_n1_relaxed"] = n_matched_n1_r / n_scored if n_scored else 0.0
        metrics["match_rate_nK_relaxed"] = n_matched_nK_r / n_scored if n_scored else 0.0
        metrics["rmse_n1_relaxed"] = (
            float(sum(rmses_n1_r) / len(rmses_n1_r)) if rmses_n1_r else None
        )
        metrics["rmse_nK_relaxed"] = (
            float(sum(rmses_nK_r) / len(rmses_nK_r)) if rmses_nK_r else None
        )
    write_run(rd, metrics, predictions)
    _stage(f"wrote {rd}/metrics.json + {rd}/predictions.jsonl")

    _total = _time.time() - _t0
    _stage(f"DONE — total wallclock {_total:.0f}s ({_total/60:.1f} min)")

    print()
    print(f"[csp] {args.benchmark} n={args.n} g={args.guidance_factor}")
    print(f"  match_rate@1  = {metrics['match_rate_n1']:.3f}")
    print(f"  match_rate@K  = {metrics['match_rate_nK']:.3f}")
    print(f"  rmse@1 (mean) = {metrics['rmse_n1']}")
    print(f"  rmse@K (mean) = {metrics['rmse_nK']}")
    if args.relax_before_match:
        print(f"  ── relaxed (MatterSim) ──")
        print(f"  match_rate@1  = {metrics['match_rate_n1_relaxed']:.3f}")
        print(f"  match_rate@K  = {metrics['match_rate_nK_relaxed']:.3f}")
        print(f"  rmse@1 (mean) = {metrics['rmse_n1_relaxed']}")
        print(f"  rmse@K (mean) = {metrics['rmse_nK_relaxed']}")
    print(f"  results in {rd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
