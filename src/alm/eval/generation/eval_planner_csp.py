"""CSP eval with an instruction-tuned LLM planner as front-end to a from-scratch MG-CSP decoder."""
from __future__ import annotations


import argparse
import csv
import json
import re
import sys
import os
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# alm/ is not a package; loader.py lives under alm/eval/.
_ALM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ALM_ROOT)

from loader import load_alm  # noqa: E402
from paths import DATA_ROOT, RUNS  # noqa: E402

BENCHMARK_CSV = {
    "mp_20":   Path(os.path.join(DATA_ROOT, "eval_data/csp/mp_20/test.csv")),
    "mpts_52": Path(os.path.join(DATA_ROOT, "eval_data/csp/mpts_52/test.csv")),
}
TOL = dict(ltol=0.3, stol=0.5, angle_tol=10.0)

PLANNER_SYSTEMS = {
    "v1": (
        "You are a planner for a crystal-structure generator. Given a query, "
        "respond with ONLY a JSON object (no prose, no markdown fences). "
        "For explicit formulas, output {\"composition\": {element: count_per_formula_unit}, "
        "\"formula_units\": int, \"target_num_atoms\": int}. The composition must "
        "sum × formula_units to target_num_atoms exactly."
    ),
    # v2: per-cell counts direct; drops the v1 formula_units indirection that tripped binaries.
    "v2": (
        "You are a planner for a crystal-structure generator. Given a query, "
        "respond with ONLY a JSON object (no prose, no markdown fences). "
        "For explicit formulas, output {\"counts\": {element: TOTAL_atoms_in_cell}}. "
        "The values are the total atom counts in the cell (not per formula unit). "
        "Examples: SrTiO3 with 5 atoms per cell → {\"counts\": {\"Sr\": 1, \"Ti\": 1, \"O\": 3}}. "
        "SrTiO3 with 10 atoms per cell → {\"counts\": {\"Sr\": 2, \"Ti\": 2, \"O\": 6}}. "
        "GaTe with 8 atoms per cell → {\"counts\": {\"Ga\": 4, \"Te\": 4}}."
    ),
    # v3: v2 + step-by-step + nested-parens + element guard.
    "v3": (
        "You are a planner for a crystal-structure generator. Given a query, "
        "respond with ONLY a JSON object (no prose, no markdown fences).\n"
        "\n"
        "Output format: {\"counts\": {ELEMENT_SYMBOL: TOTAL_atoms_in_cell, ...}}.\n"
        "  - ELEMENT_SYMBOL must be a standard IUPAC symbol (1 or 2 letters). "
        "Never combine two symbols into one key.\n"
        "  - Values are TOTAL atoms in the cell (NOT per formula unit). "
        "Sum of all values MUST equal the target atom count.\n"
        "\n"
        "Step-by-step procedure:\n"
        "  1. Parse the chemical formula and expand any parenthesised groups: "
        "M(AB)2 = M + 2A + 2B, M(ABn)k = M + kA + nk B. "
        "Example: Mg(CoGe)6 expands to 1 Mg + 6 Co + 6 Ge = 13 atoms per formula unit.\n"
        "  2. Compute the per-formula-unit atom count.\n"
        "  3. Determine the number of formula units Z so that Z × (per-formula-unit atoms) = target atom count.\n"
        "  4. Multiply each element by Z to get total counts.\n"
        "\n"
        "Examples:\n"
        "  SrTiO3 with 5 atoms  → Z=1 → {\"counts\": {\"Sr\": 1, \"Ti\": 1, \"O\": 3}}\n"
        "  SrTiO3 with 10 atoms → Z=2 → {\"counts\": {\"Sr\": 2, \"Ti\": 2, \"O\": 6}}\n"
        "  GaTe with 8 atoms    → Z=4 → {\"counts\": {\"Ga\": 4, \"Te\": 4}}\n"
        "  Y3Lu with 8 atoms    → 4 atoms per fu, Z=2 → {\"counts\": {\"Y\": 6, \"Lu\": 2}}\n"
        "  Mg(CoGe)6 with 13 atoms → 13 atoms per fu, Z=1 → {\"counts\": {\"Mg\": 1, \"Co\": 6, \"Ge\": 6}}\n"
        "  La(SiPt)2 with 10 atoms → 5 atoms per fu, Z=2 → {\"counts\": {\"La\": 2, \"Si\": 4, \"Pt\": 4}}"
    ),
    # v4: LLM emits per-formula-unit only; parser computes Z = target / sum(per_fu).
    "v4": (
        "You are a planner for a crystal-structure generator. Given a formula, "
        "respond with ONLY a JSON object (no prose, no markdown fences).\n"
        "\n"
        "Output format: {\"per_formula_unit\": {ELEMENT: count_in_one_formula_unit, ...}}.\n"
        "  - ELEMENT must be a standard IUPAC symbol (1 or 2 letters). "
        "Never combine two symbols into one key like \"ClO\".\n"
        "  - Values are atoms per ONE formula unit. Do NOT multiply by Z, do NOT scale to the target.\n"
        "\n"
        "Parsing rule for parenthesised groups:\n"
        "  M(AB)2  = 1 M + 2 A + 2 B\n"
        "  M(ABn)k = 1 M + k A + (n·k) B\n"
        "  Example: Mg(CoGe)6 → 1 Mg + 6 Co + 6 Ge.\n"
        "\n"
        "Examples (formula → per-formula-unit JSON):\n"
        "  SrTiO3    → {\"per_formula_unit\": {\"Sr\": 1, \"Ti\": 1, \"O\": 3}}\n"
        "  GaTe      → {\"per_formula_unit\": {\"Ga\": 1, \"Te\": 1}}\n"
        "  Y3Lu      → {\"per_formula_unit\": {\"Y\": 3, \"Lu\": 1}}\n"
        "  Ho3TmMn8  → {\"per_formula_unit\": {\"Ho\": 3, \"Tm\": 1, \"Mn\": 8}}\n"
        "  Mg(CoGe)6 → {\"per_formula_unit\": {\"Mg\": 1, \"Co\": 6, \"Ge\": 6}}\n"
        "  La(SiPt)2 → {\"per_formula_unit\": {\"La\": 1, \"Si\": 2, \"Pt\": 2}}\n"
        "  Fe(HO)2   → {\"per_formula_unit\": {\"Fe\": 1, \"H\": 2, \"O\": 2}}\n"
        "  V3(O2F)2  → {\"per_formula_unit\": {\"V\": 3, \"O\": 4, \"F\": 2}}"
    ),
    # v5: v4 + nested-group emphasis + anti-hallucination + ClO-mode prevention.
    "v5": (
        "You are a planner for a crystal-structure generator. Given a formula, "
        "respond with ONLY a JSON object (no prose, no markdown fences).\n"
        "\n"
        "Output format: {\"per_formula_unit\": {ELEMENT: count_in_one_formula_unit, ...}}.\n"
        "  - ELEMENT must be a standard IUPAC symbol (1 or 2 letters). "
        "Never combine two symbols into one key (e.g. NEVER write \"ClO\" as one key; "
        "split into \"Cl\" and \"O\" separately).\n"
        "  - Values are atoms per ONE formula unit. Do NOT multiply by Z, do NOT scale to the target.\n"
        "  - Only emit keys for elements that LITERALLY APPEAR in the formula. "
        "Do not invent extra elements.\n"
        "\n"
        "Parenthesised-group rule: the outer subscript multiplies EVERY element inside.\n"
        "  M(AB)n      = M + n·A + n·B            (each of A and B gets ×n)\n"
        "  M(A_k B)n   = M + (n·k)·A + n·B        (inner ×k stacks with outer ×n on A only)\n"
        "  M(A B_k)n   = M + n·A + (n·k)·B        (inner ×k stacks with outer ×n on B only)\n"
        "  M(A_j B_k)n = M + (n·j)·A + (n·k)·B\n"
        "  CRITICAL: if a group like (InSe2)2 has 2 elements In and Se2 with outer ×2:\n"
        "    In gets ×2 (outer multiplier only — In had no inner subscript)\n"
        "    Se gets 2·2 = 4 (inner ×2 stacked with outer ×2)\n"
        "\n"
        "Examples (formula → per-formula-unit JSON):\n"
        "  SrTiO3       → {\"per_formula_unit\": {\"Sr\": 1, \"Ti\": 1, \"O\": 3}}\n"
        "  GaTe         → {\"per_formula_unit\": {\"Ga\": 1, \"Te\": 1}}\n"
        "  Y3Lu         → {\"per_formula_unit\": {\"Y\": 3, \"Lu\": 1}}\n"
        "  Mg(CoGe)6    → {\"per_formula_unit\": {\"Mg\": 1, \"Co\": 6, \"Ge\": 6}}\n"
        "  Mn(InSe2)2   → {\"per_formula_unit\": {\"Mn\": 1, \"In\": 2, \"Se\": 4}}\n"
        "  KAl(SO4)2    → {\"per_formula_unit\": {\"K\": 1, \"Al\": 1, \"S\": 2, \"O\": 8}}\n"
        "  Ca2Cu(ClO)2  → {\"per_formula_unit\": {\"Ca\": 2, \"Cu\": 1, \"Cl\": 2, \"O\": 2}}\n"
        "  LaTm(Ge2Ir)2 → {\"per_formula_unit\": {\"La\": 1, \"Tm\": 1, \"Ge\": 4, \"Ir\": 2}}\n"
        "  V3(O2F)2     → {\"per_formula_unit\": {\"V\": 3, \"O\": 4, \"F\": 2}}\n"
        "  Fe(HO)2      → {\"per_formula_unit\": {\"Fe\": 1, \"H\": 2, \"O\": 2}}"
    ),
}


def planner_prompt(formula: str, n_atoms: int) -> str:
    return (
        f"Predict the crystal structure for {formula}. The target cell contains "
        f"{n_atoms} atoms total."
    )


def extract_json(text: str):
    """First balanced JSON object/list in `text` (whichever of '{'/'[' appears first), or None."""
    s = re.sub(r"^```(?:json)?\s*", "", text.strip())
    s = re.sub(r"\s*```\s*$", "", s)
    i_brace, i_bracket = s.find("{"), s.find("[")
    if i_bracket >= 0 and (i_brace < 0 or i_bracket < i_brace):
        opener, closer, i = "[", "]", i_bracket
    elif i_brace >= 0:
        opener, closer, i = "{", "}", i_brace
    else:
        return None
    depth = 0
    for j in range(i, len(s)):
        if s[j] == opener:
            depth += 1
        elif s[j] == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[i : j + 1])
                except json.JSONDecodeError:
                    return None
    return None


def llm_plan(prompt: str, alm, tok, *, prompt_version: str = "v1",
             max_new_tokens: int = 256):
    msgs = [
        {"role": "system", "content": PLANNER_SYSTEMS[prompt_version]},
        {"role": "user", "content": prompt},
    ]
    device = next(alm.llm.parameters()).device
    ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", enable_thinking=False,
    ).to(device)
    with torch.no_grad():
        out = alm.llm.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    return text, extract_json(text)


def comp_from_plan(parsed, prompt_version: str = "v1", target_atoms: int | None = None):
    """Return (per_cell_counts, formula_units|None) or (None, None) on parse/validation failure."""
    if parsed is None or isinstance(parsed, list):
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        else:
            return None, None
    if not isinstance(parsed, dict):
        return None, None
    if prompt_version in ("v4", "v5"):
        pfu = parsed.get("per_formula_unit") or parsed.get("composition")
        if not isinstance(pfu, dict) or not pfu:
            return None, None
        try:
            pfu_int = {str(el): int(n) for el, n in pfu.items()}
        except (TypeError, ValueError):
            return None, None
        if any(n <= 0 for n in pfu_int.values()) or not _valid_elements(pfu_int):
            return None, None
        s = sum(pfu_int.values())
        if s <= 0 or target_atoms is None:
            return pfu_int, 1
        if target_atoms % s != 0:
            # non-integer Z = planner mis-parsed the formula
            return None, None
        Z = target_atoms // s
        return {el: n * Z for el, n in pfu_int.items()}, Z
    if prompt_version in ("v2", "v3"):
        comp = parsed.get("counts") or parsed.get("composition")  # tolerate v1 key
        if not isinstance(comp, dict) or not comp:
            return None, None
        try:
            per_cell = {str(el): int(n) for el, n in comp.items()}
        except (TypeError, ValueError):
            return None, None
        if any(n <= 0 for n in per_cell.values()):
            return None, None
        if not _valid_elements(per_cell):
            return None, None
        return per_cell, None
    comp = parsed.get("composition")
    if not isinstance(comp, dict) or not comp:
        return None, None
    try:
        fu = int(parsed.get("formula_units", 1) or 1)
        per_cell = {str(el): int(n) * fu for el, n in comp.items()}
    except (TypeError, ValueError):
        return None, None
    if any(n <= 0 for n in per_cell.values()):
        return None, None
    if not _valid_elements(per_cell):
        return None, None
    return per_cell, fu


def _valid_elements(per_cell: dict) -> bool:
    """False if any key is not a valid element symbol (e.g. 'ClO')."""
    from pymatgen.core.periodic_table import Element
    for el in per_cell.keys():
        try:
            Element(el)
        except (ValueError, KeyError):
            return False
    return True


def load_targets(benchmark: str = "mp_20"):
    csv_path = BENCHMARK_CSV[benchmark]
    targets = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            mp_id, cif = row.get("material_id"), row.get("cif")
            if mp_id and cif:
                try:
                    targets.append((mp_id, Structure.from_str(cif, fmt="cif")))
                except Exception:
                    pass
    return targets


def load_targets_parquet(parquet_path: Path, max_rows: int = 1000,
                         max_n_atoms: int = 30, seed: int = 0):
    """Load (row_id, Structure) from a stage3a-style parquet's `atoms_struct` column: last `max_rows` rows with n_atoms <= max_n_atoms (held-out)."""
    import pyarrow.parquet as pq
    import numpy as np
    table = pq.read_table(parquet_path)
    n_total = table.num_rows
    # stream chunks from the end to avoid loading the full 1.35M-row parquet into memory
    targets = []
    chunk_size = 50_000
    pos = n_total
    while pos > 0 and len(targets) < max_rows:
        start = max(0, pos - chunk_size)
        sub = table.slice(start, pos - start).to_pylist()
        for row in reversed(sub):
            a = row.get("atoms_struct") or {}
            elems = a.get("elements")
            lattice = a.get("lattice_mat")
            coords = a.get("coords")
            cartesian = a.get("cartesian", False)
            if not (elems and lattice and coords):
                continue
            n_atoms = len(elems)
            if n_atoms > max_n_atoms:
                continue
            try:
                lattice = np.asarray(lattice, dtype=float)
                coords = np.asarray(coords, dtype=float)
                s = Structure(
                    lattice=lattice,
                    species=list(elems),
                    coords=coords,
                    coords_are_cartesian=bool(cartesian),
                )
                rid = row.get("row_id") or row.get("source_idx") or f"row_{n_total - 1 - len(targets)}"
                targets.append((rid, s))
                if len(targets) >= max_rows:
                    break
            except Exception:
                pass
        pos = start
    return list(reversed(targets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alm_checkpoint", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "alm_checkpoints/stage2_r128_arxivIT/step=12000")))
    ap.add_argument("--mg_ckpt_dir", type=Path,
                    default=Path(os.path.join(RUNS, "mg_csp_train")),
                    help="Local MG-CSP training dir (contains checkpoints/ + config.yaml).")
    ap.add_argument("--max_rows", type=int, default=1000)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Stride sharding factor for parallel GPU launches.")
    ap.add_argument("--shard_idx", type=int, default=0,
                    help="This worker's shard idx in [0, num_shards). Keeps rows "
                         "where row_idx %% num_shards == shard_idx.")
    ap.add_argument("--use_oracle_comp", action="store_true",
                    help="Bypass planner; use ground-truth composition. Useful as the "
                         "no-LLM control inside the same eval harness.")
    ap.add_argument("--benchmark", default="mp_20",
                    choices=list(BENCHMARK_CSV.keys()),
                    help="MP-20 or MPTS-52 test set (CSV-based).")
    ap.add_argument("--test_parquet", type=Path, default=None,
                    help="Alternative to --benchmark: a parquet path (stage3a-style "
                         "with atoms_struct column). When set, loads N=1000 held-out "
                         "rows from the END of the parquet.")
    ap.add_argument("--prompt_version", default="v1",
                    choices=["v1", "v2", "v3", "v4", "v5"],
                    help="v1: composition + formula_units indirection. "
                         "v2: per-cell counts direct. "
                         "v3: per-cell counts + step-by-step. "
                         "v4: per-formula-unit only, parser computes Z. "
                         "v5: v4 + explicit nested-group expansion rule + anti-hallucination "
                         "+ ClO-mode prevention.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[planner-csp] writing → {args.out_dir}", flush=True)
    t0 = time.time()

    if args.test_parquet is not None:
        all_targets = load_targets_parquet(args.test_parquet, max_rows=args.max_rows)
        src = f"parquet {args.test_parquet.name}"
    else:
        all_targets = load_targets(args.benchmark)[: args.max_rows]
        src = f"{args.benchmark} CSV"
    if args.num_shards > 1:
        targets = [t for i, t in enumerate(all_targets)
                   if (i % args.num_shards) == args.shard_idx]
        print(f"  shard {args.shard_idx}/{args.num_shards}: {len(targets)}/{len(all_targets)} rows from {src}",
              flush=True)
    else:
        targets = all_targets
        print(f"  {len(targets)} rows from {src}", flush=True)

    # Lazy-import MatterGen (heavy).
    from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
    from mattergen.generator import CrystalGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the LLM planner (skip if using oracle comp, saves ~30 GB GPU RAM).
    alm = tok = None
    if not args.use_oracle_comp:
        print(f"  loading planner from {args.alm_checkpoint} ...", flush=True)
        alm, tok = load_alm(
            checkpoint=str(args.alm_checkpoint),
            use_cached_embeddings=True,   # text-only, skip OrbV3
            merge_lora=True,
            is_trainable=False,
        )
        alm.eval()
        print(f"  planner loaded (t={time.time()-t0:.0f}s)", flush=True)

    print(f"  loading MG-CSP ckpt from {args.mg_ckpt_dir} ...", flush=True)
    ckpt_info = MatterGenCheckpointInfo(model_path=str(args.mg_ckpt_dir), load_epoch="last")

    matcher = StructureMatcher(**TOL)
    results = []
    n_match_n1 = 0
    n_match_nK = 0
    n_planner_correct = 0
    n_planner_parse_fail = 0

    for i, (mp_id, target) in enumerate(targets):
        formula = str(target.composition.reduced_formula)
        gt_counts = dict(Counter([str(s.specie.symbol) for s in target]))
        n_atoms_target = sum(gt_counts.values())

        plan_text = None
        if args.use_oracle_comp:
            target_comp = gt_counts
            planner_correct = True
            planner_fu = None
        else:
            plan_text, parsed = llm_plan(
                planner_prompt(formula, n_atoms_target), alm, tok,
                prompt_version=args.prompt_version,
            )
            target_comp, planner_fu = comp_from_plan(
                parsed, args.prompt_version, target_atoms=n_atoms_target,
            )
            if target_comp is None:
                n_planner_parse_fail += 1
                results.append({
                    "row_id": mp_id, "formula": formula,
                    "matched_n1": False, "matched_nK": False, "first_match_idx": -1,
                    "n_gen": 0, "planner_correct": False, "planner_parse_fail": True,
                    "planner_text": (plan_text or "")[:300],
                })
                continue
            planner_correct = (target_comp == gt_counts)
            if planner_correct:
                n_planner_correct += 1

        if (i % 5) == 0:
            extra = "" if args.use_oracle_comp else f"  planner→{target_comp}{'✓' if planner_correct else '✗'}"
            print(f"  [{i:3d}/{len(targets)}] {mp_id} {formula} (gt={gt_counts}){extra} "
                  f"t={time.time()-t0:.0f}s", flush=True)

        prompt_gens_dir = args.out_dir / "gens" / mp_id
        prompt_gens_dir.mkdir(parents=True, exist_ok=True)
        try:
            gen = CrystalGenerator(
                checkpoint_info=ckpt_info,
                batch_size=args.K, num_batches=1,
                target_compositions_dict=[target_comp],
                diffusion_guidance_factor=args.guidance_factor,
                sampling_config_name="csp",
            )
            samples = gen.generate(output_dir=str(prompt_gens_dir))
        except Exception as e:
            print(f"    [{mp_id}] gen failed: {e}", flush=True)
            results.append({
                "row_id": mp_id, "formula": formula,
                "matched_n1": False, "matched_nK": False, "first_match_idx": -1,
                "n_gen": 0, "planner_correct": planner_correct,
                "planner_parse_fail": False, "gen_error": str(e)[:200],
                "planner_text": (plan_text or "")[:300],
            })
            continue

        matched_n1 = False
        matched_nK = False
        first_match_idx = -1
        for j, s in enumerate(samples):
            if not isinstance(s, Structure):
                try:
                    s = AseAtomsAdaptor.get_structure(s)
                except Exception:
                    continue
            try:
                if matcher.fit(target, s):
                    matched_nK = True
                    if j == 0:
                        matched_n1 = True
                    if first_match_idx < 0:
                        first_match_idx = j
            except Exception:
                pass
        if matched_n1:
            n_match_n1 += 1
        if matched_nK:
            n_match_nK += 1
        results.append({
            "row_id": mp_id, "formula": formula,
            "matched_n1": matched_n1, "matched_nK": matched_nK,
            "first_match_idx": first_match_idx, "n_gen": len(samples),
            "planner_correct": planner_correct,
            "planner_parse_fail": False,
            "planner_text": (plan_text or "")[:300],
            "gt_counts": gt_counts,
            "planner_counts": target_comp,
        })

    n = len(results)
    headline = {
        "n_rows": n, "K": args.K, "guidance_factor": args.guidance_factor,
        "match_rate_n1": n_match_n1 / max(1, n),
        "match_rate_nK": n_match_nK / max(1, n),
        "planner_correct_rate": n_planner_correct / max(1, n - n_planner_parse_fail) if not args.use_oracle_comp else 1.0,
        "planner_parse_fail": n_planner_parse_fail,
        "mg_ckpt_dir": str(args.mg_ckpt_dir),
        "alm_checkpoint": str(args.alm_checkpoint),
        "use_oracle_comp": args.use_oracle_comp,
        "note": ("LLM planner → MG-CSP CSP-mode" if not args.use_oracle_comp
                 else "oracle composition → MG-CSP (no-LLM control)"),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(headline, indent=2))
    with (args.out_dir / "predictions.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n[planner-csp] HEADLINE on MP-20 ({n} rows, K={args.K}):")
    print(f"  M@1               = {headline['match_rate_n1']:.4f}  ({n_match_n1}/{n})")
    print(f"  M@K               = {headline['match_rate_nK']:.4f}  ({n_match_nK}/{n})")
    if not args.use_oracle_comp:
        print(f"  planner_correct   = {headline['planner_correct_rate']:.4f}  "
              f"({n_planner_correct}/{n - n_planner_parse_fail})")
        print(f"  planner_parse_fail= {n_planner_parse_fail}")
    print(f"  total time        = {time.time()-t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
