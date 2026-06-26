"""Build application-style structure-generation prompts from GPT-Narratives via a local vLLM endpoint."""
from __future__ import annotations


import argparse
import asyncio
import os
import re
from collections import Counter
from pathlib import Path

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from paths import DATA_ROOT


GENERATION_CUE_PATTERN = re.compile(
    r"\b(generate|design|build|construct|synthesize|produce|"
    r"create|propose|sketch|render|make)\b",
    re.IGNORECASE,
)

# Signals the LLM produced an off-task property question instead of a request.
PROPERTY_ASK_PATTERN = re.compile(
    r"\b(what is the|tell me the|how (much|big|dense)|find the|calculate the)\b",
    re.IGNORECASE,
)

# Skip generic/refusal-style explanations that yield no signal.
JUNK_EXPL_PATTERNS = [
    re.compile(r"^as the material .* is not (a )?stable compound", re.IGNORECASE),
    re.compile(r"cannot be used in any (industrial|commercial)", re.IGNORECASE),
    re.compile(r"its (crystal system|space group|density).*are not relevant", re.IGNORECASE),
    re.compile(r"^based on the (limited|provided) (information|properties),? (very )?little"),
    re.compile(r"^i (don'?t|do not) have (sufficient|enough) information", re.IGNORECASE),
]


def _is_junk(explanation: str) -> bool:
    if not explanation:
        return True
    s = explanation.strip()
    if len(s) < 80:
        return True
    for pat in JUNK_EXPL_PATTERNS:
        if pat.search(s):
            return True
    return False


# Per-parent source-column -> canonical-property-name mapping.
APP_PROPS_PER_PARENT = {
    "dft_3d": {
        "formation_energy_eV_per_atom": "formation energy per atom (eV/atom)",
        "band_gap_eV": "band gap (eV)",
        "density_g_per_cm3": "density (g/cm³)",
        "total_magnetization_uB_per_fu": "total magnetization (μB/f.u.)",
        "energy_above_hull_eV_per_atom": "energy above hull (eV/atom)",
        "crystal_system": "crystal system",
    },
    "mp_3d_2020": {
        "formation_energy_eV_per_atom": "formation energy per atom (eV/atom)",
        "band_gap_eV": "band gap (eV)",
        "density_g_per_cm3": "density (g/cm³)",
        "total_magnetization_uB_per_fu": "total magnetization (μB/f.u.)",
        "energy_above_hull_eV_per_atom": "energy above hull (eV/atom)",
        "crystal_system": "crystal system",
    },
    "aflow2": {
        "formation_energy_eV_per_atom": "formation energy per atom (eV/atom)",
        "band_gap_eV": "band gap (eV)",
        "density_g_per_cm3": "density (g/cm³)",
        "energy_above_hull_eV_per_atom": "energy above hull (eV/atom)",
        "crystal_system": "crystal system",
    },
    "oqmd": {
        "formation_energy_eV_per_atom": "_oqmd_delta_e",
        "band_gap_eV": "_oqmd_band_gap",
        "density_g_per_cm3": "density (g/cm3)",
        "crystal_system": "crystal system",
    },
}


def _format_prop_value(k: str, v):
    if v is None:
        return None
    try:
        if isinstance(v, str):
            return v
        f = float(v)
        if not (f == f):  # NaN
            return None
        if 'eV' in k or 'gap' in k or 'energy' in k or 'magnetiz' in k:
            return f"{f:.3f}"
        if 'density' in k:
            return f"{f:.2f} g/cm³"
        return f"{f:.3f}"
    except (TypeError, ValueError):
        return None


DOMAINS = [
    "Li-ion battery cathode or anode",
    "solid-state battery electrolyte",
    "photovoltaic absorber for solar cells",
    "wide-bandgap semiconductor for power electronics",
    "narrow-bandgap semiconductor for IR detection",
    "transparent conducting oxide for displays",
    "thermoelectric for waste-heat recovery",
    "ferroelectric or piezoelectric for sensors/actuators",
    "magnetic refrigeration material",
    "permanent magnet for motors",
    "spintronic / magnetoresistive material",
    "photocatalyst for water splitting or CO2 reduction",
    "heterogeneous catalyst for hydrocarbon conversion",
    "structural ceramic for high-temperature applications",
    "refractory for furnace lining",
    "superconductor for magnets or transmission",
    "optical phosphor for LED lighting",
    "scintillator for radiation detection",
    "electrochromic material for smart windows",
    "thermal-barrier coating for jet engines",
    "hydrogen storage material",
    "ionic conductor for fuel cells",
    "2D material for nanoelectronics",
    "topological insulator",
    "metal-organic framework for gas separation",
    "shape-memory alloy",
    "biocompatible material for implants",
    "anti-corrosion coating",
    "neutron absorber",
    "high-entropy alloy for structural applications",
]


def _build_prompt(formula: str, properties: dict,
                   target_domain: str) -> list[dict]:
    # Pass structured `properties` (not the explanation text) to avoid mode collapse onto templated prose.
    prop_lines = []
    for k, v in properties.items():
        if v is None:
            continue
        prop_lines.append(f"  {k}: {v}")
    if not prop_lines:
        prop_lines.append("  (no properties — generate based on TARGET DOMAIN only)")
    prop_block = "\n".join(prop_lines)

    system = (
        "You write ONE-LINE user requests asking for a crystal structure with "
        "specific properties or for a specific application domain. The user wants "
        "to GENERATE a new material; do not return descriptions of existing materials.\n\n"
        "Rules:\n"
        "- Output exactly ONE LINE, 6-40 words, no preamble, no commentary.\n"
        "- Begin with a generation verb (Generate, Design, Build, Make, Create, Synthesize, Propose).\n"
        "- Use the TARGET DOMAIN as the framing for the request whenever the "
        "  PROPERTIES are at least loosely consistent with it (e.g. low-bandgap "
        "  metals can be magnetic-storage, structural alloys, or thermoelectric; "
        "  wide-bandgap insulators can be dielectrics or photocatalysts; etc.).\n"
        "- If the properties are flatly incompatible with the target domain "
        "  (e.g. target='superconductor' but properties show a wide-bandgap "
        "  insulator), pick a DIFFERENT specific domain consistent with the "
        "  properties. Be specific.\n"
        "- DO NOT include any chemical formula or element symbols in your output.\n"
        "- DO NOT mention 'scintillator', 'scintillation', or 'radiation detection' — "
        "  these are over-represented in our training data; suppress them. (Even if "
        "  the property values would suggest it, choose another framing.)\n"
        "- Reference at most one property numerically in your output (e.g. "
        "  '~1.5 eV bandgap'); be qualitative ('wide-bandgap', 'low-density', "
        "  'highly magnetic') rather than quoting the raw number for the rest.\n\n"
        "Examples (each from different domains):\n"
        "- 'Generate a perovskite absorber with ~1.4 eV band gap for tandem solar cells.'\n"
        "- 'Design a wide-bandgap nitride semiconductor for high-voltage MOSFET applications.'\n"
        "- 'Make a stable Li-ion conductor with low formation energy for solid-state batteries.'\n"
        "- 'Create a 2D transition metal dichalcogenide for room-temperature spintronic logic.'\n"
        "- 'Build a high-permittivity dielectric oxide suitable for gate capacitor stacks.'\n"
        "- 'Synthesize a low-thermal-conductivity narrow-gap thermoelectric for waste-heat recovery.'\n"
        "- 'Propose a hexagonal close-packed structural alloy for jet-engine turbine blades.'\n"
        "- 'Generate a heavy-fermion compound with strong f-electron correlations for magnetic refrigeration.'\n"
        "- 'Design a topological semimetal candidate with linearly-crossing bands near the Fermi level.'\n"
        "- 'Make a porous framework material with internal cavities for hydrogen storage.'"
    )
    user = (
        f"TARGET DOMAIN: {target_domain}\n\n"
        f"PROPERTIES OF THE MATERIAL (for grounding only — do NOT echo specific numbers "
        f"unless qualitatively meaningful):\n{prop_block}\n\n"
        f"ONE-LINE USER REQUEST (single line, no preamble, generation verb to start):"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _validate_prompt(text: str, formula: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    # Strip stray "Output:"/"Prompt:" prefixes the model sometimes adds.
    s = re.sub(r"^(prompt|output|user request|response)\s*:\s*", "", s, flags=re.IGNORECASE)
    s = s.strip().strip("\"'`")
    if "\n" in s:
        s = s.split("\n", 1)[0].strip()
    n_words = len(s.split())
    if n_words < 6 or n_words > 40:
        return None
    if not GENERATION_CUE_PATTERN.search(s):
        return None
    if PROPERTY_ASK_PATTERN.search(s):
        return None
    # Reject only a standalone-token formula leak, not a substring of another word.
    if formula and re.search(r"\b" + re.escape(formula) + r"\b", s):
        return None
    if s.lower().startswith(("here is", "here are", "below is", "as ", "okay", "sure", "i ")):
        return None
    return s


async def _one_call(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    formula: str,
    properties: dict,
    target_domain: str,
    max_tokens: int,
    sem: asyncio.Semaphore,
    failure_counts: Counter,
) -> str | None:
    payload = {
        "model": model,
        "messages": _build_prompt(formula, properties, target_domain),
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    async with sem:
        try:
            async with session.post(url + "/chat/completions", json=payload, timeout=120) as resp:
                if resp.status != 200:
                    failure_counts[f"http_{resp.status}"] += 1
                    return None
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
        except aiohttp.ClientConnectorError:
            failure_counts["connect_refused"] += 1
            return None
        except asyncio.TimeoutError:
            failure_counts["timeout"] += 1
            return None
        except Exception as exc:
            failure_counts[f"other:{type(exc).__name__}"] += 1
            return None
    v = _validate_prompt(content, formula)
    if v is None:
        failure_counts["validator_rejected"] += 1
    return v


def _formula_from_atoms_struct(atoms_struct: dict) -> str:
    elements = [str(e).strip() for e in atoms_struct.get("elements", [])]
    if not elements:
        return ""
    counts = Counter(elements)
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))


def _load_properties(narratives_root: Path) -> dict:
    """Build a {(parent, source_idx): {canonical_prop: pretty_value}} lookup."""
    out: dict = {}
    for parent, prop_map in APP_PROPS_PER_PARENT.items():
        p = narratives_root / f"{parent}_gpt_narratives.parquet"
        if not p.exists():
            print(f"[load] skip {parent} (not found at {p})")
            continue
        pf = pq.ParquetFile(p)
        cols = [c for c in prop_map.values() if c in pf.schema_arrow.names]
        print(f"[load] {parent}: {pf.metadata.num_rows:,} rows, "
              f"{len(cols)}/{len(prop_map)} property columns present")
        idx = 0
        for batch in pf.iter_batches(batch_size=10000, columns=cols):
            for r in batch.to_pylist():
                row_props = {}
                for canonical, source_col in prop_map.items():
                    if source_col not in cols:
                        continue
                    v = r.get(source_col)
                    pretty = _format_prop_value(canonical, v)
                    if pretty is not None:
                        row_props[canonical] = pretty
                out[(parent, idx)] = row_props
                idx += 1
    print(f"[load] total property rows cached: {len(out):,}")
    return out


async def _process_rows(args, rows: list[dict]) -> list[dict]:
    """Send `rows` to vLLM round-robin and return those with `app_prompt` filled."""
    urls = [u.strip().rstrip("/") for u in args.vllm_url.split(",") if u.strip()]
    if not urls:
        raise ValueError("No URLs parsed from --vllm_url")
    print(f"[gen] {len(urls)} endpoint(s)")

    sem = asyncio.Semaphore(args.concurrency)
    failure_counts: Counter = Counter()
    out_rows: list[dict] = []

    # Pre-sample a target domain per row; seeded for reproducibility.
    import random as _rnd
    domain_rng = _rnd.Random(args.seed)
    row_domains = [domain_rng.choice(DOMAINS) for _ in rows]

    async with aiohttp.ClientSession() as session:
        async def task(i, r):
            url = urls[i % len(urls)]
            return r, await _one_call(
                session, url, args.model, r["formula"], r["properties"],
                row_domains[i],
                args.max_tokens, sem, failure_counts,
            )
        coros = [task(i, r) for i, r in enumerate(rows)]
        n = 0
        n_ok = 0
        for fut in asyncio.as_completed(coros):
            r, prompt = await fut
            n += 1
            if prompt:
                r["app_prompt"] = prompt
                out_rows.append(r)
                n_ok += 1
            if n % 1000 == 0:
                print(f"[gen] {n}/{len(rows)} done; kept {n_ok} so far; "
                      f"failures: {dict(failure_counts.most_common(5))}", flush=True)
    print(f"[gen] FINAL — {n} processed, {n_ok} kept; "
          f"failures: {dict(failure_counts)}", flush=True)
    return out_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm_url", default="http://localhost:8000/v1",
                    help="Comma-separated vLLM endpoints for round-robin.")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pairs_parquet", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs.parquet")))
    ap.add_argument("--narratives_root", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "GPT-Narratives-for-Materials")))
    ap.add_argument("--out_path", type=Path,
                    default=Path(os.path.join(DATA_ROOT, "stage3_outputs/stage3a/pairs_app.parquet")))
    ap.add_argument("--max_rows", type=int, default=200000,
                    help="Cap on rows to process. Default 200K — enough for a "
                         "meaningful bucket without spending excessive vLLM time.")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[main] loading structured properties from {args.narratives_root} ...")
    prop_lookup = _load_properties(args.narratives_root)

    print(f"[main] loading pairs from {args.pairs_parquet} ...")
    pf = pq.ParquetFile(args.pairs_parquet)
    schema = pf.schema_arrow

    import random as _rnd
    rng = _rnd.Random(args.seed)
    candidates: list[dict] = []
    n_seen = 0
    n_skipped = 0
    for batch in tqdm(pf.iter_batches(batch_size=10000),
                       total=(pf.metadata.num_rows + 9999) // 10000,
                       desc="scan-pairs"):
        for r in batch.to_pylist():
            n_seen += 1
            parent = r["parent"]
            source_idx = r["source_idx"]
            props = prop_lookup.get((parent, source_idx), {})
            if not props:
                n_skipped += 1
                continue
            formula = _formula_from_atoms_struct(r["atoms_struct"])
            if not formula:
                continue
            candidates.append({
                **r,
                "formula": formula,
                "properties": props,
            })
    print(f"[main] candidates: {len(candidates):,} of {n_seen:,} seen "
          f"({n_skipped:,} skipped — no properties)")

    if args.max_rows > 0 and len(candidates) > args.max_rows:
        rng.shuffle(candidates)
        candidates = candidates[:args.max_rows]
        print(f"[main] sampled down to {len(candidates):,} (capped at --max_rows)")

    print(f"[main] sending {len(candidates):,} rows to vLLM ...")
    completed = asyncio.run(_process_rows(args, candidates))

    print(f"[main] writing {len(completed):,} rows → {args.out_path}")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out_path, schema, compression="snappy")
    BATCH = 10000
    for i in range(0, len(completed), BATCH):
        chunk = completed[i:i+BATCH]
        out_dict = {col: [] for col in schema.names}
        for r in chunk:
            for col in schema.names:
                if col == "user_prompt":
                    out_dict[col].append(r["app_prompt"])
                else:
                    out_dict[col].append(r[col])
        writer.write_table(pa.Table.from_pydict(out_dict, schema=schema))
    writer.close()
    print(f"[main] done. Sample 5 prompts:")
    for r in completed[:5]:
        print(f"  {r['app_prompt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
