"""Generate diverse OOD user-prompt templates via a local vLLM endpoint."""
from __future__ import annotations


import argparse
import asyncio
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

import aiohttp


PLACEHOLDERS_ALLOWED = {
    "formula",
    "sg_symbol",
    "crystal_system",
    "density",
    "n_atoms",
    "n_elements",
    "first_element",
    "elements_csv",
    "volume",
    "n_atoms_int",
}

# Template must contain at least one generation cue.
GENERATION_CUE_PATTERN = re.compile(
    r"\b(structure|crystal|lattice|cell|atomic|"
    r"generate|design|sketch|build|construct|synthesize|"
    r"produce|show|render|visualize|describe|create|"
    r"derive|propose|compose|model|provide)\b",
    re.IGNORECASE,
)

# Reject templates that ask for a property instead of a structure.
PROPERTY_QUESTION_PATTERN = re.compile(
    r"\b(what is the|tell me the|find the|calculate the|compute the|"
    r"predict the|return the|give me the|how (much|many|big|dense))\b.*"
    r"(density|volume|formation energy|hull|band gap|formula|composition)",
    re.IGNORECASE,
)

PERSONAS = [
    "an undergraduate chemistry student",
    "a senior materials science professor",
    "an industrial battery R&D engineer",
    "a science journalist writing for a general audience",
    "a curious hobbyist who reads pop-science articles",
    "a computational chemist familiar with DFT",
    "a software engineer learning materials science",
    "a high school teacher preparing lecture material",
    "a crystallographer reviewing a manuscript",
    "an AI researcher prompting a generative model",
]

STYLES = [
    "imperative commands ('Generate ...', 'Build ...', 'Show me ...')",
    "interrogative questions ('What does X look like?', 'How is X arranged?')",
    "declarative requests ('I need ...', 'I'd like ...')",
    "conditional/hypothetical ('If I had X, what crystal ...', 'Given Y, what structure ...')",
    "tersely-phrased specs (telegraphic, no full sentence; e.g. 'Crystal: X, SG Y, please')",
]

INFO_COMPLETENESS = [
    "full specification (use formula AND space group AND at least one property like density or crystal_system)",
    "formula-only (use only {formula}, do NOT mention space group)",
    "space-group-only (use only {sg_symbol} and {crystal_system}, do NOT mention formula)",
    "property-only (use density, n_atoms, or n_elements but NOT formula or space group)",
    "constraint-set (vague constraints like 'a metallic compound with low density' filled from properties)",
    "minimal hint (extremely sparse — just one piece of info)",
]

LENGTHS = [
    "very short (5-12 words, single line)",
    "short (15-25 words)",
    "medium (30-50 words, may include one supporting clause)",
    "long (one or two full sentences, 50-90 words)",
]

TONES = [
    "formal academic",
    "casual conversational",
    "terse technical (no filler words)",
    "verbose and friendly",
    "professional but approachable",
]


def _meta_prompt(persona, style, info, length, tone, n_templates: int) -> list[dict]:
    placeholders_list = ", ".join("{" + p + "}" for p in sorted(PLACEHOLDERS_ALLOWED))
    system = (
        "You generate diverse user-prompt TEMPLATES that will be answered by producing "
        "a crystal structure. Each template uses {curly_brace} placeholders where it "
        "references metadata about a target material. The TEMPLATES must REQUEST a "
        "crystal structure as the answer — not properties, not narrative, not text "
        "about the material. They must be answerable by outputting an atomic structure.\n\n"
        f"Allowed placeholders ONLY: {placeholders_list}\n\n"
        "Templates must contain at least one placeholder. Use a subset of placeholders "
        "(do not use all of them). Vary phrasing — do NOT repeat the same opening verb "
        "or sentence structure across templates."
    )
    user = (
        f"Generate {n_templates} distinct templates conforming to:\n"
        f"  PERSONA   : prompts as if written by {persona}\n"
        f"  STYLE     : {style}\n"
        f"  INFO LEVEL: {info}\n"
        f"  LENGTH    : {length}\n"
        f"  TONE      : {tone}\n\n"
        "Output exactly one template per line. NO numbering, NO bullets, NO commentary, "
        "NO blank lines, NO Markdown. Just the templates, one per line.\n\n"
        "Reminder: every template must REQUEST a structure (not ask for a property). "
        "Every template must contain at least one allowed placeholder."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _validate_template(line: str) -> str | None:
    line = line.strip().rstrip(".")
    line = re.sub(r"^\s*[-*\d]+[\.)]\s*", "", line)
    line = line.strip("\"'`")
    if not line:
        return None
    if len(line) < 10 or len(line) > 500:
        return None
    found_placeholders = set(re.findall(r"\{([a-z_]+)\}", line))
    if not found_placeholders:
        return None
    if not found_placeholders.issubset(PLACEHOLDERS_ALLOWED):
        return None
    if not GENERATION_CUE_PATTERN.search(line):
        return None
    if PROPERTY_QUESTION_PATTERN.search(line):
        return None
    if line.lower().startswith(("here are", "below are", "as ", "1.", "okay", "sure", "i ")):
        return None
    # Reject placeholders adjacent to each other or to alphanumerics (renders as nonsense).
    if re.search(r"\}\{", line):
        return None
    if re.search(r"\}[A-Za-z0-9]", line) or re.search(r"[A-Za-z0-9]\{", line):
        return None
    return line


_FAILURE_COUNTS: Counter = Counter()
_VERBOSE_FAILURES_REMAINING = 5


async def _one_call(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> list[str]:
    global _VERBOSE_FAILURES_REMAINING
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": max_tokens,
    }
    async with sem:
        try:
            async with session.post(url + "/chat/completions", json=payload, timeout=120) as resp:
                status = resp.status
                if status != 200:
                    body_preview = (await resp.text())[:300]
                    _FAILURE_COUNTS[f"http_{status}"] += 1
                    if _VERBOSE_FAILURES_REMAINING > 0:
                        _VERBOSE_FAILURES_REMAINING -= 1
                        print(f"[gen][FAIL] {url} → HTTP {status}\n         body: {body_preview}",
                              flush=True)
                    return []
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
        except aiohttp.ClientConnectorError as exc:
            _FAILURE_COUNTS["connect_refused"] += 1
            if _VERBOSE_FAILURES_REMAINING > 0:
                _VERBOSE_FAILURES_REMAINING -= 1
                print(f"[gen][FAIL] {url} → connect refused: {exc}", flush=True)
            return []
        except asyncio.TimeoutError:
            _FAILURE_COUNTS["timeout"] += 1
            if _VERBOSE_FAILURES_REMAINING > 0:
                _VERBOSE_FAILURES_REMAINING -= 1
                print(f"[gen][FAIL] {url} → timeout", flush=True)
            return []
        except Exception as exc:
            _FAILURE_COUNTS[f"other:{type(exc).__name__}"] += 1
            if _VERBOSE_FAILURES_REMAINING > 0:
                _VERBOSE_FAILURES_REMAINING -= 1
                print(f"[gen][FAIL] {url} → {type(exc).__name__}: {exc}", flush=True)
            return []
    lines = []
    n_rejected = 0
    for line in content.split("\n"):
        v = _validate_template(line)
        if v:
            lines.append(v)
        elif line.strip():
            n_rejected += 1
    if not lines and n_rejected > 0:
        _FAILURE_COUNTS["all_rejected_by_validator"] += 1
        if _VERBOSE_FAILURES_REMAINING > 0:
            _VERBOSE_FAILURES_REMAINING -= 1
            print(f"[gen][FAIL-VALIDATE] {url}: all {n_rejected} non-empty output lines "
                  f"rejected by validator. First raw line:\n   "
                  f"{repr(content.split(chr(10))[0][:200])}", flush=True)
    return lines


async def _gather_calls(args, calls: list[dict]) -> list[str]:
    """Round-robin calls across comma-separated vLLM endpoints under one shared concurrency semaphore."""
    urls = [u.strip().rstrip("/") for u in args.vllm_url.split(",") if u.strip()]
    if not urls:
        raise ValueError("No URLs parsed from --vllm_url")
    print(f"[gen] {len(urls)} endpoint(s): {urls}")

    sem = asyncio.Semaphore(args.concurrency)
    results: list[str] = []
    counter = Counter()
    async with aiohttp.ClientSession() as session:
        tasks = [
            _one_call(
                session, urls[i % len(urls)], args.model, c["messages"],
                c["temperature"], args.max_tokens, sem,
            )
            for i, c in enumerate(calls)
        ]
        for i, fut in enumerate(asyncio.as_completed(tasks)):
            templates = await fut
            results.extend(templates)
            for t in templates:
                counter[t] += 1
            if (i + 1) % 50 == 0:
                print(f"[gen] {i+1}/{len(tasks)} calls done; "
                      f"raw templates so far: {len(results)}, unique: {len(counter)}",
                      flush=True)
    return results


def _build_call_specs(args) -> list[dict]:
    rng = random.Random(args.seed)
    specs = []
    axes = list(itertools.product(PERSONAS, STYLES, INFO_COMPLETENESS, LENGTHS, TONES))
    for _ in range(args.n_calls):
        persona, style, info, length, tone = rng.choice(axes)
        temperature = rng.uniform(0.8, 1.2)
        specs.append({
            "messages": _meta_prompt(persona, style, info, length, tone, args.templates_per_call),
            "temperature": temperature,
        })
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm_url", default="http://localhost:8000/v1",
                    help="vLLM OpenAI-compatible base URL. May be comma-separated to "
                         "round-robin across multiple servers (one per L40S, etc.).")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="Model name as registered in vLLM.")
    ap.add_argument("--n_calls", type=int, default=2000,
                    help="Total LLM calls. ~100K raw templates at 50 per call.")
    ap.add_argument("--templates_per_call", type=int, default=50)
    ap.add_argument("--max_tokens", type=int, default=1500,
                    help="Output token budget per call. Must satisfy "
                         "prompt_tokens + max_tokens <= context_length (4096 for Qwen3-4B). "
                         "1500 is plenty for 50 templates × ~25 tokens each.")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="Max in-flight requests; vLLM batches across these.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_path", type=Path, required=True)
    args = ap.parse_args()

    print(f"[gen] generating templates: {args.n_calls} calls × {args.templates_per_call} each")
    print(f"[gen] target endpoint: {args.vllm_url} model={args.model}")
    print(f"[gen] concurrency={args.concurrency}")

    specs = _build_call_specs(args)
    raw = asyncio.run(_gather_calls(args, specs))
    print(f"[gen] raw templates collected: {len(raw)}")
    if _FAILURE_COUNTS:
        print(f"[gen] FAILURE BREAKDOWN: {dict(_FAILURE_COUNTS)}")

    seen = set()
    unique = []
    for t in raw:
        key = re.sub(r"\s+", " ", t.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    print(f"[gen] unique templates after dedup: {len(unique)}")

    breakdown = Counter()
    for t in unique:
        ph = set(re.findall(r"\{([a-z_]+)\}", t))
        has_formula = "formula" in ph
        has_sg = "sg_symbol" in ph
        breakdown[(len(ph), has_formula, has_sg)] += 1
    print(f"[gen] breakdown by (n_placeholders, has_formula, has_sg): "
          f"{dict(breakdown.most_common(10))}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        for t in unique:
            f.write(json.dumps({"template": t}) + "\n")
    print(f"[gen] wrote {len(unique)} templates to {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
