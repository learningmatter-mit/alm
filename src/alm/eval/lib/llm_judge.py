"""Shared OpenAI LLM-judge utilities for app + atom+text generation evals."""
from __future__ import annotations


import asyncio
import json
import os
import re
from collections import Counter

import aiohttp


OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"


_FAILURE_COUNTS: Counter = Counter()
_VERBOSE_REMAINING = 5


def reset_failure_counts() -> None:
    global _VERBOSE_REMAINING
    _FAILURE_COUNTS.clear()
    _VERBOSE_REMAINING = 5


def get_failure_counts() -> dict:
    return dict(_FAILURE_COUNTS)


def _check_api_key() -> str:
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if not k:
        raise RuntimeError(
            "OPENAI_API_KEY is not set in environment. "
            "Export it before running the eval: `export OPENAI_API_KEY=sk-...`"
        )
    return k


async def _one_judge_call(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    messages: list[dict],
    sem: asyncio.Semaphore,
    max_tokens: int = 400,
    temperature: float = 0.0,
    max_retries: int = 6,
) -> dict | None:
    """One chat completion with exponential-backoff retry on 429/5xx; forces JSON output."""
    import random
    global _VERBOSE_REMAINING
    payload = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    # Reasoning models require max_completion_tokens (with headroom for reasoning tokens); o-series reject temperature != 1.
    if model.startswith(("gpt-5", "o1", "o3", "o4")):
        payload["max_completion_tokens"] = max(max_tokens, 1500)
        if model.startswith("gpt-5"):
            payload["temperature"] = temperature
    else:
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_status: int | None = None
    for attempt in range(max_retries):
        async with sem:
            try:
                async with session.post(
                    OPENAI_BASE_URL + "/chat/completions",
                    json=payload, headers=headers, timeout=120,
                ) as resp:
                    status = resp.status
                    last_status = status
                    if status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        try:
                            return json.loads(content)
                        except (json.JSONDecodeError, TypeError):
                            _FAILURE_COUNTS["json_parse"] += 1
                            if _VERBOSE_REMAINING > 0:
                                _VERBOSE_REMAINING -= 1
                                print(f"[llm_judge][FAIL json] content: {content[:300]}", flush=True)
                            return None
                    if status == 429 or 500 <= status < 600:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after is not None:
                            try:
                                wait_s = float(retry_after)
                            except ValueError:
                                wait_s = 2 ** attempt + random.random()
                        else:
                            wait_s = min(60.0, 2 ** attempt + random.random())
                        if attempt == 0 and _VERBOSE_REMAINING > 0:
                            _VERBOSE_REMAINING -= 1
                            print(f"[llm_judge][retry http {status}] sleeping {wait_s:.1f}s "
                                  f"(attempt {attempt+1}/{max_retries})", flush=True)
                        pass
                    else:
                        body = (await resp.text())[:500]
                        _FAILURE_COUNTS[f"http_{status}"] += 1
                        if _VERBOSE_REMAINING > 0:
                            _VERBOSE_REMAINING -= 1
                            print(f"[llm_judge][FAIL http {status}] body: {body}", flush=True)
                        return None
            except aiohttp.ClientConnectorError as exc:
                _FAILURE_COUNTS["connect_refused"] += 1
                if attempt == max_retries - 1:
                    return None
                wait_s = min(60.0, 2 ** attempt + random.random())
            except asyncio.TimeoutError:
                _FAILURE_COUNTS["timeout"] += 1
                if attempt == max_retries - 1:
                    return None
                wait_s = min(60.0, 2 ** attempt + random.random())
            except Exception as exc:
                _FAILURE_COUNTS[f"other:{type(exc).__name__}"] += 1
                return None
        # Sleep outside the semaphore so other workers can proceed.
        if last_status == 429 or (last_status is not None and 500 <= last_status < 600) or last_status is None:
            await asyncio.sleep(wait_s)
    _FAILURE_COUNTS[f"retry_exhausted_http_{last_status}"] += 1
    if _VERBOSE_REMAINING > 0:
        _VERBOSE_REMAINING -= 1
        print(f"[llm_judge][FAIL retries-exhausted last_status={last_status}]", flush=True)
    return None


async def batch_judge(
    items: list[dict],
    build_messages_fn,           # (item: dict) -> list[{"role", "content"}]
    model: str = DEFAULT_MODEL,
    concurrency: int = 32,
    max_tokens: int = 400,
    temperature: float = 0.0,
) -> list[dict | None]:
    """Parallel judge calls; returns parsed dicts aligned 1:1 with `items` (None on fail)."""
    api_key = _check_api_key()
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        async def task(i: int, item: dict):
            msgs = build_messages_fn(item)
            r = await _one_judge_call(
                session, api_key, model, msgs, sem,
                max_tokens=max_tokens, temperature=temperature,
            )
            return i, r
        coros = [task(i, item) for i, item in enumerate(items)]
        out: list[dict | None] = [None] * len(items)
        n_done = 0
        for fut in asyncio.as_completed(coros):
            i, r = await fut
            out[i] = r
            n_done += 1
            if n_done % 100 == 0:
                fc = dict(_FAILURE_COUNTS.most_common(3))
                print(f"[llm_judge] {n_done}/{len(items)} done; failures: {fc}", flush=True)
    return out


# Prompt templates

APP_CONSISTENCY_SYSTEM = """\
You are a senior materials scientist. Given (1) a user's request for a material \
with specific applications and (2) a generated crystal structure, judge whether \
the structure's properties make it suitable for the requested application.

Use chemistry knowledge to assess. The structure has been DFT-relaxed; assume \
the relaxed numbers are representative of the actual material.

Output STRICT JSON with these keys:
- "verdict": "consistent" | "inconsistent" | "ambiguous"
- "score": 2 (clearly consistent) | 1 (ambiguous, plausible but not strong) | 0 (inconsistent / wrong domain)
- "reason": one sentence explanation, ≤30 words
- "extracted_application": short canonical name of the application class (e.g. "wide_bandgap_semiconductor", "battery_cathode")"""


def build_app_consistency_messages(item: dict) -> list[dict]:
    """item keys: prompt, formula, density, volume_per_atom, formation_energy_per_atom, elements, space_group, n_atoms."""
    user = (
        f"USER PROMPT: {item['prompt']}\n\n"
        f"GENERATED STRUCTURE (after MatterSim relaxation):\n"
        f"  formula: {item.get('formula')}\n"
        f"  space_group: {item.get('space_group')}\n"
        f"  n_atoms in cell: {item.get('n_atoms')}\n"
        f"  elements: {item.get('elements')}\n"
        f"  density: {item.get('density'):.3f} g/cm^3\n"
        f"  volume_per_atom: {item.get('volume_per_atom'):.3f} A^3\n"
        f"  formation_energy_per_atom: {item.get('formation_energy_per_atom'):.3f} eV/atom\n"
        f"\nIs this structure consistent with the application requested in the prompt? "
        f"Reply with strict JSON only."
    )
    return [
        {"role": "system", "content": APP_CONSISTENCY_SYSTEM},
        {"role": "user", "content": user},
    ]


ATOMTXT_DIRECTION_SYSTEM = """\
You are a senior materials scientist. The user asked for a structural/property \
modification of a known material; compare the input and output structures to \
judge whether the modification went in the requested direction at the chemistry \
level (above and beyond raw property numbers).

Output STRICT JSON with these keys:
- "verdict": "correct_direction" | "wrong_direction" | "no_change" | "wrong_chemistry"
- "score": 2 (clearly moved in right direction with sensible chemistry) | 1 (ambiguous / partial) | 0 (wrong)
- "reason": one sentence explanation, ≤30 words"""


def build_atomtxt_direction_messages(item: dict) -> list[dict]:
    """item keys: prompt, prop_target, direction_target, input_{formula,density,vpa,fe}, output_{formula,density,vpa,fe}."""
    user = (
        f"USER PROMPT: {item['prompt']}\n"
        f"REQUESTED MODIFICATION: {item['prop_target']} should be {item['direction_target']}\n\n"
        f"INPUT STRUCTURE (the user's starting material, MatterSim-relaxed):\n"
        f"  formula: {item['input_formula']}\n"
        f"  density: {item['input_density']:.3f} g/cm^3\n"
        f"  volume_per_atom: {item['input_vpa']:.3f} A^3\n"
        f"  formation_energy_per_atom: {item['input_fe']:.3f} eV/atom\n\n"
        f"OUTPUT STRUCTURE (model's response, MatterSim-relaxed):\n"
        f"  formula: {item['output_formula']}\n"
        f"  density: {item['output_density']:.3f} g/cm^3\n"
        f"  volume_per_atom: {item['output_vpa']:.3f} A^3\n"
        f"  formation_energy_per_atom: {item['output_fe']:.3f} eV/atom\n\n"
        f"Did the output move in the requested direction? Reply with strict JSON only."
    )
    return [
        {"role": "system", "content": ATOMTXT_DIRECTION_SYSTEM},
        {"role": "user", "content": user},
    ]


def parse_score(judge_response: dict | None, default: int = 0) -> int:
    if not judge_response:
        return default
    s = judge_response.get("score")
    if isinstance(s, int) and 0 <= s <= 2:
        return s
    if isinstance(s, str):
        m = re.match(r"\s*(\d+)\s*$", s)
        if m:
            v = int(m.group(1))
            if 0 <= v <= 2:
                return v
    return default
