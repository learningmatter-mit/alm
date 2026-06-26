"""Shared element/composition helpers: symbol->Z map and composition vectors."""
from __future__ import annotations


import numpy as np

# Z range modeled, matches CompositionHead's 100-d output.
N_ELEMENTS = 100

# Per-element count cap; pairs.parquet is <=20 atoms/cell so 20 bounds any element.
MAX_COUNT = 20


def symbol_to_z() -> dict[str, int]:
    """ASE element symbol -> atomic number, restricted to Z in [1, N_ELEMENTS]."""
    from ase.data import atomic_numbers
    return {s: z for s, z in atomic_numbers.items() if 1 <= z <= N_ELEMENTS}


def composition_multihot(elements: list[str], sym2z: dict | None = None) -> np.ndarray:
    """Presence multi-hot over Z=1..N_ELEMENTS at slot Z-1; shape (N_ELEMENTS,) float32."""
    if sym2z is None:
        sym2z = symbol_to_z()
    v = np.zeros(N_ELEMENTS, dtype=np.float32)
    for s in elements:
        z = sym2z.get(s.strip())
        if z is not None and 1 <= z <= N_ELEMENTS:
            v[z - 1] = 1.0
    return v


def composition_count_vec(elements: list[str], sym2z: dict | None = None,
                          dtype: np.dtype = np.int64) -> np.ndarray:
    """Per-Z count clamped to [0, MAX_COUNT], shape (N_ELEMENTS,); use float32 dtype to torch.stack with other float aux targets."""
    if sym2z is None:
        sym2z = symbol_to_z()
    v = np.zeros(N_ELEMENTS, dtype=np.int64)
    for s in elements:
        z = sym2z.get(s.strip())
        if z is not None and 1 <= z <= N_ELEMENTS:
            v[z - 1] += 1
    np.clip(v, 0, MAX_COUNT, out=v)
    return v.astype(dtype, copy=False)
