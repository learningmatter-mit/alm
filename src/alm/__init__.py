"""Atomistic Language Model (ALM): a Qwen3-8B backbone bridged to a diffusion crystal decoder."""
from . import _alm_bootstrap  # noqa: F401  side effect: extends sys.path

__all__ = ["AtomisticLanguageModel"]


def __getattr__(name):  # lazy re-export so package import doesn't pull in torch
    if name == "AtomisticLanguageModel":
        from model import AtomisticLanguageModel
        return AtomisticLanguageModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
