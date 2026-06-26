"""Put every ALM source dir plus the MatterGen fork on sys.path once, so the flat namespace's bare imports resolve from any entry point."""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

_DIRS = [
    _HERE,
    os.path.join(_HERE, "utils"),
    os.path.join(_HERE, "eval", "lib"),
    os.path.join(_HERE, "eval", "understanding"),
    os.path.join(_HERE, "eval", "generation"),
    os.path.join(_HERE, "train"),
    os.path.join(_HERE, "inference"),
    os.path.join(_REPO, "scripts"),
    os.path.join(_REPO, "external", "mattergen"),           # patched MatterGen fork
]

for _d in _DIRS:
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
