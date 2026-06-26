#!/usr/bin/env bash
# Set up the MatterGen fork for ALM Stage 3 (the LLM→diffusion bridge).
#
# What this does:
#   1. Clones microsoft/mattergen at the pinned commit into external/mattergen
#      (skipped if already present).
#   2. Applies external/mattergen_alm_steering.patch. The patch retargets
#      pyproject.toml to CUDA-12 (H200), bumps pytorch-lightning >=2.4, registers
#      AtomsMapper / AtomsMapperProducerConsumer as the `alm_embedding` conditional
#      embedding module, appends "alm_embedding" to PROPERTY_SOURCE_IDS, adds the
#      from-scratch CSP data-module configs (csp_backbone{,_v2,_stable70,
#      _stable70_cap64}.yaml) + the task_direction embedding config, adds the
#      GemNetTCtrl IP-Adapter / tenc-fuse bridge, and writes install_for_h200.sh.
#   3. Marks install_for_h200.sh executable (git diff doesn't preserve +x).
#
# The ALM bridge modules (alm/atoms_mapper*.py) must be on PYTHONPATH at runtime;
# the patched alm_embedding.yaml references them by name.
#
# Usage (from repo root):
#   bash external/setup_mattergen.sh
#
# Verify the patch matches the checkout:
#   git -C external/mattergen apply --reverse --check external/mattergen_alm_steering.patch

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBMODULE="${REPO_ROOT}/external/mattergen"
PATCH="${REPO_ROOT}/external/mattergen_alm_steering.patch"
MATTERGEN_URL="https://github.com/microsoft/mattergen.git"
# Last upstream microsoft/mattergen commit the ALM patch is built against
# ("save detailed metrics, #239"). The patch carries every ALM edit on top of
# this (the bridge adapter, alm_embedding cond_field, csp_backbone data modules,
# CFG/LMDB/numpy-2 fixes), so a clean clone @ this commit + the patch == the
# full ALM fork. Do NOT pin to a local ALM commit — those SHAs aren't on GitHub.
MATTERGEN_COMMIT="a245cf2b7538eea6d873e6430b0e30c56d26c60e"

[[ -f "$PATCH" ]] || { echo "ERROR: patch not found at $PATCH"; exit 1; }

echo "[1/3] fetch microsoft/mattergen @ ${MATTERGEN_COMMIT:0:10} ..."
if [[ ! -d "$SUBMODULE/.git" ]]; then
  git clone "$MATTERGEN_URL" "$SUBMODULE"
  git -C "$SUBMODULE" checkout -q "$MATTERGEN_COMMIT"
else
  echo "  $SUBMODULE already present — leaving as-is."
fi

echo "[2/3] apply ALM Stage-3 patch ..."
cd "$SUBMODULE"
if git diff --quiet HEAD; then
  if git apply --check "$PATCH" 2>/dev/null; then
    git apply "$PATCH"
    echo "  patch applied."
  else
    echo "  ERROR: patch does not apply cleanly. Ensure the checkout is at"
    echo "  $MATTERGEN_COMMIT (\`git -C $SUBMODULE reset --hard $MATTERGEN_COMMIT\`)."
    exit 1
  fi
else
  echo "  working tree already has edits — skipping (reset --hard to start fresh)."
fi

echo "[3/3] chmod +x install_for_h200.sh ..."
chmod +x "$SUBMODULE/install_for_h200.sh" 2>/dev/null || true

echo
echo "MatterGen fork ready. Install it into the alm env (CUDA-12 / torch 2.9):"
echo "  cd $SUBMODULE && bash install_for_h200.sh && bash build_pyg_for_torch29.sh"
