#!/usr/bin/env bash
# Set up the MatterGen fork used by ALM Stage 3 (the LLM to diffusion bridge).
#
# What this does:
#   1. Clones microsoft/mattergen at the pinned commit into external/mattergen
#      (skipped if already present).
#   2. Applies external/mattergen_alm_steering.patch. The patch retargets
#      pyproject.toml to CUDA 12, bumps pytorch-lightning to >=2.4, adds the
#      `alm_embedding` cond_field (alm_embedding.yaml, backed by bridge.AtomsMapper)
#      and appends it to PROPERTY_SOURCE_IDS, adds the from-scratch CSP data-module
#      config (csp_backbone.yaml) and the task_direction embedding config, adds the
#      GemNetTCtrl IP-Adapter / tenc-fuse bridge, and writes install_cu128.sh.
#   3. Marks install_cu128.sh executable (git diff doesn't preserve +x).
#
# The patched alm_embedding.yaml references the bridge modules (src/alm/bridge.py)
# by bare module name. `import alm` puts them on sys.path; MatterGen's own CLIs
# (mattergen-finetune, mattergen-generate) need PYTHONPATH=<repo>/src/alm.
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
# Upstream microsoft/mattergen commit the patch is built against
# ("save detailed metrics, #239"). The patch carries every ALM edit on top of it
# (the bridge adapter, alm_embedding cond_field, csp_backbone data module, and the
# CFG/LMDB/numpy-2 fixes), so a clean clone at this commit plus the patch is the
# full fork. Keep this an upstream commit so the clone can fetch it from GitHub.
MATTERGEN_COMMIT="a245cf2b7538eea6d873e6430b0e30c56d26c60e"

[[ -f "$PATCH" ]] || { echo "ERROR: patch not found at $PATCH"; exit 1; }

echo "[1/3] fetch microsoft/mattergen @ ${MATTERGEN_COMMIT:0:10} ..."
if [[ ! -d "$SUBMODULE/.git" ]]; then
  git clone "$MATTERGEN_URL" "$SUBMODULE"
  git -C "$SUBMODULE" checkout -q "$MATTERGEN_COMMIT"
else
  echo "  $SUBMODULE exists; skipping clone."
fi

echo "[2/3] apply ALM patch ..."
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
  echo "  working tree already has edits; skipping (reset --hard to start fresh)."
fi

echo "[3/3] chmod +x install_cu128.sh ..."
chmod +x "$SUBMODULE/install_cu128.sh" 2>/dev/null || true

echo
echo "MatterGen fork ready. Install it into the alm env (CUDA 12, torch 2.9):"
echo "  cd $SUBMODULE && bash install_cu128.sh && bash build_pyg_for_torch29.sh"
