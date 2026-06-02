#!/usr/bin/env bash
# Install leWorldRobot and its upstream dependencies.
#
# Assumes Python 3.12+ and a virtual environment is already active.
# Run once after cloning:
#
#   python -m venv .venv && source .venv/bin/activate
#   ./install.sh
#
# To upgrade in place: ./install.sh --upgrade
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPGRADE_FLAG="${1:-}"

pip_install() {
    if [[ "$UPGRADE_FLAG" == "--upgrade" ]]; then
        pip install --upgrade "$@"
    else
        pip install "$@"
    fi
}

echo "================================================================"
echo "  leWorldRobot — install"
echo "================================================================"
echo ""

# ── 1. stable-worldmodel ─────────────────────────────────────────────────────
#   If a local clone lives at stable-worldmodel/ use it (editable, so upstream
#   changes take effect immediately). Otherwise fall back to PyPI.
SWM_DIR="$REPO_ROOT/stable-worldmodel"
if [[ -f "$SWM_DIR/pyproject.toml" ]]; then
    echo "▶ Installing stable-worldmodel from local clone (editable)"
    pip_install -e "$SWM_DIR"
else
    echo "▶ Installing stable-worldmodel from PyPI"
    pip_install "stable-worldmodel>=0.1.0"
fi

# ── 2. stable-pretraining ─────────────────────────────────────────────────────
#   Required for the ViT backbone and Lightning training utilities.
echo "▶ Installing stable-pretraining"
pip_install "stable-pretraining"

# ── 3. LeRobot ───────────────────────────────────────────────────────────────
#   Check for a local clone at the sibling path used during development, then
#   fall back to PyPI.
LEROBOT_DIR="$(dirname "$REPO_ROOT")/lerobot"
if [[ -f "$LEROBOT_DIR/pyproject.toml" ]]; then
    echo "▶ Installing LeRobot from local clone at $LEROBOT_DIR (editable)"
    pip_install -e "$LEROBOT_DIR"
else
    echo "▶ Installing LeRobot from PyPI"
    pip_install "lerobot[dataset]>=0.5.1"
fi

# ── 4. This repo (leWorldRobot + lewm_robot package) ─────────────────────────
echo "▶ Installing leWorldRobot (editable)"
pip_install -e "$REPO_ROOT"

echo ""
echo "================================================================"
echo "  Done.  Verify with:"
echo "    python -c \"import lewm_robot; print('lewm_robot OK')\""
echo "    python -c \"from lerobot.policies.factory import get_policy_class; \\"
echo "                print(get_policy_class('jepa'))\""
echo "================================================================"
