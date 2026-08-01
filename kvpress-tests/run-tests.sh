#!/usr/bin/env bash
set -euo pipefail

FORK_URL="${FORK_URL:-https://github.com/fax4ever/kvpress.git}"
FORK_BRANCH="${FORK_BRANCH:-main}"

echo "=== Cloning kvpress fork ==="
echo "  URL:    $FORK_URL"
echo "  Branch: $FORK_BRANCH"
git clone --depth 1 --branch "$FORK_BRANCH" "$FORK_URL" /tmp/kvpress-src

echo ""
echo "=== Installing kvpress from fork ==="
pip install --no-deps /tmp/kvpress-src

echo ""
echo "=== Environment ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "from importlib.metadata import version; print(f'kvpress {version(\"kvpress\")}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Running tests ==="
cd /tmp/kvpress-src
pytest "$@"
