#!/bin/bash
# One-time setup (Linux, NVIDIA GPU, Python 3.11, uv):
#   .venv      Diffusion Policy, ACT, DSRL           (torch 2.15 / CUDA 13.2 wheels)
#   .venv-tw   Touch in the Wild and DexScrew        (torch 2.4.1 / CUDA 12.1, the versions their code expects)
#   ext/       the two upstream repositories, at the commits this code was developed against (unmodified)
# Usage: setup/setup.sh            (VENV_ONLY=main|tw to build one environment)
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd); cd $REPO
command -v uv >/dev/null || { echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

clone() {  # url dir commit
  [ -d ext/$2/.git ] || git clone -q $1 ext/$2
  git -C ext/$2 checkout -q $3
}
mkdir -p ext
clone https://github.com/XinyueZhuXY/touch_in_the_wild touch_in_the_wild 2a946261d34bee5ac363c5dc696810273a0b20d8
clone https://github.com/x-robotics-lab/dexscrew dexscrew 3bde4e3a4d973743921c75719ca88167de144e83

if [ "${VENV_ONLY:-main}" = main ] || [ -z "$VENV_ONLY" ]; then
  [ -d .venv ] || uv venv -q --python 3.11 .venv
  uv pip install -q --python .venv/bin/python -r setup/requirements-main.txt \
    --extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match
fi
if [ "${VENV_ONLY:-tw}" = tw ] || [ -z "$VENV_ONLY" ]; then
  [ -d .venv-tw ] || uv venv -q --python 3.11 .venv-tw
  uv pip install -q --python .venv-tw/bin/python -r setup/requirements-tw.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match
fi
echo "setup done: source .venv/bin/activate (DP / ACT / DSRL) or .venv-tw/bin/activate (Touch in the Wild / DexScrew)"
