# source adamu_nut_yam/cotrain/env.sh : the main venv (setup/setup.sh) + EGL rendering. If torch was installed from
# CUDA 13 wheels, the bundled cuDNN is put on the loader path (needed for bf16 convolutions).
_R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$_R/.venv/bin/activate"
_N="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia"
[ -d "$_N/cudnn/lib" ] && export LD_LIBRARY_PATH="$_N/cudnn/lib:$_N/cu13/lib:${LD_LIBRARY_PATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl} OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
