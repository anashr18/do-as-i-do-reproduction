#!/bin/bash
# R1 setup on the 5080: clone do-as-i-do at the commit READING.md was written against,
# snapshot the authors' shipped retargeting outputs (launch.py --force overwrites them),
# and build the single `retargeting` conda env the README specifies.
# Idempotent: re-running skips what exists.
set -eo pipefail
COMMIT=824591b
WORK="$HOME/rig_work/do_as_i_do"
mkdir -p "$WORK" && cd "$WORK"

if [ ! -d do-as-i-do ]; then
  git clone https://github.com/malik-group/do-as-i-do.git
fi
cd do-as-i-do
git checkout -q "$COMMIT"
echo "[setup] repo at $(git rev-parse --short HEAD)"

# The shipped whisking result is the known-good reference. Keep an untouched copy.
if [ ! -d "$WORK/shipped_outputs" ]; then
  cp -a retargeting/outputs "$WORK/shipped_outputs"
  echo "[setup] shipped outputs copied to $WORK/shipped_outputs"
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"
if ! conda env list | grep -q '^retargeting '; then
  conda create -y -n retargeting python=3.12
fi
conda activate retargeting
cd retargeting
pip install -e . 2>&1 | tail -3

python - <<'EOF'
import torch, warp, mujoco, mujoco_warp, coacd, mink
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
print("warp", warp.config.version, "mujoco", mujoco.__version__)
warp.init()
EOF
df -h "$HOME" | tail -1
echo "[setup] done"
