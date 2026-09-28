#!/bin/bash
# R2 pod setup (Blackwell, CUDA 12.8): repo at 824591b + the three reconstruction envs we need.
# HaWoR is skipped (no MANO licence yet); R2 uses the authors' shipped HaWoR output instead.
#
#   sam3   : setup/01_create_envs.sh sam3    (the repo's cu128 recipe, Python 3.12, torch 2.8)
#   tapnet : setup/01_create_envs.sh tapnet  (the repo's cu128 recipe, Python 3.10, torch 2.7)
#   sam3d  : built here to match env/sam3d.yml (the authors' known-good Blackwell env):
#            Python 3.11, torch 2.8.0+cu128, the export's pinned pip set, and the CUDA extensions
#            from the fork's own git pins (pytorch3d, gsplat, MoGe) or the export's versions
#            (kaolin 0.18.0, nvdiffrast, spconv-cu120 2.3.6). The export has no flash-attn or
#            xformers, so neither is installed.
# Each extension step records ok/FAIL in $STATUS and the script carries on, so one pass
# shows every failure.
set -o pipefail
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 MAX_JOBS=64
W=~/rig_work; mkdir -p $W; STATUS=$W/r2_setup_status.txt; : > $STATUS
step() { local name=$1; shift; local t0=$(date +%s); if "$@" > $W/setup_$name.log 2>&1; then r=ok; else r=FAIL; fi;
         echo "$name $r $(( $(date +%s) - t0 ))s" | tee -a $STATUS; }

# --- conda
if [ ! -d ~/miniforge3 ]; then
  curl -sL -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/mf.sh -b -p ~/miniforge3 > $W/setup_miniforge.log 2>&1
fi
source ~/miniforge3/etc/profile.d/conda.sh

# --- repo + the four submodules we use (HaWoR not initialised)
cd $W
if [ ! -d do-as-i-do ]; then GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/malik-group/do-as-i-do.git; fi
cd do-as-i-do && git checkout -q 824591b
step submodules env GIT_LFS_SKIP_SMUDGE=1 git submodule update --init \
  reconstruction/modules/sam3 reconstruction/modules/sam-3d-objects \
  reconstruction/modules/Fast-SAM3D reconstruction/modules/tapnet
git submodule status | tee -a $STATUS
R=$W/do-as-i-do/reconstruction

# --- sam3 and tapnet: the repo's own cu128 recipes
step env_sam3 bash $R/setup/01_create_envs.sh sam3
step env_tapnet bash $R/setup/01_create_envs.sh tapnet
# 01_create_envs.sh installs an unpinned ffmpeg, which resolved to 9.0.2 on 2026-09-26. ffmpeg 9
# dropped -vsync, which run_pipeline.sh passes and which mediapy's video reader trips on
# ("Unable to parse video header"). The authors' envs had ffmpeg 8.0.x and mediapy 1.2.6.
step tapnet_ffmpeg8 bash -c "conda install -y -n tapnet -c conda-forge 'ffmpeg=8.0.0' && conda run -n tapnet pip install mediapy==1.2.6"

# --- sam3d
mk_sam3d_base() {
  conda create -y -n sam3d python=3.11 && conda activate sam3d &&
  conda install -y -c conda-forge ffmpeg &&
  pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 &&
  awk '/- pip:/{f=1;next} f' $R/env/sam3d.yml | sed 's/^ *- //' \
    | grep -vE '^(--extra-index-url|torch==|torchvision==|torchaudio==|triton==|nvidia-|sam3d-objects|diff-gaussian|pytorch3d|kaolin|nvdiffrast|gsplat|moge|utils3d|conda-env-export|warp-lang)' \
    > $W/sam3d_requirements.txt &&
  pip install -r $W/sam3d_requirements.txt
}
step sam3d_base mk_sam3d_base
conda activate sam3d
step sam3d_moge pip install "MoGe @ git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b"
step sam3d_pytorch3d pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
step sam3d_kaolin pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
step sam3d_gsplat pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7"
step sam3d_nvdiffrast pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast.git"
step sam3d_pkg pip install --no-deps -e $R/modules/sam-3d-objects
step sam3d_geocalib pip install "geocalib @ git+https://github.com/cvg/GeoCalib.git"
pip uninstall -y notebook > /dev/null 2>&1
step sam3d_hydra_patch bash -c "cd $R/modules/sam-3d-objects && ls patching && (bash patching/hydra || python patching/hydra)"
# SAM 3D's GLB/texture baking needs mip-splatting's rasterizer (reconstruction/env/README.md).
# Without it, generate_mesh_sam3d.py dies with NameError: GaussianRasterizationSettings (seen 2026-09-26).
step sam3d_diffgauss bash -c "cd $W && (test -d mip-splatting || git clone -q --recursive https://github.com/autonomousvision/mip-splatting.git) && cd mip-splatting/submodules/diff-gaussian-rasterization && python setup.py install"
# Upstream sets this before importing (notebook/inference.py; Fast-SAM3D track_object.py:65);
# without it `import sam3d_objects` looks for a module that is not in the repo.
conda env config vars set LIDRA_SKIP_INIT=true > /dev/null; conda deactivate; conda activate sam3d
step sam3d_import python -c "
import torch, pytorch3d, kaolin, gsplat, nvdiffrast, spconv, moge, geocalib, sam3d_objects
print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
from pytorch3d.renderer import MeshRasterizer; print('pytorch3d', pytorch3d.__version__, 'kaolin', kaolin.__version__)"
echo "=== setup finished $(date +%T)" | tee -a $STATUS
df -h / | tail -1 | tee -a $STATUS
