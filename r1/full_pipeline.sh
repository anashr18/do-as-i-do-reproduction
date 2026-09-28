#!/bin/bash
# R1 last step: run all five retargeting stages from the shipped reconstruction output
# (not the authors' prepared scene), then compare what stages 1-4.5 regenerate with what
# the authors shipped, and score stage 5 against the regenerated reference.
set -eo pipefail
DID="${1:-200}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate retargeting
cd "$HOME/rig_work/do_as_i_do/do-as-i-do/retargeting"
T0=$(date +%s)
python launch.py --task whisking --raw-dir ../reconstruction/whisking --data-id "$DID" \
  --no-show-viewer --no-wait-on-finish 2>&1 \
  | grep -E "Stage|stage|Saved|Final object|Early termination|Traceback|Error|coacd|parts|IK" | grep -v WARNING | cut -c1-240
echo "full_pipeline_wall_s=$(( $(date +%s) - T0 ))"
python r1_run.py score --keep-inputs --data-id "$DID" \
  --npz "outputs/sharpa/right/whisking/$DID/trajectory_mjwp.npz" \
  --out "$HOME/rig_work/do_as_i_do/r1_results.jsonl" 2>&1 | grep '^{'
python - "$DID" <<'EOF'
import sys, numpy as np, pathlib
did = sys.argv[1]
ship = pathlib.Path.home() / "rig_work/do_as_i_do/shipped_outputs/sharpa/right/whisking/0"
regen = pathlib.Path(f"outputs/sharpa/right/whisking/{did}")
a, b = np.load(ship / "trajectory_kinematic.npz"), np.load(regen / "trajectory_kinematic.npz")
print("kinematic keys shipped", a.files, "regen", b.files)
for k in a.files:
    if k in b.files and a[k].shape == b[k].shape and a[k].dtype.kind == "f":
        d = np.abs(a[k] - b[k])
        print(f"  {k} {a[k].shape}: max|diff| {d.max():.3e} mean {d.mean():.3e}")
    elif k in b.files:
        print(f"  {k}: shipped {a[k].shape} regen {b[k].shape}")
s1, s2 = (ship / "scene.xml").read_text(), (regen / "scene.xml").read_text()
print("scene.xml identical:", s1 == s2, "| lines", len(s1.splitlines()), len(s2.splitlines()),
      "| mesh refs", s1.count("<mesh "), s2.count("<mesh "))
EOF
# Stage 2 overwrote the shared object assets. Compare, then put the authors' copy back so
# later shipped-input runs keep using the meshes their scene.xml was built with.
SHIP_OBJ="$HOME/rig_work/do_as_i_do/shipped_outputs/assets/objects/whisking"
echo "convex parts: shipped $(ls "$SHIP_OBJ/convex" | wc -l) regen $(ls outputs/assets/objects/whisking/convex | wc -l)"
mkdir -p "$HOME/rig_work/do_as_i_do/regen_assets"
rm -rf "$HOME/rig_work/do_as_i_do/regen_assets/whisking"
cp -a outputs/assets/objects/whisking "$HOME/rig_work/do_as_i_do/regen_assets/whisking"
rm -rf outputs/assets/objects/whisking && cp -a "$SHIP_OBJ" outputs/assets/objects/whisking
echo "restored shipped object assets; regenerated copy kept in regen_assets/whisking"
