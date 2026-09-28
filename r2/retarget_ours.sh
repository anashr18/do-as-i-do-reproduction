#!/bin/bash
# R2 end to end (pod): all five retargeting stages on OUR reconstruction of the whisking clip,
# scored with R1's harness (all frames and post-warmup). This is the stages 1-5 check R1 could
# not run from the shipped demo, which lacks config.json and gravity.json; our run writes both.
set -eo pipefail
DID="${1:-300}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate retargeting
cd ~/rig_work/do-as-i-do/retargeting
cp ~/rig_work/r2/r1_run.py .
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > ~/rig_work/r2_run/mem_retarget.txt &
MON=$!
T0=$(date +%s)
python launch.py --task whisking --raw-dir ~/rig_work/r2_run/whisking --data-id "$DID" \
  --no-show-viewer --no-wait-on-finish 2>&1 | tr '\r' '\n' \
  | grep -vE "Optimizing|WARNING" | grep -E "Stage|stage|Saved|Final object|Early termination|Traceback|Error|mesh scale|parts|frames|Prepended" | cut -c1-240
kill $MON 2>/dev/null || true
echo "retarget_wall_s=$(( $(date +%s) - T0 )) gpu_peak_mib=$(sort -n ~/rig_work/r2_run/mem_retarget.txt | tail -1)"
python r1_run.py score --keep-inputs --data-id "$DID" \
  --npz "outputs/sharpa/right/whisking/$DID/trajectory_mjwp.npz" --out ~/rig_work/r2_run/r2_retarget.jsonl 2>&1 | grep '^{'
