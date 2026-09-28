#!/bin/bash
# R1 ablation: Table 3's four cumulative arms on the shipped whisking inputs, several seeds.
# One run at a time (the 5080 is the only GPU). Each run appends one JSON row to
# ~/rig_work/do_as_i_do/r1_results.jsonl and leaves its trajectory in outputs/.../<data-id>/.
# Usage: bash ablation.sh "0 1 2" "annealed warmup perturb full"
set -o pipefail
SEEDS="${1:-0 1 2}"
ARMS="${2:-annealed warmup perturb full}"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate retargeting
cd "$HOME/rig_work/do_as_i_do/do-as-i-do/retargeting"
declare -A ARM_ID=([annealed]=1 [warmup]=2 [perturb]=3 [full]=4)
for seed in $SEEDS; do
  for arm in $ARMS; do
    did=$((100 + 10 * seed + ARM_ID[$arm]))
    echo "=== arm=$arm seed=$seed data_id=$did $(date +%T) ==="
    python r1_run.py run --arm "$arm" --seed "$seed" --data-id "$did" 2>&1 \
      | grep -E '^\{|Final object tracking error|Early termination|Error|Traceback|OutOfMemory' \
      || echo "!!! arm=$arm seed=$seed failed"
  done
done
echo "=== ablation done $(date +%T) ==="
