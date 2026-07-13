#!/usr/bin/env bash
set -euo pipefail

scene=${1:-office0}
exp=${2:-ActiveSem}
gpus=${3:-0}
dry_run=${4:-1}

source /home/chen/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

cmd=(python /home/chen/Desktop/ActiveSGM/src/main/sgm_launcher.py \
  --dataset Replica \
  --scene "$scene" \
  --exp "$exp" \
  --gpus "$gpus" \
  --enable_vis 0)

if [[ "$dry_run" == "1" ]]; then
  cmd+=(--dry_run)
fi

"${cmd[@]}"
