#!/usr/bin/env bash
set -euo pipefail

scene=${1:-office0}
gpus=${2:-0}
dry_run=${3:-1}

source /home/chen/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

cmd=(bash /home/chen/Desktop/ActiveSGM/scripts/framework/run_sgm.sh "$scene" ActiveSafe "$gpus" "$dry_run")
"${cmd[@]}"
