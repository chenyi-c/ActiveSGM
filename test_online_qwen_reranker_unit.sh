#!/bin/bash
#SBATCH -J qwen_rerank_unit
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:10:00

source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

cd /data/run01/scxj889/projects/ActiveSGM_qwen_planner

export PYTHONPATH=/data/run01/scxj889/qwen_pydeps:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export ACTIVE_SGM_LLM_MODE=qwen_tiebreak_top3_distance_strict
export QWEN_PLANNER_MODEL_PATH=/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct
export QWEN_PLANNER_DTYPE=float32
export QWEN_PLANNER_TOP_N=3
export QWEN_PLANNER_TIE_GAP=0.10
export QWEN_PLANNER_MIN_WEIGHT_KEEP=0.90
export QWEN_PLANNER_MIN_EXPLORE_KEEP=0.80
export QWEN_PLANNER_MAX_DISTANCE_INCREASE=0.0
export QWEN_PLANNER_MAX_NEW_TOKENS=96

echo "===== ENV CHECK ====="
hostname
date
which python
python --version
nvidia-smi

python - <<'PY'
from src.llm.planner_reranker import fake_llm_rerank

cands = [
    {"id": 0, "distance": 1.0, "explore_ig": 1000.0, "semantic_entropy": 0.10, "weighted_score": 1.000},
    {"id": 1, "distance": 0.5, "explore_ig": 950.0, "semantic_entropy": 0.11, "weighted_score": 0.950},
    {"id": 2, "distance": 2.0, "explore_ig": 940.0, "semantic_entropy": 0.09, "weighted_score": 0.940},
]

r = fake_llm_rerank(cands, original_next_visit=0)

print("===== RESULT =====")
print(r)

assert r["mode"] == "qwen_tiebreak_top3_distance_strict"
assert "selected_candidate_id" in r
assert "guarded_final_id" in r
assert "guard_accept_qwen" in r
assert "qwen_called" in r
assert "tie_case" in r
assert "guard_metrics" in r

print("QWEN RERANKER UNIT TEST PASSED")
PY
