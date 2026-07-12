#!/bin/bash
#SBATCH -J qwen_rankeff_unit
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00

source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

cd /data/run01/scxj889/projects/ActiveSGM_qwen_planner

export PYTHONPATH=/data/run01/scxj889/qwen_pydeps:$PYTHONPATH

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID

export HF_HOME=$PWD/data/checkpoint/huggingface
export TORCH_HOME=$PWD/data/checkpoint/torch
export TRANSFORMERS_CACHE=$HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_SILENT=true

export ACTIVE_SGM_LLM_MODE=qwen_rank_efficiency_logonly
export QWEN_PLANNER_MODEL_PATH=/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct
export QWEN_PLANNER_DTYPE=float32
export QWEN_PLANNER_TOP_N=3
export QWEN_PLANNER_TIE_GAP=0.10
export QWEN_PLANNER_MAX_NEW_TOKENS=96

export QWEN_RANK_EFF_MAX_WEIGHTED_RANK=3
export QWEN_RANK_EFF_MAX_SCORE_DROP=0.09
export QWEN_RANK_EFF_MIN_DISTANCE_SAVING=0.5
export QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO=1.0

echo "===== ENV CHECK ====="
hostname
date
pwd
which python
python --version
nvidia-smi
echo "ACTIVE_SGM_LLM_MODE=$ACTIVE_SGM_LLM_MODE"
echo "QWEN_PLANNER_MODEL_PATH=$QWEN_PLANNER_MODEL_PATH"

python3 - <<'PY'
import json
from src.llm.planner_reranker import fake_llm_rerank

# Synthetic tie case:
# original id=0 has the highest weighted_score,
# candidate id=1 has slightly lower score but clear distance saving and better stable efficiency.
candidates = [
    {
        "id": 0,
        "distance": 2.0,
        "explore_ig": 1000.0,
        "semantic_entropy": 0.50,
        "weighted_score": 1.00,
    },
    {
        "id": 1,
        "distance": 1.0,
        "explore_ig": 990.0,
        "semantic_entropy": 0.55,
        "weighted_score": 0.96,
    },
    {
        "id": 2,
        "distance": 1.6,
        "explore_ig": 970.0,
        "semantic_entropy": 0.60,
        "weighted_score": 0.94,
    },
]

res = fake_llm_rerank(candidates, original_next_visit=0)

print("===== RESULT =====")
print(json.dumps(res, ensure_ascii=False, indent=2))

assert res["mode"] == "qwen_rank_efficiency_logonly"
assert res["tie_case"] is True
assert res["qwen_called"] is True
assert res["guarded_final_id"] == 0
assert res["guard_accept_qwen"] is False
assert isinstance(res.get("raw_response"), str) and len(res["raw_response"]) > 0

gm = res.get("guard_metrics") or {}
assert "rank_efficiency_guard_accept" in gm
assert "selected_weighted_rank" in gm
assert "score_drop_ratio_vs_original" in gm
assert "distance_saving_vs_original" in gm
assert "stable_efficiency_ratio_vs_original" in gm

print("QWEN_RANK_EFFICIENCY_UNIT_TEST_PASSED")
PY
