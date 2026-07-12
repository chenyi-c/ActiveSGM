#!/bin/bash
#SBATCH -J qwen_glv3_unit
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:25:00

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

export ACTIVE_SGM_LLM_MODE=qwen_global_local_v3_logonly
export QWEN_PLANNER_MODEL_PATH=/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct
export QWEN_PLANNER_DTYPE=float32
export QWEN_PLANNER_TOP_N=3
export QWEN_PLANNER_TIE_GAP=0.10
export QWEN_PLANNER_MAX_NEW_TOKENS=128

export QWEN_V3_SOFT_MAX_WEIGHTED_RANK=3
export QWEN_V3_SOFT_MAX_SCORE_DROP=0.09
export QWEN_V3_SOFT_MIN_DISTANCE_SAVING=0.5
export QWEN_V3_SOFT_MIN_STABLE_EFF_RATIO=1.0

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

candidates = [
    {
        "id": 0,
        "distance": 2.0,
        "explore_ig": 1000.0,
        "semantic_entropy": 0.50,
        "weighted_score": 1.00,
        "step": 90,
    },
    {
        "id": 1,
        "distance": 1.0,
        "explore_ig": 990.0,
        "semantic_entropy": 0.55,
        "weighted_score": 0.96,
        "step": 90,
    },
    {
        "id": 2,
        "distance": 1.6,
        "explore_ig": 970.0,
        "semantic_entropy": 0.60,
        "weighted_score": 0.94,
        "step": 90,
    },
]

res = fake_llm_rerank(candidates, original_next_visit=0)

print("===== RESULT =====")
print(json.dumps(res, ensure_ascii=False, indent=2))

assert res["mode"] == "qwen_global_local_v3_logonly"
assert res["tie_case"] is True
assert res["qwen_called"] is True
assert res["guarded_final_id"] == 0
assert res["guard_accept_qwen"] is False
assert isinstance(res.get("raw_response"), str) and len(res["raw_response"]) > 0
assert res["selected_candidate_id"] in [0, 1, 2]

gm = res.get("guard_metrics") or {}

required_keys = [
    "global_state",
    "global_local_alignment",
    "trajectory_risk",
    "decision_confidence",
    "qwen_should_change_original_raw",
    "should_change_original",
    "global_local_consistency_pass",
    "global_local_v3_soft_gate_accept",
    "global_local_v3_would_accept",
    "global_local_v3_reject_reason",
    "global_local_v3_thresholds",
]

for k in required_keys:
    assert k in gm, f"missing guard_metrics key: {k}"

assert gm["global_local_alignment"] in ["low", "medium", "high", "unknown"]
assert gm["trajectory_risk"] in ["low", "medium", "high", "unknown"]
assert gm["decision_confidence"] in ["low", "medium", "high", "unknown"]

global_state = gm["global_state"]
assert global_state["exploration_stage"] == "early"
assert global_state["candidate_count"] == 3

# v3 log-only invariant:
assert gm["should_change_original"] is False

# If v3 would accept, all soft-consistency conditions must hold.
if gm["global_local_v3_would_accept"]:
    assert gm["global_local_consistency_pass"] is True
    assert gm["global_local_v3_soft_gate_accept"] is True
    assert gm["trajectory_risk"] == "low"
    assert gm["global_local_alignment"] in ["medium", "high"]
    assert gm["decision_confidence"] in ["medium", "high"]
    assert res["selected_candidate_id"] != 0
else:
    assert gm["global_local_consistency_pass"] is False

print("QWEN_GLOBAL_LOCAL_V3_UNIT_TEST_PASSED")
PY
