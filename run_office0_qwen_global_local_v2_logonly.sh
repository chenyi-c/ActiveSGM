#!/bin/bash
#SBATCH -J qwen_glv2_log
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH -t 05:00:00

source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

cd /data/run01/scxj889/projects/ActiveSGM_qwen_planner

export PYTHONPATH=/data/run01/scxj889/qwen_pydeps:$PYTHONPATH

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

export HF_HOME=$PWD/data/checkpoint/huggingface
export TORCH_HOME=$PWD/data/checkpoint/torch
export TRANSFORMERS_CACHE=$HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_SILENT=true

# Stricter global-local Qwen v2 log-only mode.
# This mode records Qwen's structured global-local judgment,
# then applies post-hoc consistency checks, but does NOT change trajectory.
export ACTIVE_SGM_LLM_MODE=qwen_global_local_v2_logonly
export ACTIVE_SGM_LLM_APPLY=0

export QWEN_PLANNER_MODEL_PATH=/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct
export QWEN_PLANNER_DTYPE=float32
export QWEN_PLANNER_TOP_N=3
export QWEN_PLANNER_TIE_GAP=0.10
export QWEN_PLANNER_MAX_NEW_TOKENS=128

# Conservative deterministic guard metrics are still computed for hypothetical analysis.
export QWEN_RANK_EFF_MAX_WEIGHTED_RANK=3
export QWEN_RANK_EFF_MAX_SCORE_DROP=0.09
export QWEN_RANK_EFF_MIN_DISTANCE_SAVING=0.8
export QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO=1.1

RESULT_DIR="results/Replica/office0/ActiveSem/run_qwen_global_local_v2_logonly_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "===== ENV CHECK ====="
hostname
date
pwd
which python
python --version
nvidia-smi
echo "ACTIVE_SGM_LLM_MODE=$ACTIVE_SGM_LLM_MODE"
echo "ACTIVE_SGM_LLM_APPLY=$ACTIVE_SGM_LLM_APPLY"
echo "QWEN_PLANNER_DTYPE=$QWEN_PLANNER_DTYPE"
echo "QWEN_PLANNER_MAX_NEW_TOKENS=$QWEN_PLANNER_MAX_NEW_TOKENS"
echo "QWEN_RANK_EFF_MIN_DISTANCE_SAVING=$QWEN_RANK_EFF_MIN_DISTANCE_SAVING"
echo "QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO=$QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO"
echo "RESULT_DIR=$RESULT_DIR"

echo "===== START ActiveSGM Qwen Global-Local V2 LOG-ONLY ====="

python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveSem_LLMLog.py \
  --seed 0 \
  --result_dir "$RESULT_DIR" \
  --enable_vis 0
