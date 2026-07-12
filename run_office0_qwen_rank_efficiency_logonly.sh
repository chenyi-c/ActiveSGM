#!/bin/bash
#SBATCH -J qwen_rankeff_log
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

# Enhanced-metrics Qwen log-only mode.
# This mode records Qwen decisions and rank-efficiency guard metrics,
# but does NOT allow Qwen to change final_next_visit.
export ACTIVE_SGM_LLM_MODE=qwen_rank_efficiency_logonly

export QWEN_PLANNER_MODEL_PATH=/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct
export QWEN_PLANNER_DTYPE=float32
export QWEN_PLANNER_TOP_N=3
export QWEN_PLANNER_TIE_GAP=0.10
export QWEN_PLANNER_MAX_NEW_TOKENS=96

# Rank-efficiency guard thresholds for hypothetical analysis.
export QWEN_RANK_EFF_MAX_WEIGHTED_RANK=3
export QWEN_RANK_EFF_MAX_SCORE_DROP=0.09
export QWEN_RANK_EFF_MIN_DISTANCE_SAVING=0.5
export QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO=1.0

RESULT_DIR="results/Replica/office0/ActiveSem/run_qwen_rank_efficiency_logonly_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "===== ENV CHECK ====="
hostname
date
pwd
which python
python --version
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ACTIVE_SGM_LLM_MODE=$ACTIVE_SGM_LLM_MODE"
echo "QWEN_PLANNER_DTYPE=$QWEN_PLANNER_DTYPE"
echo "QWEN_RANK_EFF_MAX_WEIGHTED_RANK=$QWEN_RANK_EFF_MAX_WEIGHTED_RANK"
echo "QWEN_RANK_EFF_MAX_SCORE_DROP=$QWEN_RANK_EFF_MAX_SCORE_DROP"
echo "QWEN_RANK_EFF_MIN_DISTANCE_SAVING=$QWEN_RANK_EFF_MIN_DISTANCE_SAVING"
echo "QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO=$QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO"
echo "RESULT_DIR=$RESULT_DIR"

echo "===== START ActiveSGM Qwen Rank-Efficiency enhanced metrics LOG-ONLY ====="

python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveSem_LLMLog.py \
  --seed 0 \
  --result_dir "$RESULT_DIR" \
  --enable_vis 0
