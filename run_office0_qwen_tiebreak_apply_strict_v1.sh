#!/bin/bash
#SBATCH -J activesgm_qwen_strict1
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

# Qwen online apply mode.
export ACTIVE_SGM_LLM_MODE=qwen_tiebreak_top3_distance_strict
export ACTIVE_SGM_LLM_APPLY=1

export QWEN_PLANNER_MODEL_PATH=/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct
export QWEN_PLANNER_DTYPE=float32
export QWEN_PLANNER_TOP_N=3
export QWEN_PLANNER_TIE_GAP=0.10
export QWEN_PLANNER_MIN_WEIGHT_KEEP=0.90
export QWEN_PLANNER_MIN_EXPLORE_KEEP=0.90
export QWEN_PLANNER_MAX_DISTANCE_INCREASE=-0.000001
export QWEN_PLANNER_MAX_NEW_TOKENS=96

RESULT_DIR="results/Replica/office0/ActiveSem/run_qwen_tiebreak_apply_strict_v1_$(date +%Y%m%d_%H%M%S)"
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
echo "ACTIVE_SGM_LLM_APPLY=$ACTIVE_SGM_LLM_APPLY"
echo "QWEN_PLANNER_DTYPE=$QWEN_PLANNER_DTYPE"
echo "RESULT_DIR=$RESULT_DIR"

echo "===== START ActiveSGM Qwen Top3 TieBreak APPLY ====="

python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveSem_LLMLog.py \
  --seed 0 \
  --result_dir "$RESULT_DIR" \
  --enable_vis 0
