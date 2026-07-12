#!/bin/bash
#SBATCH -J activesgm_fake_llm_full
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH -t 05:00:00

source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

cd /data/run01/scxj889/projects/ActiveSGM_qwen_planner

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

RESULT_DIR="results/Replica/office0/ActiveSem/run_fake_llm_full_night_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "===== ENV CHECK ====="
hostname
date
pwd
which python
python --version
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "RESULT_DIR=$RESULT_DIR"

echo "===== START ActiveSGM fake LLM full run ====="

python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveSem_LLMLog.py \
  --seed 0 \
  --result_dir "$RESULT_DIR" \
  --enable_vis 0
