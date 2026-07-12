#!/bin/bash
#SBATCH -J activesgm_llm_log
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH -t 00:40:00

source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

cd /data/run01/scxj889/projects/ActiveSGM_qwen_planner

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export HF_HOME=$PWD/data/checkpoint/huggingface
export TORCH_HOME=$PWD/data/checkpoint/torch
export TRANSFORMERS_CACHE=$HF_HOME
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
export WANDB_SILENT=true
mkdir -p "$HF_HOME"

echo "===== ENV CHECK ====="
hostname
date
pwd
which python
python --version
nvidia-smi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS=$SLURM_STEP_GPUS"

python - <<'PY'
import os
import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("SLURM_JOB_GPUS:", os.environ.get("SLURM_JOB_GPUS"))
print("SLURM_STEP_GPUS:", os.environ.get("SLURM_STEP_GPUS"))

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("device", i, torch.cuda.get_device_name(i))
    x = torch.zeros(1, device="cuda:0")
    print("cuda:0 tensor test OK:", x)
PY

echo "===== START ActiveSGM office0 LLM LOG TEST ====="
mkdir -p results/Replica/office0/ActiveSem/run_fake_llm_test_fix

python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveSem_LLMLog.py \
  --seed 0 \
  --result_dir results/Replica/office0/ActiveSem/run_fake_llm_test_fix \
  --enable_vis 0
