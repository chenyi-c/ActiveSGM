#!/bin/bash
#SBATCH -J activesgm_o0_main
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH -t 05:00:00

source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

cd ~/run/projects/ActiveSGM

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
which python
python --version
nvidia-smi

echo "===== START ActiveSGM office0 MAIN ONLY ====="
mkdir -p results/Replica/office0/ActiveSem/run_full_semfix

python src/main/activesgm.py \
  --cfg configs/Replica/office0/ActiveSem.py \
  --seed 0 \
  --result_dir results/Replica/office0/ActiveSem/run_full_semfix \
  --enable_vis 0
