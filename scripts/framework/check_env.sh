#!/usr/bin/env bash
set -euo pipefail

source /home/chen/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

python - <<'PY'
import importlib.util as u
import torch

core_mods = [
    'torch',
    'habitat_sim',
    'diff_gaussian_rasterization',
    'mmengine',
    'transformers',
    'mmdet',
    'mmseg',
]
optional_heavy_mods = [
    'channel_rasterization',
    'sparse_channel_rasterization',
    'torch_sparse',
    'pytorch3d',
    'tinycudann',
]

print('core_dependency_check:', {m: bool(u.find_spec(m)) for m in core_mods})
print('optional_heavy_check:', {m: bool(u.find_spec(m)) for m in optional_heavy_mods})
print('torch:', torch.__version__)
print('torch_cuda:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
