#!/bin/bash
#SBATCH -J qwen_json_v2
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -t 00:20:00

source ~/run/miniconda3/etc/profile.d/conda.sh
conda activate activegamer

cd /data/run01/scxj889/projects/ActiveSGM_qwen_planner

export PYTHONPATH=/data/run01/scxj889/qwen_pydeps:$PYTHONPATH
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "===== ENV CHECK ====="
hostname
date
pwd
which python
python --version
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

python - <<'PY'
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct"

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
)

print("eos_token:", tok.eos_token, tok.eos_token_id)
print("pad_token:", tok.pad_token, tok.pad_token_id)

print("Loading model in float32 for stability...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to("cuda:0")

model.eval()

candidates = [
    {"id": 0, "distance": 0.64, "explore_ig": 816000, "semantic_entropy": 0.202, "weighted_score": 0.0389},
    {"id": 1, "distance": 1.20, "explore_ig": 519145, "semantic_entropy": 0.198, "weighted_score": 0.0243},
    {"id": 2, "distance": 2.00, "explore_ig": 753731, "semantic_entropy": 0.201, "weighted_score": 0.0357},
]

prompt = (
    "You are a robot next-best-view planning assistant.\n"
    "Choose exactly one candidate from the list.\n"
    "Return only one JSON object. No markdown. No extra text.\n\n"
    "Candidates:\n"
    f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
    "JSON output format:\n"
    '{"selected_candidate_id": 0, "reason": "short reason"}\n'
)

messages = [
    {"role": "system", "content": "You are a strict JSON generator. You must output valid JSON only."},
    {"role": "user", "content": prompt},
]

text = tok.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("===== PROMPT TAIL =====")
print(text[-1000:])

inputs = tok([text], return_tensors="pt").to("cuda:0")

eos_ids = []
if tok.eos_token_id is not None:
    eos_ids.append(tok.eos_token_id)

im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
    eos_ids.append(im_end_id)

print("eos_ids:", eos_ids)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=eos_ids if eos_ids else None,
        pad_token_id=tok.eos_token_id,
    )

generated = out[0][inputs.input_ids.shape[-1]:]
resp = tok.decode(generated, skip_special_tokens=True)

print("===== RAW RESPONSE =====")
print(resp)

m = re.search(r"\{.*?\}", resp, flags=re.S)
if not m:
    raise RuntimeError("No JSON object found in Qwen response")

obj = json.loads(m.group(0))

print("===== PARSED JSON =====")
print(obj)

assert "selected_candidate_id" in obj
assert isinstance(obj["selected_candidate_id"], int)
assert obj["selected_candidate_id"] in [0, 1, 2]

print("QWEN JSON TEST V2 PASSED")
PY
