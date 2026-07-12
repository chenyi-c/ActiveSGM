from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
local_dir = "/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct"

snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("Downloaded to:", local_dir)
