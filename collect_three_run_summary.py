import json
import csv
from pathlib import Path

runs = {
    "log-only": {
        "slurm": Path("slurm-489943.out"),
        "llm": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_logonly_20260529_192548/splatam/llm_logs/planner_semantic_candidates.jsonl"),
    },
    "apply-v1": {
        "slurm": Path("slurm-491621.out"),
        "llm": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_apply_20260530_130614/splatam/llm_logs/planner_semantic_candidates.jsonl"),
    },
    "strict-v1": {
        "slurm": Path("slurm-499558.out"),
        "llm": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_apply_strict_v1_20260602_152234/splatam/llm_logs/planner_semantic_candidates.jsonl"),
    },
}

metric_keys = {
    "Final Average ATE RMSE": "ate_rmse_cm",
    "Average PSNR": "psnr",
    "Average Depth RMSE": "depth_rmse_cm",
    "Average Depth L1": "depth_l1_cm",
    "Average MS-SSIM": "ms_ssim",
    "Average LPIPS": "lpips",
}

metric_rows = []
llm_rows = []
changed_rows = []

for name, paths in runs.items():
    lines = paths["slurm"].read_text(encoding="utf-8", errors="ignore").splitlines()

    metric = {"run": name}
    for line in lines:
        for prefix, key in metric_keys.items():
            if prefix in line:
                metric[key] = line.split(":", 1)[-1].strip().split()[0]
        if "semantic eval frames" in line:
            metric["semantic_eval"] = line.split("semantic eval frames:", 1)[-1].strip()

    metric_rows.append(metric)

    records = [json.loads(x) for x in paths["llm"].read_text(encoding="utf-8").splitlines() if x.strip()]
    changed = [r for r in records if r.get("final_next_visit") != r.get("original_next_visit")]

    llm_rows.append({
        "run": name,
        "records": len(records),
        "qwen_called": sum(bool(r.get("llm_qwen_called")) for r in records),
        "tie_cases": sum(bool(r.get("llm_tie_case")) for r in records),
        "guard_accept_qwen": sum(bool(r.get("llm_guard_accept_qwen")) for r in records),
        "apply_enabled": sum(bool(r.get("llm_apply_enabled")) for r in records),
        "apply_used": sum(bool(r.get("llm_apply_used")) for r in records),
        "final_changed": len(changed),
    })

    for r in changed:
        gm = r.get("llm_guard_metrics", {})
        changed_rows.append({
            "run": name,
            "step": r.get("step"),
            "original_next_visit": r.get("original_next_visit"),
            "final_next_visit": r.get("final_next_visit"),
            "weighted_ratio": gm.get("weighted_ratio"),
            "explore_ratio": gm.get("explore_ratio"),
            "distance_delta": gm.get("distance_delta"),
        })

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

write_csv("analysis_three_runs_metrics.csv", metric_rows)
write_csv("analysis_three_runs_llm_stats.csv", llm_rows)
write_csv("analysis_three_runs_changed_steps.csv", changed_rows)

print("Wrote analysis_three_runs_metrics.csv")
print("Wrote analysis_three_runs_llm_stats.csv")
print("Wrote analysis_three_runs_changed_steps.csv")
