import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

RUNS = {
    "log-only": {
        "result_dir": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_logonly_20260529_192548"),
        "llm_log": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_logonly_20260529_192548/splatam/llm_logs/planner_semantic_candidates.jsonl"),
        "mode_family": "top3_tiebreak",
        "apply_type": "log_only",
        "metrics": {
            "ate_rmse_cm": 122.69,
            "psnr": 27.75,
            "depth_rmse_cm": 0.53,
            "ms_ssim": 0.974,
            "lpips": 0.092,
        },
    },
    "apply-v1": {
        "result_dir": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_apply_20260530_130614"),
        "llm_log": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_apply_20260530_130614/splatam/llm_logs/planner_semantic_candidates.jsonl"),
        "mode_family": "top3_tiebreak",
        "apply_type": "apply",
        "metrics": {
            "ate_rmse_cm": 120.37,
            "psnr": 27.68,
            "depth_rmse_cm": 0.83,
            "ms_ssim": 0.973,
            "lpips": 0.088,
        },
    },
    "strict-v1": {
        "result_dir": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_apply_strict_v1_20260602_152234"),
        "llm_log": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_apply_strict_v1_20260602_152234/splatam/llm_logs/planner_semantic_candidates.jsonl"),
        "mode_family": "top3_tiebreak_strict",
        "apply_type": "apply",
        "metrics": {
            "ate_rmse_cm": 135.43,
            "psnr": 27.52,
            "depth_rmse_cm": 1.19,
            "ms_ssim": 0.968,
            "lpips": 0.100,
        },
    },
    "rank-eff-logonly": {
        "result_dir": Path("results/Replica/office0/ActiveSem/run_qwen_rank_efficiency_logonly_20260605_060010"),
        "llm_log": Path("results/Replica/office0/ActiveSem/run_qwen_rank_efficiency_logonly_20260605_060010/splatam/llm_logs/planner_semantic_candidates.jsonl"),
        "mode_family": "rank_efficiency",
        "apply_type": "log_only",
        "metrics": {
            "ate_rmse_cm": 122.69,
            "psnr": 27.75,
            "depth_rmse_cm": 0.53,
            "ms_ssim": 0.974,
            "lpips": 0.092,
        },
    },
    "apply-v2": {
        "result_dir": Path("results/Replica/office0/ActiveSem/run_qwen_rank_efficiency_apply_v2_20260605_233557"),
        "llm_log": Path("results/Replica/office0/ActiveSem/run_qwen_rank_efficiency_apply_v2_20260605_233557/splatam/llm_logs/planner_semantic_candidates.jsonl"),
        "mode_family": "rank_efficiency",
        "apply_type": "apply",
        "metrics": {
            "ate_rmse_cm": 130.40,
            "psnr": 27.71,
            "depth_rmse_cm": 0.61,
            "ms_ssim": 0.972,
            "lpips": 0.094,
        },
    },
}

OUT_ALL = Path("analysis_view_decision_graph_all.csv")
OUT_CHANGED = Path("analysis_view_decision_graph_changed.csv")
OUT_SUMMARY = Path("analysis_view_decision_graph_summary.txt")

def as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def as_bool(x):
    return bool(x)

def stage_from_step(step):
    try:
        step = int(step)
    except Exception:
        return "unknown"
    if step < 200:
        return "early"
    if step < 500:
        return "middle"
    return "late"

def get_gm_value(gm, key):
    if gm is None:
        return None
    return gm.get(key)

all_rows = []
changed_rows = []

for run_name, cfg in RUNS.items():
    llm_log = cfg["llm_log"]
    if not llm_log.exists():
        print(f"WARNING missing log: {run_name}: {llm_log}")
        continue

    records = [json.loads(x) for x in llm_log.read_text(encoding="utf-8").splitlines() if x.strip()]
    metrics = cfg["metrics"]

    for r in records:
        gm = r.get("llm_guard_metrics") or {}
        step = r.get("step")
        original = r.get("original_next_visit")
        selected = r.get("llm_selected_id")
        final = r.get("final_next_visit")

        final_changed = final != original
        qwen_selected_diff = selected is not None and selected != original

        row = {
            "run": run_name,
            "mode_family": cfg["mode_family"],
            "apply_type": cfg["apply_type"],
            "llm_mode": r.get("llm_mode"),
            "step": step,
            "stage": stage_from_step(step),

            "original_next_visit": original,
            "llm_selected_id": selected,
            "final_next_visit": final,
            "qwen_selected_diff_from_original": qwen_selected_diff,
            "final_changed": final_changed,

            "qwen_called": as_bool(r.get("llm_qwen_called")),
            "tie_case": as_bool(r.get("llm_tie_case")),
            "guard_accept_qwen": as_bool(r.get("llm_guard_accept_qwen")),
            "apply_used": as_bool(r.get("llm_apply_used")),

            "selected_weighted_rank": get_gm_value(gm, "selected_weighted_rank"),
            "selected_explore_rank": get_gm_value(gm, "selected_explore_rank"),
            "selected_distance_rank": get_gm_value(gm, "selected_distance_rank"),

            "weighted_ratio": get_gm_value(gm, "weighted_ratio"),
            "explore_ratio": get_gm_value(gm, "explore_ratio"),
            "score_drop_ratio_vs_original": get_gm_value(gm, "score_drop_ratio_vs_original"),
            "distance_saving_vs_original": get_gm_value(gm, "distance_saving_vs_original"),
            "distance_delta": get_gm_value(gm, "distance_delta"),
            "stable_efficiency_ratio_vs_original": get_gm_value(gm, "stable_efficiency_ratio_vs_original"),

            "original_weighted_score": get_gm_value(gm, "original_weighted_score"),
            "qwen_weighted_score": get_gm_value(gm, "qwen_weighted_score"),
            "original_explore_ig": get_gm_value(gm, "original_explore_ig"),
            "qwen_explore_ig": get_gm_value(gm, "qwen_explore_ig"),
            "original_distance": get_gm_value(gm, "original_distance"),
            "qwen_distance": get_gm_value(gm, "qwen_distance"),

            "run_ate_rmse_cm": metrics["ate_rmse_cm"],
            "run_psnr": metrics["psnr"],
            "run_depth_rmse_cm": metrics["depth_rmse_cm"],
            "run_ms_ssim": metrics["ms_ssim"],
            "run_lpips": metrics["lpips"],

            "llm_reason": r.get("llm_reason", ""),
        }

        all_rows.append(row)
        if final_changed:
            changed_rows.append(row)

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUT_ALL, all_rows)
write_csv(OUT_CHANGED, changed_rows)

by_run = defaultdict(list)
by_run_stage = defaultdict(list)
for row in all_rows:
    by_run[row["run"]].append(row)
    by_run_stage[(row["run"], row["stage"])].append(row)

lines = []
lines.append("ActiveSGM View-Decision Graph Summary")
lines.append("=" * 80)
lines.append(f"Total decision records: {len(all_rows)}")
lines.append(f"Total changed records: {len(changed_rows)}")
lines.append("")

lines.append("Run-level statistics:")
for run_name in RUNS:
    rows = by_run.get(run_name, [])
    if not rows:
        lines.append(f"  {run_name}: missing or empty")
        continue

    changed = [r for r in rows if r["final_changed"]]
    qwen_called = sum(r["qwen_called"] for r in rows)
    selected_diff = sum(r["qwen_selected_diff_from_original"] for r in rows)
    guard_accept = sum(r["guard_accept_qwen"] for r in rows)
    apply_used = sum(r["apply_used"] for r in rows)

    metrics = RUNS[run_name]["metrics"]
    lines.append(f"  {run_name}:")
    lines.append(f"    records: {len(rows)}")
    lines.append(f"    qwen_called: {qwen_called}")
    lines.append(f"    qwen_selected_diff_from_original: {selected_diff}")
    lines.append(f"    guard_accept_qwen: {guard_accept}")
    lines.append(f"    apply_used: {apply_used}")
    lines.append(f"    final_changed: {len(changed)}")
    lines.append(f"    changed_steps: {[int(r['step']) for r in changed]}")
    lines.append(
        f"    metrics: ATE={metrics['ate_rmse_cm']}, PSNR={metrics['psnr']}, "
        f"Depth={metrics['depth_rmse_cm']}, MS-SSIM={metrics['ms_ssim']}, LPIPS={metrics['lpips']}"
    )

lines.append("")
lines.append("Changed records by stage:")
stage_counter = Counter(r["stage"] for r in changed_rows)
for stage in ["early", "middle", "late", "unknown"]:
    lines.append(f"  {stage}: {stage_counter.get(stage, 0)}")

lines.append("")
lines.append("Changed records details:")
for r in changed_rows:
    lines.append(
        "  "
        f"run={r['run']}, "
        f"step={r['step']}, "
        f"stage={r['stage']}, "
        f"orig={r['original_next_visit']}, "
        f"selected={r['llm_selected_id']}, "
        f"final={r['final_next_visit']}, "
        f"score_drop={as_float(r['score_drop_ratio_vs_original']):.4f}, "
        f"distance_saving={as_float(r['distance_saving_vs_original']):.4f}, "
        f"stable_eff_ratio={as_float(r['stable_efficiency_ratio_vs_original']):.4f}, "
        f"run_ate={r['run_ate_rmse_cm']}, "
        f"run_depth={r['run_depth_rmse_cm']}, "
        f"run_lpips={r['run_lpips']}"
    )

lines.append("")
lines.append("Interpretation:")
lines.append("  This view-decision graph aggregates Qwen-related planner decisions across log-only, apply-v1, strict-v1, rank-efficiency log-only, and apply-v2.")
lines.append("  It is intended to support trajectory-aware analysis rather than single-step threshold tuning.")
lines.append("  The key next question is whether changed decisions differ systematically by stage and whether late-stage changes correlate with worse global metrics.")

OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("Wrote:", OUT_ALL)
print("Wrote:", OUT_CHANGED)
print("Wrote:", OUT_SUMMARY)
