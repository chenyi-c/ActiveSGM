import csv
import json
from pathlib import Path
from collections import Counter

RESULT_DIR = Path("results/Replica/office0/ActiveSem/run_qwen_rank_efficiency_apply_v2_20260605_233557")
LLM_LOG = RESULT_DIR / "splatam/llm_logs/planner_semantic_candidates.jsonl"

OUT_ALL = Path("analysis_qwen_rank_efficiency_apply_v2_524227_all_qwen.csv")
OUT_CHANGED = Path("analysis_qwen_rank_efficiency_apply_v2_524227_changed.csv")
OUT_SUMMARY = Path("analysis_qwen_rank_efficiency_apply_v2_524227_summary.txt")

METRICS = {
    "ate_rmse_cm": 130.40,
    "psnr": 27.71,
    "depth_rmse_cm": 0.61,
    "depth_l1_cm": 0.61,
    "ms_ssim": 0.972,
    "lpips": 0.094,
    "semantic_eval": "0 / dataset 2000",
}

BASELINE_LOGONLY = {
    "ate_rmse_cm": 122.69,
    "psnr": 27.75,
    "depth_rmse_cm": 0.53,
    "ms_ssim": 0.974,
    "lpips": 0.092,
}

def as_bool(x):
    return bool(x)

def as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def reason_inconsistency_flags(reason, gm):
    r = (reason or "").lower()
    flags = []

    distance_saving = as_float(gm.get("distance_saving_vs_original"))
    stable_eff_ratio = as_float(gm.get("stable_efficiency_ratio_vs_original"))

    if distance_saving < 0 and "distance_saving" in r and ("preferred" in r or "clear" in r):
        flags.append("negative_distance_saving_described_as_preferred")

    if stable_eff_ratio < 1.0 and "stable_efficiency_ratio" in r and (">= 1" in r or "> 1" in r):
        flags.append("stable_efficiency_below_1_described_as_above_1")

    if distance_saving > 0 and "distance_saving_vs_original (-" in r:
        flags.append("reason_mentions_negative_distance_saving_but_metric_positive")

    return ";".join(flags)

records = [json.loads(x) for x in LLM_LOG.read_text(encoding="utf-8").splitlines() if x.strip()]

rows_all = []
rows_changed = []

for r in records:
    gm = r.get("llm_guard_metrics") or {}
    final_changed = r.get("final_next_visit") != r.get("original_next_visit")

    row = {
        "step": r.get("step"),
        "original_next_visit": r.get("original_next_visit"),
        "llm_selected_id": r.get("llm_selected_id"),
        "final_next_visit": r.get("final_next_visit"),
        "final_changed": final_changed,

        "qwen_called": as_bool(r.get("llm_qwen_called")),
        "tie_case": as_bool(r.get("llm_tie_case")),
        "guard_accept_qwen": as_bool(r.get("llm_guard_accept_qwen")),
        "apply_used": as_bool(r.get("llm_apply_used")),

        "selected_weighted_rank": gm.get("selected_weighted_rank"),
        "selected_explore_rank": gm.get("selected_explore_rank"),
        "selected_distance_rank": gm.get("selected_distance_rank"),

        "weighted_ratio": gm.get("weighted_ratio"),
        "explore_ratio": gm.get("explore_ratio"),
        "score_drop_ratio_vs_original": gm.get("score_drop_ratio_vs_original"),
        "distance_saving_vs_original": gm.get("distance_saving_vs_original"),
        "distance_delta": gm.get("distance_delta"),
        "stable_efficiency_ratio_vs_original": gm.get("stable_efficiency_ratio_vs_original"),

        "original_weighted_score": gm.get("original_weighted_score"),
        "qwen_weighted_score": gm.get("qwen_weighted_score"),
        "original_explore_ig": gm.get("original_explore_ig"),
        "qwen_explore_ig": gm.get("qwen_explore_ig"),
        "original_distance": gm.get("original_distance"),
        "qwen_distance": gm.get("qwen_distance"),

        "reason_inconsistency_flags": reason_inconsistency_flags(r.get("llm_reason", ""), gm),
        "llm_reason": r.get("llm_reason", ""),
    }

    rows_all.append(row)
    if final_changed:
        rows_changed.append(row)

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUT_ALL, rows_all)
write_csv(OUT_CHANGED, rows_changed)

mode_counter = Counter(r.get("llm_mode", "NA") for r in records)
flag_counter = Counter(row["reason_inconsistency_flags"] for row in rows_all if row["reason_inconsistency_flags"])

lines = []
lines.append("Qwen Rank-Efficiency Apply-v2 524227 Analysis")
lines.append("=" * 80)
lines.append(f"Result dir: {RESULT_DIR}")
lines.append(f"LLM log: {LLM_LOG}")
lines.append("")

lines.append("Final metrics:")
for k, v in METRICS.items():
    lines.append(f"  {k}: {v}")

lines.append("")
lines.append("Comparison with enhanced log-only:")
for k, v in BASELINE_LOGONLY.items():
    apply_v = METRICS.get(k)
    if isinstance(apply_v, (int, float)):
        delta = apply_v - v
        lines.append(f"  {k}: log-only={v}, apply-v2={apply_v}, delta={delta:+.4f}")

lines.append("")
lines.append("LLM statistics:")
lines.append(f"  records: {len(rows_all)}")
lines.append(f"  qwen_called: {sum(row['qwen_called'] for row in rows_all)}")
lines.append(f"  tie_cases: {sum(row['tie_case'] for row in rows_all)}")
lines.append(f"  guard_accept_qwen: {sum(row['guard_accept_qwen'] for row in rows_all)}")
lines.append(f"  apply_used: {sum(row['apply_used'] for row in rows_all)}")
lines.append(f"  final_changed: {sum(row['final_changed'] for row in rows_all)}")
lines.append(f"  modes: {dict(mode_counter)}")

lines.append("")
lines.append("Changed decisions:")
for row in rows_changed:
    lines.append(
        "  "
        f"step={row['step']}, "
        f"orig={row['original_next_visit']}, "
        f"selected={row['llm_selected_id']}, "
        f"final={row['final_next_visit']}, "
        f"score_drop={as_float(row['score_drop_ratio_vs_original']):.4f}, "
        f"distance_saving={as_float(row['distance_saving_vs_original']):.4f}, "
        f"stable_eff_ratio={as_float(row['stable_efficiency_ratio_vs_original']):.4f}, "
        f"flags={row['reason_inconsistency_flags']}"
    )

lines.append("")
lines.append("Reason inconsistency flags:")
if flag_counter:
    for k, v in flag_counter.items():
        lines.append(f"  {k}: {v}")
else:
    lines.append("  none detected by heuristic")

lines.append("")
lines.append("Interpretation:")
lines.append("  Apply-v2 successfully changed the trajectory under conservative rank-efficiency guard.")
lines.append("  However, final metrics are worse than enhanced log-only, especially ATE and LPIPS.")
lines.append("  This confirms that local candidate-level benefits do not reliably translate into global trajectory-level gains.")
lines.append("  The next stage should build a view-decision graph and move toward global-local / trajectory-aware Qwen planning.")

OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("Wrote:", OUT_ALL)
print("Wrote:", OUT_CHANGED)
print("Wrote:", OUT_SUMMARY)
