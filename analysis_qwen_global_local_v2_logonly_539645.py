import csv
import json
from pathlib import Path
from collections import Counter

RESULT_DIR = Path("results/Replica/office0/ActiveSem/run_qwen_global_local_v2_logonly_20260608_204421")
LLM_LOG = RESULT_DIR / "splatam/llm_logs/planner_semantic_candidates.jsonl"

OUT_ALL = Path("analysis_qwen_global_local_v2_logonly_539645_all.csv")
OUT_QWEN = Path("analysis_qwen_global_local_v2_logonly_539645_qwen_called.csv")
OUT_WOULD_ACCEPT = Path("analysis_qwen_global_local_v2_logonly_539645_would_accept.csv")
OUT_SUMMARY = Path("analysis_qwen_global_local_v2_logonly_539645_summary.txt")

METRICS = {
    "ate_rmse_cm": 121.31,
    "psnr": 27.72,
    "depth_rmse_cm": 0.52,
    "depth_l1_cm": 0.52,
    "ms_ssim": 0.975,
    "lpips": 0.089,
    "semantic_eval": "0 / dataset 2000",
}

REFERENCE = {
    "rank_eff_logonly": {
        "ate_rmse_cm": 122.69,
        "psnr": 27.75,
        "depth_rmse_cm": 0.53,
        "ms_ssim": 0.974,
        "lpips": 0.092,
    },
    "global_local_v1_logonly": {
        "ate_rmse_cm": 133.64,
        "psnr": 27.67,
        "depth_rmse_cm": 0.52,
        "ms_ssim": 0.971,
        "lpips": 0.100,
    },
}

def b(x):
    return bool(x)

def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def reason_flags(reason, gm):
    r = str(reason or "").lower()
    flags = []

    distance_saving = f(gm.get("distance_saving_vs_original"))
    stable_eff = f(gm.get("stable_efficiency_ratio_vs_original"))
    risk = str(gm.get("trajectory_risk", "missing"))
    raw_should = b(gm.get("qwen_should_change_original_raw"))
    corrected_should = b(gm.get("should_change_original"))

    if risk == "high" and raw_should:
        flags.append("raw_should_change_true_with_high_risk")

    if corrected_should and not b(gm.get("rank_efficiency_guard_accept")):
        flags.append("corrected_should_change_true_without_rank_guard")

    if distance_saving < 0 and "distance_saving" in r and ("preferred" in r or "clear" in r):
        flags.append("negative_distance_saving_described_as_preferred")

    if stable_eff < 1.0 and "stable_efficiency" in r and (">= 1" in r or "> 1" in r):
        flags.append("stable_efficiency_below_1_described_as_above_1")

    return ";".join(flags)

records = [json.loads(x) for x in LLM_LOG.read_text(encoding="utf-8").splitlines() if x.strip()]

rows = []
for r in records:
    gm = r.get("llm_guard_metrics") or {}
    qwen_called = b(r.get("llm_qwen_called"))
    final_changed = r.get("final_next_visit") != r.get("original_next_visit")
    selected_diff = r.get("llm_selected_id") != r.get("original_next_visit")

    row = {
        "step": r.get("step"),
        "original_next_visit": r.get("original_next_visit"),
        "llm_selected_id": r.get("llm_selected_id"),
        "final_next_visit": r.get("final_next_visit"),
        "qwen_called": qwen_called,
        "tie_case": b(r.get("llm_tie_case")),
        "guard_accept_qwen": b(r.get("llm_guard_accept_qwen")),
        "apply_used": b(r.get("llm_apply_used")),
        "final_changed": final_changed,
        "qwen_selected_diff_from_original": selected_diff,

        "global_local_alignment": gm.get("global_local_alignment", "missing"),
        "trajectory_risk": gm.get("trajectory_risk", "missing"),
        "decision_confidence": gm.get("decision_confidence", "missing"),

        "qwen_should_change_original_raw": b(gm.get("qwen_should_change_original_raw")),
        "should_change_original": b(gm.get("should_change_original")),
        "global_local_consistency_pass": b(gm.get("global_local_consistency_pass")),
        "global_local_v2_would_accept": b(gm.get("global_local_v2_would_accept")),
        "global_local_v2_reject_reason": gm.get("global_local_v2_reject_reason", "missing"),
        "rank_efficiency_guard_accept": b(gm.get("rank_efficiency_guard_accept")),

        "selected_weighted_rank": gm.get("selected_weighted_rank"),
        "selected_explore_rank": gm.get("selected_explore_rank"),
        "selected_distance_rank": gm.get("selected_distance_rank"),
        "score_drop_ratio_vs_original": gm.get("score_drop_ratio_vs_original"),
        "distance_saving_vs_original": gm.get("distance_saving_vs_original"),
        "stable_efficiency_ratio_vs_original": gm.get("stable_efficiency_ratio_vs_original"),

        "exploration_stage": (gm.get("global_state") or {}).get("exploration_stage", "missing"),
        "candidate_count": (gm.get("global_state") or {}).get("candidate_count", "missing"),
        "planner_uncertainty": (gm.get("global_state") or {}).get("planner_uncertainty", "missing"),

        "reason_flags": reason_flags(r.get("llm_reason", ""), gm),
        "llm_reason": r.get("llm_reason", ""),
        "raw_response": r.get("llm_raw_response", ""),
    }
    rows.append(row)

qwen_rows = [r for r in rows if r["qwen_called"]]
would_accept_rows = [r for r in rows if r["global_local_v2_would_accept"]]

def write_csv(path, data):
    if not data:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)

write_csv(OUT_ALL, rows)
write_csv(OUT_QWEN, qwen_rows)
write_csv(OUT_WOULD_ACCEPT, would_accept_rows)

def count(field, data):
    return Counter(str(r.get(field, "missing")) for r in data)

unknown_all = [
    r for r in rows
    if r["global_local_alignment"] in ["missing", "unknown"]
    or r["trajectory_risk"] in ["missing", "unknown"]
    or r["decision_confidence"] in ["missing", "unknown"]
]
unknown_qwen = [
    r for r in qwen_rows
    if r["global_local_alignment"] in ["missing", "unknown"]
    or r["trajectory_risk"] in ["missing", "unknown"]
    or r["decision_confidence"] in ["missing", "unknown"]
]

lines = []
lines.append("Qwen Global-Local V2 Log-Only 539645 Analysis")
lines.append("=" * 80)
lines.append(f"Result dir: {RESULT_DIR}")
lines.append(f"LLM log: {LLM_LOG}")
lines.append("")

lines.append("Final metrics:")
for k, v in METRICS.items():
    lines.append(f"  {k}: {v}")

lines.append("")
lines.append("Comparison with previous log-only runs:")
for ref_name, ref in REFERENCE.items():
    lines.append(f"  vs {ref_name}:")
    for k in ["ate_rmse_cm", "psnr", "depth_rmse_cm", "ms_ssim", "lpips"]:
        lines.append(f"    {k}: ref={ref[k]}, v2={METRICS[k]}, delta={METRICS[k] - ref[k]:+.4f}")

lines.append("")
lines.append("Basic statistics:")
lines.append(f"  records: {len(rows)}")
lines.append(f"  qwen_called: {sum(r['qwen_called'] for r in rows)}")
lines.append(f"  tie_cases: {sum(r['tie_case'] for r in rows)}")
lines.append(f"  guard_accept_qwen: {sum(r['guard_accept_qwen'] for r in rows)}")
lines.append(f"  apply_used: {sum(r['apply_used'] for r in rows)}")
lines.append(f"  final_changed: {sum(r['final_changed'] for r in rows)}")
lines.append(f"  qwen_selected_diff_from_original: {sum(r['qwen_selected_diff_from_original'] for r in rows)}")
lines.append(f"  qwen_should_change_original_raw: {sum(r['qwen_should_change_original_raw'] for r in rows)}")
lines.append(f"  corrected_should_change_original: {sum(r['should_change_original'] for r in rows)}")
lines.append(f"  rank_efficiency_guard_accept: {sum(r['rank_efficiency_guard_accept'] for r in rows)}")
lines.append(f"  global_local_consistency_pass: {sum(r['global_local_consistency_pass'] for r in rows)}")
lines.append(f"  global_local_v2_would_accept: {sum(r['global_local_v2_would_accept'] for r in rows)}")

lines.append("")
lines.append("Global-local v2 distributions over all records:")
lines.append(f"  alignment: {dict(count('global_local_alignment', rows))}")
lines.append(f"  trajectory_risk: {dict(count('trajectory_risk', rows))}")
lines.append(f"  decision_confidence: {dict(count('decision_confidence', rows))}")
lines.append(f"  reject_reason: {dict(count('global_local_v2_reject_reason', rows))}")
lines.append(f"  exploration_stage: {dict(count('exploration_stage', rows))}")

lines.append("")
lines.append("Global-local v2 distributions over qwen_called records only:")
lines.append(f"  alignment: {dict(count('global_local_alignment', qwen_rows))}")
lines.append(f"  trajectory_risk: {dict(count('trajectory_risk', qwen_rows))}")
lines.append(f"  decision_confidence: {dict(count('decision_confidence', qwen_rows))}")
lines.append(f"  reject_reason: {dict(count('global_local_v2_reject_reason', qwen_rows))}")
lines.append(f"  exploration_stage: {dict(count('exploration_stage', qwen_rows))}")

lines.append("")
lines.append("Unknown-field diagnosis:")
lines.append(f"  records_with_unknown_or_missing_fields_all: {len(unknown_all)}")
lines.append(f"  records_with_unknown_or_missing_fields_qwen_called: {len(unknown_qwen)}")

lines.append("")
lines.append("Reason / consistency flags:")
flag_counter = Counter(r["reason_flags"] for r in rows if r["reason_flags"])
if flag_counter:
    for k, v in flag_counter.items():
        lines.append(f"  {k}: {v}")
else:
    lines.append("  none detected by heuristic")

lines.append("")
lines.append("Global-local v2 would-accept rows:")
if would_accept_rows:
    for r in would_accept_rows:
        lines.append(
            "  "
            f"step={r['step']}, orig={r['original_next_visit']}, selected={r['llm_selected_id']}, "
            f"alignment={r['global_local_alignment']}, risk={r['trajectory_risk']}, "
            f"confidence={r['decision_confidence']}, score_drop={r['score_drop_ratio_vs_original']}, "
            f"distance_saving={r['distance_saving_vs_original']}, stable_eff={r['stable_efficiency_ratio_vs_original']}"
        )
else:
    lines.append("  none")

lines.append("")
lines.append("Interpretation:")
lines.append("  Global-local v2 log-only safety worked: final_changed is 0 and apply_used is 0.")
lines.append("  v2 removed the v1-style structural contradiction by post-filtering Qwen outputs through deterministic consistency gates.")
lines.append("  However, v2 is overly conservative: raw should_change_original is 0 and v2 would_accept is 0.")
lines.append("  The good final metrics cannot be attributed to Qwen trajectory changes because no trajectory change was applied.")
lines.append("  Next step: inspect qwen_called rows and design a softer but still consistent global-local v3 or run repeat log-only controls to estimate variance.")

OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("Wrote:", OUT_ALL)
print("Wrote:", OUT_QWEN)
print("Wrote:", OUT_WOULD_ACCEPT)
print("Wrote:", OUT_SUMMARY)
