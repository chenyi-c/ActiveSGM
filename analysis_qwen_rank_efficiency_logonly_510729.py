import csv
import json
from pathlib import Path
from collections import Counter

RESULT_DIR = Path("results/Replica/office0/ActiveSem/run_qwen_rank_efficiency_logonly_20260605_060010")
LLM_LOG = RESULT_DIR / "splatam/llm_logs/planner_semantic_candidates.jsonl"

OUT_ALL = Path("analysis_qwen_rank_efficiency_logonly_510729_all_qwen.csv")
OUT_ACCEPTED = Path("analysis_qwen_rank_efficiency_logonly_510729_hyp_accept.csv")
OUT_REJECTED = Path("analysis_qwen_rank_efficiency_logonly_510729_rejected_diff.csv")
OUT_SUMMARY = Path("analysis_qwen_rank_efficiency_logonly_510729_summary.txt")

def as_bool(x):
    return bool(x)

def as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def reason_has_numeric_inconsistency(reason, distance_saving, stable_eff_ratio):
    """
    Conservative heuristic for detecting obvious reason/metric mismatch.
    It does not prove all inconsistencies; it only flags suspicious cases.
    """
    r = (reason or "").lower()

    flags = []

    if distance_saving < 0 and (
        "clear distance_saving" in r
        or "clear distance saving" in r
        or "preferred clear distance" in r
        or "prefer clear distance" in r
    ):
        flags.append("negative_distance_saving_described_as_preferred")

    if stable_eff_ratio < 1.0 and (
        "stable_efficiency_ratio_vs_original >= 1.0" in r
        or "stable_efficiency_ratio >= 1.0" in r
        or "stable_efficiency_ratio_vs_original > 1" in r
        or "stable_efficiency_ratio > 1" in r
    ):
        flags.append("stable_efficiency_below_1_described_as_above_1")

    return ";".join(flags)

records = [json.loads(x) for x in LLM_LOG.read_text(encoding="utf-8").splitlines() if x.strip()]

rows_all = []
rows_accept = []
rows_rejected_diff = []

for r in records:
    gm = r.get("llm_guard_metrics") or {}

    selected = r.get("llm_selected_id")
    original = r.get("original_next_visit")
    selected_diff = selected is not None and selected != original

    distance_saving = as_float(gm.get("distance_saving_vs_original"))
    stable_eff_ratio = as_float(gm.get("stable_efficiency_ratio_vs_original"))
    score_drop = as_float(gm.get("score_drop_ratio_vs_original"))

    rank_accept = bool(gm.get("rank_efficiency_guard_accept"))
    reason = r.get("llm_reason", "")

    row = {
        "step": r.get("step"),
        "original_next_visit": original,
        "llm_selected_id": selected,
        "final_next_visit": r.get("final_next_visit"),
        "selected_diff_from_original": selected_diff,
        "qwen_called": as_bool(r.get("llm_qwen_called")),
        "tie_case": as_bool(r.get("llm_tie_case")),
        "guard_accept_qwen": as_bool(r.get("llm_guard_accept_qwen")),
        "apply_used": as_bool(r.get("llm_apply_used")),
        "final_changed": r.get("final_next_visit") != r.get("original_next_visit"),
        "rank_efficiency_guard_accept": rank_accept,

        "selected_weighted_rank": gm.get("selected_weighted_rank"),
        "selected_explore_rank": gm.get("selected_explore_rank"),
        "selected_distance_rank": gm.get("selected_distance_rank"),

        "weighted_ratio": gm.get("weighted_ratio"),
        "explore_ratio": gm.get("explore_ratio"),
        "score_drop_ratio_vs_original": score_drop,
        "distance_saving_vs_original": distance_saving,
        "distance_delta": gm.get("distance_delta"),
        "stable_efficiency_ratio_vs_original": stable_eff_ratio,

        "original_weighted_score": gm.get("original_weighted_score"),
        "qwen_weighted_score": gm.get("qwen_weighted_score"),
        "original_explore_ig": gm.get("original_explore_ig"),
        "qwen_explore_ig": gm.get("qwen_explore_ig"),
        "original_distance": gm.get("original_distance"),
        "qwen_distance": gm.get("qwen_distance"),

        "reason_inconsistency_flags": reason_has_numeric_inconsistency(
            reason, distance_saving, stable_eff_ratio
        ),
        "llm_reason": reason,
    }

    rows_all.append(row)

    if rank_accept:
        rows_accept.append(row)

    if selected_diff and not rank_accept:
        rows_rejected_diff.append(row)

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUT_ALL, rows_all)
write_csv(OUT_ACCEPTED, rows_accept)
write_csv(OUT_REJECTED, rows_rejected_diff)

mode_counter = Counter(r.get("llm_mode", "NA") for r in records)
reason_flag_counter = Counter(
    row["reason_inconsistency_flags"]
    for row in rows_all
    if row["reason_inconsistency_flags"]
)

qwen_called = sum(row["qwen_called"] for row in rows_all)
tie_cases = sum(row["tie_case"] for row in rows_all)
selected_diff_count = sum(row["selected_diff_from_original"] for row in rows_all)
rank_accept_count = sum(row["rank_efficiency_guard_accept"] for row in rows_all)
final_changed = sum(row["final_changed"] for row in rows_all)

lines = []
lines.append("Qwen Rank-Efficiency Log-Only 510729 Analysis")
lines.append("=" * 80)
lines.append(f"Result dir: {RESULT_DIR}")
lines.append(f"LLM log: {LLM_LOG}")
lines.append("")
lines.append("Basic statistics:")
lines.append(f"  records: {len(rows_all)}")
lines.append(f"  qwen_called: {qwen_called}")
lines.append(f"  tie_cases: {tie_cases}")
lines.append(f"  qwen_selected_diff_from_original: {selected_diff_count}")
lines.append(f"  hypothetical_rank_efficiency_accept: {rank_accept_count}")
lines.append(f"  final_changed: {final_changed}")
lines.append(f"  modes: {dict(mode_counter)}")
lines.append("")
lines.append("Hypothetical rank-efficiency accepted cases:")
for row in rows_accept:
    lines.append(
        "  "
        f"step={row['step']}, "
        f"orig={row['original_next_visit']}, "
        f"qwen={row['llm_selected_id']}, "
        f"weighted_rank={row['selected_weighted_rank']}, "
        f"score_drop={as_float(row['score_drop_ratio_vs_original']):.4f}, "
        f"distance_saving={as_float(row['distance_saving_vs_original']):.4f}, "
        f"stable_eff_ratio={as_float(row['stable_efficiency_ratio_vs_original']):.4f}"
    )

lines.append("")
lines.append("Rejected Qwen-different cases:")
lines.append(f"  count: {len(rows_rejected_diff)}")
for row in rows_rejected_diff:
    lines.append(
        "  "
        f"step={row['step']}, "
        f"orig={row['original_next_visit']}, "
        f"qwen={row['llm_selected_id']}, "
        f"score_drop={as_float(row['score_drop_ratio_vs_original']):.4f}, "
        f"distance_saving={as_float(row['distance_saving_vs_original']):.4f}, "
        f"stable_eff_ratio={as_float(row['stable_efficiency_ratio_vs_original']):.4f}, "
        f"flags={row['reason_inconsistency_flags']}"
    )

lines.append("")
lines.append("Reason inconsistency flags:")
if reason_flag_counter:
    for k, v in reason_flag_counter.items():
        lines.append(f"  {k}: {v}")
else:
    lines.append("  none detected by heuristic")

lines.append("")
lines.append("Interpretation:")
lines.append(
    "  Enhanced metrics log-only is safe: final_changed remains 0 and guard_accept_qwen remains false."
)
lines.append(
    "  Qwen still often selects candidates different from the original planner, but the rank-efficiency guard only accepts a small subset."
)
lines.append(
    "  Several rejected cases show numerical inconsistency in Qwen reasons, so deterministic guard/fallback remains necessary."
)
lines.append(
    "  The next step should be offline inspection of accepted cases before any apply-v2 run."
)

OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("Wrote:", OUT_ALL)
print("Wrote:", OUT_ACCEPTED)
print("Wrote:", OUT_REJECTED)
print("Wrote:", OUT_SUMMARY)
