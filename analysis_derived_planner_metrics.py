import json
import csv
from pathlib import Path

RUNS = {
    "log-only": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_logonly_20260529_192548/splatam/llm_logs/planner_semantic_candidates.jsonl"),
    "apply-v1": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_apply_20260530_130614/splatam/llm_logs/planner_semantic_candidates.jsonl"),
    "strict-v1": Path("results/Replica/office0/ActiveSem/run_qwen_tiebreak_apply_strict_v1_20260602_152234/splatam/llm_logs/planner_semantic_candidates.jsonl"),
}

OUT_ALL = Path("analysis_derived_planner_metrics_all_qwen.csv")
OUT_CHANGED = Path("analysis_derived_planner_metrics_changed_steps.csv")
OUT_SUMMARY = Path("analysis_derived_planner_metrics_summary.txt")

EPS = 1e-9


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def get_candidate(records, cand_id):
    for c in records:
        if int(c.get("id", -999999)) == int(cand_id):
            return c
    return None


def rank_map(candidates, key, reverse=True):
    # reverse=True: larger is better
    ordered = sorted(
        candidates,
        key=lambda c: to_float(c.get(key, 0.0)),
        reverse=reverse,
    )
    return {int(c["id"]): i + 1 for i, c in enumerate(ordered)}


def enrich_record(run_name, r):
    candidates = r.get("candidates", [])
    if not candidates:
        return None

    original_id = int(r.get("original_next_visit"))
    final_id = int(r.get("final_next_visit"))
    selected_id = int(r.get("llm_selected_id", final_id))

    original = get_candidate(candidates, original_id)
    selected = get_candidate(candidates, selected_id)
    final = get_candidate(candidates, final_id)

    if original is None or selected is None:
        return None

    weighted_ranks = rank_map(candidates, "weighted_score", reverse=True)
    explore_ranks = rank_map(candidates, "explore_ig", reverse=True)
    distance_ranks = rank_map(candidates, "distance", reverse=False)

    orig_weight = to_float(original.get("weighted_score"))
    sel_weight = to_float(selected.get("weighted_score"))

    orig_explore = to_float(original.get("explore_ig"))
    sel_explore = to_float(selected.get("explore_ig"))

    orig_dist = to_float(original.get("distance"))
    sel_dist = to_float(selected.get("distance"))

    orig_eff = orig_explore / (orig_dist + EPS)
    sel_eff = sel_explore / (sel_dist + EPS)

    row = {
        "run": run_name,
        "step": r.get("step"),
        "num_candidates": r.get("num_candidates"),
        "original_next_visit": original_id,
        "llm_selected_id": selected_id,
        "final_next_visit": final_id,
        "changed_final": final_id != original_id,
        "llm_qwen_called": bool(r.get("llm_qwen_called")),
        "llm_guard_accept_qwen": bool(r.get("llm_guard_accept_qwen")),
        "llm_apply_used": bool(r.get("llm_apply_used")),

        "original_weighted_score": orig_weight,
        "selected_weighted_score": sel_weight,
        "weighted_ratio_vs_original": sel_weight / orig_weight if abs(orig_weight) > EPS else None,
        "score_drop_ratio_vs_original": 1.0 - (sel_weight / orig_weight) if abs(orig_weight) > EPS else None,
        "selected_weighted_rank": weighted_ranks.get(selected_id),
        "original_weighted_rank": weighted_ranks.get(original_id),

        "original_explore_ig": orig_explore,
        "selected_explore_ig": sel_explore,
        "explore_ratio_vs_original": sel_explore / orig_explore if abs(orig_explore) > EPS else None,
        "selected_explore_rank": explore_ranks.get(selected_id),
        "original_explore_rank": explore_ranks.get(original_id),

        "original_distance": orig_dist,
        "selected_distance": sel_dist,
        "distance_delta": sel_dist - orig_dist,
        "distance_saving_vs_original": orig_dist - sel_dist,
        "selected_distance_rank": distance_ranks.get(selected_id),
        "original_distance_rank": distance_ranks.get(original_id),

        "original_explore_efficiency": orig_eff,
        "selected_explore_efficiency": sel_eff,
        "explore_efficiency_ratio_vs_original": sel_eff / orig_eff if abs(orig_eff) > EPS else None,
    }

    gm = r.get("llm_guard_metrics", {}) or {}
    for k in [
        "weighted_ratio",
        "explore_ratio",
        "distance_delta",
        "min_weight_keep",
        "min_explore_keep",
        "max_distance_increase",
    ]:
        row[f"guard_{k}"] = gm.get(k)

    row["llm_reason"] = r.get("llm_reason", "")
    return row


all_rows = []
changed_rows = []

for run_name, path in RUNS.items():
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in records:
        row = enrich_record(run_name, r)
        if row is None:
            continue
        all_rows.append(row)
        if row["changed_final"]:
            changed_rows.append(row)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


write_csv(OUT_ALL, all_rows)
write_csv(OUT_CHANGED, changed_rows)

lines = []
lines.append("Derived Planner Metrics Analysis")
lines.append("=" * 80)
lines.append(f"All Qwen-related records: {len(all_rows)}")
lines.append(f"Changed final records: {len(changed_rows)}")
lines.append("")

for run_name in RUNS:
    rs = [r for r in all_rows if r["run"] == run_name]
    ch = [r for r in changed_rows if r["run"] == run_name]

    lines.append(f"Run: {run_name}")
    lines.append(f"  records: {len(rs)}")
    lines.append(f"  changed_final: {len(ch)}")

    if ch:
        avg_distance_saving = sum(to_float(r["distance_saving_vs_original"]) for r in ch) / len(ch)
        avg_score_drop = sum(to_float(r["score_drop_ratio_vs_original"]) for r in ch) / len(ch)
        avg_eff_ratio = sum(to_float(r["explore_efficiency_ratio_vs_original"]) for r in ch) / len(ch)
        avg_weighted_rank = sum(to_float(r["selected_weighted_rank"]) for r in ch) / len(ch)
        avg_explore_rank = sum(to_float(r["selected_explore_rank"]) for r in ch) / len(ch)

        lines.append(f"  avg_distance_saving_vs_original: {avg_distance_saving:.4f}")
        lines.append(f"  avg_score_drop_ratio_vs_original: {avg_score_drop:.4f}")
        lines.append(f"  avg_explore_efficiency_ratio_vs_original: {avg_eff_ratio:.4f}")
        lines.append(f"  avg_selected_weighted_rank: {avg_weighted_rank:.2f}")
        lines.append(f"  avg_selected_explore_rank: {avg_explore_rank:.2f}")
        lines.append("  changed_steps:")
        for r in ch:
            lines.append(
                "    "
                f"step={r['step']}, "
                f"{r['original_next_visit']}->{r['final_next_visit']}, "
                f"weighted_rank={r['selected_weighted_rank']}, "
                f"explore_rank={r['selected_explore_rank']}, "
                f"distance_rank={r['selected_distance_rank']}, "
                f"distance_saving={to_float(r['distance_saving_vs_original']):.4f}, "
                f"score_drop={to_float(r['score_drop_ratio_vs_original']):.4f}, "
                f"eff_ratio={to_float(r['explore_efficiency_ratio_vs_original']):.4f}"
            )

    lines.append("")

OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("Wrote:", OUT_ALL)
print("Wrote:", OUT_CHANGED)
print("Wrote:", OUT_SUMMARY)
