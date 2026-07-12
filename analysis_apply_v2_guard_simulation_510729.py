import csv
from pathlib import Path

INPUT = Path("analysis_qwen_rank_efficiency_logonly_510729_hyp_accept.csv")
OUT_TXT = Path("analysis_apply_v2_guard_simulation_510729_summary.txt")
OUT_CSV = Path("analysis_apply_v2_guard_simulation_510729_decisions.csv")

POLICIES = [
    {
        "name": "rank_eff_current_dist05_eff10",
        "max_weighted_rank": 3,
        "max_score_drop": 0.09,
        "min_distance_saving": 0.5,
        "min_stable_eff_ratio": 1.0,
    },
    {
        "name": "rank_eff_apply_v2_dist08_eff11",
        "max_weighted_rank": 3,
        "max_score_drop": 0.09,
        "min_distance_saving": 0.8,
        "min_stable_eff_ratio": 1.1,
    },
    {
        "name": "rank_eff_apply_v2_dist10_eff11",
        "max_weighted_rank": 3,
        "max_score_drop": 0.09,
        "min_distance_saving": 1.0,
        "min_stable_eff_ratio": 1.1,
    },
]

def f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return default

def accept(row, p):
    return (
        f(row, "selected_weighted_rank") <= p["max_weighted_rank"]
        and f(row, "score_drop_ratio_vs_original") <= p["max_score_drop"]
        and f(row, "distance_saving_vs_original") >= p["min_distance_saving"]
        and f(row, "stable_efficiency_ratio_vs_original") >= p["min_stable_eff_ratio"]
    )

rows = []
with INPUT.open("r", encoding="utf-8") as fp:
    for row in csv.DictReader(fp):
        rows.append(row)

decision_rows = []
lines = []
lines.append("Apply-v2 Guard Simulation on Qwen Rank-Efficiency Log-Only 510729")
lines.append("=" * 80)
lines.append(f"Input: {INPUT}")
lines.append(f"Candidate records: {len(rows)}")
lines.append("")

for policy in POLICIES:
    accepted = []
    rejected = []

    for row in rows:
        ok = accept(row, policy)
        item = {
            "policy": policy["name"],
            "step": row["step"],
            "original_next_visit": row["original_next_visit"],
            "llm_selected_id": row["llm_selected_id"],
            "selected_weighted_rank": row["selected_weighted_rank"],
            "score_drop_ratio_vs_original": row["score_drop_ratio_vs_original"],
            "distance_saving_vs_original": row["distance_saving_vs_original"],
            "stable_efficiency_ratio_vs_original": row["stable_efficiency_ratio_vs_original"],
            "accepted": ok,
        }
        decision_rows.append(item)
        if ok:
            accepted.append(item)
        else:
            rejected.append(item)

    lines.append(f"Policy: {policy['name']}")
    lines.append(f"  max_weighted_rank: {policy['max_weighted_rank']}")
    lines.append(f"  max_score_drop: {policy['max_score_drop']}")
    lines.append(f"  min_distance_saving: {policy['min_distance_saving']}")
    lines.append(f"  min_stable_eff_ratio: {policy['min_stable_eff_ratio']}")
    lines.append(f"  accepted_count: {len(accepted)}")
    lines.append(f"  rejected_count: {len(rejected)}")
    lines.append(f"  accepted_steps: {[int(x['step']) for x in accepted]}")
    lines.append(f"  rejected_steps: {[int(x['step']) for x in rejected]}")
    lines.append("")

with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(decision_rows[0].keys()))
    writer.writeheader()
    writer.writerows(decision_rows)

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("Wrote:", OUT_CSV)
print("Wrote:", OUT_TXT)
