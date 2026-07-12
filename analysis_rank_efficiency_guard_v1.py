import csv
import json
from pathlib import Path

INPUT = Path("analysis_derived_planner_metrics_changed_steps.csv")
OUT_TXT = Path("analysis_rank_efficiency_guard_v1_summary.txt")
OUT_JSON = Path("analysis_rank_efficiency_guard_v1_summary.json")
OUT_CSV = Path("analysis_rank_efficiency_guard_v1_decisions.csv")

POLICIES = [
    {
        "name": "rank_eff_v1_dist1_score09",
        "max_weighted_rank": 3,
        "max_score_drop": 0.09,
        "min_distance_saving": 1.0,
        "min_stable_eff_ratio": 1.0,
    },
    {
        "name": "rank_eff_v1_dist05_score09",
        "max_weighted_rank": 3,
        "max_score_drop": 0.09,
        "min_distance_saving": 0.5,
        "min_stable_eff_ratio": 1.0,
    },
    {
        "name": "rank_eff_v1_dist1_score08",
        "max_weighted_rank": 3,
        "max_score_drop": 0.08,
        "min_distance_saving": 1.0,
        "min_stable_eff_ratio": 1.0,
    },
]

def f(row, key, default=0.0):
    try:
        return float(row[key])
    except Exception:
        return default

def stable_eff(explore_ig, distance):
    return explore_ig / (distance + 1.0)

def enrich(row):
    original_eff = stable_eff(
        f(row, "original_explore_ig"),
        f(row, "original_distance"),
    )
    selected_eff = stable_eff(
        f(row, "selected_explore_ig"),
        f(row, "selected_distance"),
    )
    row["stable_original_efficiency"] = original_eff
    row["stable_selected_efficiency"] = selected_eff
    row["stable_efficiency_ratio"] = selected_eff / original_eff if original_eff > 1e-12 else 0.0
    return row

def accept(row, policy):
    return (
        f(row, "selected_weighted_rank") <= policy["max_weighted_rank"]
        and f(row, "score_drop_ratio_vs_original") <= policy["max_score_drop"]
        and f(row, "distance_saving_vs_original") >= policy["min_distance_saving"]
        and f(row, "stable_efficiency_ratio") >= policy["min_stable_eff_ratio"]
    )

rows = []
with INPUT.open("r", encoding="utf-8") as fp:
    for row in csv.DictReader(fp):
        rows.append(enrich(row))

summary = []
decision_rows = []
lines = []
lines.append("Rank-Efficiency Guard v1 Offline Analysis")
lines.append("=" * 80)
lines.append(f"Input: {INPUT}")
lines.append(f"Changed-step records: {len(rows)}")
lines.append("")

for policy in POLICIES:
    accepted = []
    rejected = []

    for row in rows:
        item = {
            "policy": policy["name"],
            "run": row["run"],
            "step": int(f(row, "step")),
            "original_next_visit": int(f(row, "original_next_visit")),
            "final_next_visit": int(f(row, "final_next_visit")),
            "selected_weighted_rank": int(f(row, "selected_weighted_rank")),
            "selected_explore_rank": int(f(row, "selected_explore_rank")),
            "distance_saving": f(row, "distance_saving_vs_original"),
            "score_drop": f(row, "score_drop_ratio_vs_original"),
            "stable_efficiency_ratio": f(row, "stable_efficiency_ratio"),
            "accepted": accept(row, policy),
        }
        decision_rows.append(item)
        if item["accepted"]:
            accepted.append(item)
        else:
            rejected.append(item)

    rec = {
        "policy": policy,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_steps_by_run": {},
        "rejected_steps_by_run": {},
    }

    for run in sorted(set(x["run"] for x in rows)):
        rec["accepted_steps_by_run"][run] = [x["step"] for x in accepted if x["run"] == run]
        rec["rejected_steps_by_run"][run] = [x["step"] for x in rejected if x["run"] == run]

    summary.append(rec)

    lines.append(f"Policy: {policy['name']}")
    lines.append(f"  max_weighted_rank: {policy['max_weighted_rank']}")
    lines.append(f"  max_score_drop: {policy['max_score_drop']}")
    lines.append(f"  min_distance_saving: {policy['min_distance_saving']}")
    lines.append(f"  min_stable_eff_ratio: {policy['min_stable_eff_ratio']}")
    lines.append(f"  accepted_count: {len(accepted)}")
    lines.append(f"  rejected_count: {len(rejected)}")
    for run in sorted(set(x["run"] for x in rows)):
        lines.append(f"  {run} accepted: {rec['accepted_steps_by_run'][run]}")
        lines.append(f"  {run} rejected: {rec['rejected_steps_by_run'][run]}")
    lines.append("")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(decision_rows[0].keys()))
    writer.writeheader()
    writer.writerows(decision_rows)

print("\n".join(lines))
print("Wrote:", OUT_TXT)
print("Wrote:", OUT_JSON)
print("Wrote:", OUT_CSV)
