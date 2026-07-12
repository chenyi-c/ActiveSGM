import csv
import json
from pathlib import Path

INPUT = Path("analysis_qwen_apply_changed_steps.csv")
OUT_TXT = Path("analysis_next_guard_policy_summary.txt")
OUT_JSON = Path("analysis_next_guard_policy_summary.json")

POLICIES = [
    {
        "name": "current_w90_e80_dist_le0",
        "weighted_min": 0.90,
        "explore_min": 0.80,
        "distance_rule": "le0",
    },
    {
        "name": "next_w90_e90_dist_lt0",
        "weighted_min": 0.90,
        "explore_min": 0.90,
        "distance_rule": "lt0",
    },
    {
        "name": "strict_w92_e90_dist_lt0",
        "weighted_min": 0.92,
        "explore_min": 0.90,
        "distance_rule": "lt0",
    },
    {
        "name": "very_strict_w95_e90_dist_lt0",
        "weighted_min": 0.95,
        "explore_min": 0.90,
        "distance_rule": "lt0",
    },
]


def f(x):
    return float(x)


def accept(row, policy):
    weighted_ok = f(row["weighted_ratio"]) >= policy["weighted_min"]
    explore_ok = f(row["explore_ratio"]) >= policy["explore_min"]

    d = f(row["distance_delta"])
    if policy["distance_rule"] == "le0":
        distance_ok = d <= 0.0
    elif policy["distance_rule"] == "lt0":
        distance_ok = d < 0.0
    else:
        raise ValueError(policy["distance_rule"])

    return weighted_ok and explore_ok and distance_ok


rows = list(csv.DictReader(INPUT.open("r", encoding="utf-8")))

summary = {
    "input": str(INPUT),
    "num_changed_steps_current_apply": len(rows),
    "policies": [],
}

lines = []
lines.append("Next Guard Policy Offline Analysis")
lines.append("=" * 80)
lines.append(f"Input changed-step table: {INPUT}")
lines.append(f"Current apply changed steps: {len(rows)}")
lines.append("")

for p in POLICIES:
    accepted = []
    rejected = []

    for r in rows:
        item = {
            "step": int(r["step"]),
            "original_next_visit": int(r["original_next_visit"]),
            "final_next_visit": int(r["final_next_visit"]),
            "weighted_ratio": f(r["weighted_ratio"]),
            "explore_ratio": f(r["explore_ratio"]),
            "distance_delta": f(r["distance_delta"]),
        }

        if accept(r, p):
            accepted.append(item)
        else:
            rejected.append(item)

    rec = {
        "policy": p,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_steps": [x["step"] for x in accepted],
        "rejected_steps": [x["step"] for x in rejected],
        "accepted": accepted,
        "rejected": rejected,
    }
    summary["policies"].append(rec)

    lines.append(f"Policy: {p['name']}")
    lines.append(f"  weighted_min: {p['weighted_min']}")
    lines.append(f"  explore_min: {p['explore_min']}")
    lines.append(f"  distance_rule: {p['distance_rule']}")
    lines.append(f"  accepted_count: {len(accepted)}")
    lines.append(f"  rejected_count: {len(rejected)}")
    lines.append(f"  accepted_steps: {[x['step'] for x in accepted]}")
    lines.append(f"  rejected_steps: {[x['step'] for x in rejected]}")

    if rejected:
        lines.append("  rejected_details:")
        for x in rejected:
            lines.append(
                "    "
                f"step={x['step']}, "
                f"weighted_ratio={x['weighted_ratio']:.4f}, "
                f"explore_ratio={x['explore_ratio']:.4f}, "
                f"distance_delta={x['distance_delta']:.4f}"
            )
    lines.append("")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n".join(lines))
print("Wrote:", OUT_TXT)
print("Wrote:", OUT_JSON)
