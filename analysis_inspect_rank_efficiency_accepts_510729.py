import json
import csv
from pathlib import Path

RESULT_DIR = Path("results/Replica/office0/ActiveSem/run_qwen_rank_efficiency_logonly_20260605_060010")
LLM_LOG = RESULT_DIR / "splatam/llm_logs/planner_semantic_candidates.jsonl"

OUT_CSV = Path("analysis_inspect_rank_efficiency_accepts_510729_candidates.csv")
OUT_TXT = Path("analysis_inspect_rank_efficiency_accepts_510729_summary.txt")

def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def stable_eff(c):
    return f(c.get("explore_ig")) / (f(c.get("distance")) + 1.0)

def rank_map(candidates, key, reverse=True):
    ordered = sorted(candidates, key=lambda c: f(c.get(key)), reverse=reverse)
    return {int(c["id"]): i + 1 for i, c in enumerate(ordered)}

def get_candidate(candidates, cid):
    for c in candidates:
        if int(c["id"]) == int(cid):
            return c
    return None

records = [json.loads(x) for x in LLM_LOG.read_text(encoding="utf-8").splitlines() if x.strip()]

accepted = []
for r in records:
    gm = r.get("llm_guard_metrics") or {}
    if gm.get("rank_efficiency_guard_accept"):
        accepted.append(r)

rows = []
lines = []
lines.append("Rank-Efficiency Accepted Case Inspection")
lines.append("=" * 80)
lines.append(f"accepted_count: {len(accepted)}")
lines.append("")

for r in accepted:
    step = r.get("step")
    candidates = r.get("candidates", [])
    original_id = int(r.get("original_next_visit"))
    qwen_id = int(r.get("llm_selected_id"))

    original = get_candidate(candidates, original_id)

    weighted_rank = rank_map(candidates, "weighted_score", reverse=True)
    explore_rank = rank_map(candidates, "explore_ig", reverse=True)
    distance_rank = rank_map(candidates, "distance", reverse=False)

    lines.append(f"Step {step}: original={original_id}, qwen={qwen_id}")
    lines.append(f"  reason: {r.get('llm_reason')}")
    lines.append("  Top candidates by weighted_score:")

    top_by_weight = sorted(candidates, key=lambda c: f(c.get("weighted_score")), reverse=True)[:5]
    orig_eff = stable_eff(original) if original else 0.0

    for c in top_by_weight:
        cid = int(c["id"])
        cand_eff = stable_eff(c)

        ow = f(original.get("weighted_score")) if original else 0.0
        cw = f(c.get("weighted_score"))
        score_drop = 1.0 - (cw / ow) if abs(ow) > 1e-12 else 0.0

        od = f(original.get("distance")) if original else 0.0
        cd = f(c.get("distance"))
        distance_saving = od - cd

        eff_ratio = cand_eff / orig_eff if abs(orig_eff) > 1e-12 else 0.0

        marker = ""
        if cid == original_id:
            marker += "[ORIGINAL]"
        if cid == qwen_id:
            marker += "[QWEN]"

        line = (
            f"    id={cid} {marker} "
            f"weighted_rank={weighted_rank.get(cid)} "
            f"explore_rank={explore_rank.get(cid)} "
            f"distance_rank={distance_rank.get(cid)} "
            f"weighted={cw:.8f} "
            f"explore={f(c.get('explore_ig')):.4f} "
            f"distance={cd:.4f} "
            f"score_drop={score_drop:.4f} "
            f"distance_saving={distance_saving:.4f} "
            f"stable_eff_ratio={eff_ratio:.4f}"
        )
        lines.append(line)

        rows.append({
            "step": step,
            "candidate_id": cid,
            "is_original": cid == original_id,
            "is_qwen_selected": cid == qwen_id,
            "weighted_rank": weighted_rank.get(cid),
            "explore_rank": explore_rank.get(cid),
            "distance_rank": distance_rank.get(cid),
            "weighted_score": cw,
            "explore_ig": f(c.get("explore_ig")),
            "distance": cd,
            "score_drop_ratio_vs_original": score_drop,
            "distance_saving_vs_original": distance_saving,
            "stable_efficiency_ratio_vs_original": eff_ratio,
            "llm_reason": r.get("llm_reason"),
        })

    lines.append("")

with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("Wrote:", OUT_CSV)
print("Wrote:", OUT_TXT)
