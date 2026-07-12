import json
from pathlib import Path


BASE = Path(
    "results/Replica/office0/ActiveSem/"
    "run_fake_llm_full_night_20260526_140049/"
    "splatam/llm_logs"
)

ORIG_LOG = BASE / "planner_semantic_candidates.jsonl"
QWEN_LOG = BASE / "qwen_offline_rerank_topk_results.jsonl"

SUMMARY_TXT = BASE / "qwen_guarded_policy_sweep_summary.txt"
SUMMARY_JSON = BASE / "qwen_guarded_policy_sweep_summary.json"


POLICIES = [
    {
        "name": "strict_w90_e85_top5",
        "weighted_keep": 0.90,
        "explore_keep": 0.85,
        "max_weight_rank": 5,
        "max_explore_rank": 5,
        "max_distance_increase": 0.5,
    },
    {
        "name": "medium_w80_e75_top5",
        "weighted_keep": 0.80,
        "explore_keep": 0.75,
        "max_weight_rank": 5,
        "max_explore_rank": 8,
        "max_distance_increase": 1.0,
    },
    {
        "name": "loose_w70_e60_top8",
        "weighted_keep": 0.70,
        "explore_keep": 0.60,
        "max_weight_rank": 8,
        "max_explore_rank": 10,
        "max_distance_increase": 1.5,
    },
    {
        "name": "distance_rescue_w70_e60_top10",
        "weighted_keep": 0.70,
        "explore_keep": 0.60,
        "max_weight_rank": 10,
        "max_explore_rank": 10,
        "max_distance_increase": 0.0,
    },
]


def load_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def get_candidate(record, cand_id):
    for c in record["candidates"]:
        if int(c["id"]) == int(cand_id):
            return c
    return None


def value(cand, key):
    return float(cand.get(key, 0.0))


def ratio(new, old):
    if abs(old) < 1e-12:
        return 1.0 if new >= old else 0.0
    return new / old


def rank_maps(record):
    candidates = record["candidates"]

    by_weight = sorted(
        candidates,
        key=lambda c: (
            value(c, "weighted_score"),
            value(c, "explore_ig"),
        ),
        reverse=True,
    )

    by_explore = sorted(
        candidates,
        key=lambda c: (
            value(c, "explore_ig"),
            value(c, "weighted_score"),
        ),
        reverse=True,
    )

    weight_rank = {int(c["id"]): i + 1 for i, c in enumerate(by_weight)}
    explore_rank = {int(c["id"]): i + 1 for i, c in enumerate(by_explore)}

    return weight_rank, explore_rank


def evaluate_policy(policy, orig_records, qwen_records):
    results = []

    total = 0
    direct_changed = 0
    accepted_changed = 0
    guarded_same = 0
    fallback_count = 0
    invalid_count = 0

    accepted_weighted_lower = 0
    accepted_explore_lower = 0
    accepted_distance_lower = 0
    accepted_distance_higher = 0

    for qr in qwen_records:
        total += 1

        idx = int(qr["record_index"])
        original_record = orig_records[idx]

        original_id = int(qr["original_next_visit"])
        qwen_id = int(qr["qwen_selected_id"])

        original_cand = get_candidate(original_record, original_id)
        qwen_cand = get_candidate(original_record, qwen_id)

        if original_cand is None or qwen_cand is None:
            final_id = original_id
            accept_qwen = False
            reason = "candidate_not_found_fallback"
            invalid_count += 1
            fallback_count += 1
        else:
            ow = value(original_cand, "weighted_score")
            qw = value(qwen_cand, "weighted_score")
            oe = value(original_cand, "explore_ig")
            qe = value(qwen_cand, "explore_ig")
            od = value(original_cand, "distance")
            qd = value(qwen_cand, "distance")

            w_ratio = ratio(qw, ow)
            e_ratio = ratio(qe, oe)
            d_delta = qd - od

            weight_rank, explore_rank = rank_maps(original_record)
            q_weight_rank = weight_rank.get(qwen_id, 10**9)
            q_explore_rank = explore_rank.get(qwen_id, 10**9)

            direct_is_changed = qwen_id != original_id
            if direct_is_changed:
                direct_changed += 1

            parse_ok = bool(qr.get("parse_ok", False))
            direct_fallback = bool(qr.get("fallback", False))

            accept_qwen = (
                direct_is_changed
                and parse_ok
                and not direct_fallback
                and w_ratio >= policy["weighted_keep"]
                and e_ratio >= policy["explore_keep"]
                and q_weight_rank <= policy["max_weight_rank"]
                and q_explore_rank <= policy["max_explore_rank"]
                and d_delta <= policy["max_distance_increase"]
            )

            if accept_qwen:
                final_id = qwen_id
                accepted_changed += 1
                reason = "accepted_by_guard"

                if qw < ow:
                    accepted_weighted_lower += 1
                if qe < oe:
                    accepted_explore_lower += 1
                if qd < od:
                    accepted_distance_lower += 1
                if qd > od:
                    accepted_distance_higher += 1
            else:
                final_id = original_id
                if direct_is_changed:
                    fallback_count += 1
                reason = "guard_fallback_to_original"

            if final_id == original_id:
                guarded_same += 1

            row = {
                "record_index": idx,
                "step": qr.get("step"),
                "policy": policy["name"],
                "original_next_visit": original_id,
                "direct_qwen_selected_id": qwen_id,
                "guarded_final_id": final_id,
                "direct_changed": direct_is_changed,
                "guard_accept_qwen": accept_qwen,
                "guard_reason": reason,
                "original_weighted_score": ow,
                "qwen_weighted_score": qw,
                "weighted_ratio": w_ratio,
                "original_explore_ig": oe,
                "qwen_explore_ig": qe,
                "explore_ratio": e_ratio,
                "original_distance": od,
                "qwen_distance": qd,
                "distance_delta": d_delta,
                "qwen_weight_rank": q_weight_rank,
                "qwen_explore_rank": q_explore_rank,
                "qwen_reason": qr.get("qwen_reason", ""),
            }

        results.append(row)

    out_path = BASE / f"qwen_guarded_{policy['name']}_results.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "policy": policy["name"],
        "output_log": str(out_path),
        "total_decisions": total,
        "direct_changed": direct_changed,
        "accepted_qwen_changes": accepted_changed,
        "guarded_same_as_original": guarded_same,
        "guard_fallback_count": fallback_count,
        "invalid_count": invalid_count,
        "accepted_weighted_lower": accepted_weighted_lower,
        "accepted_explore_lower": accepted_explore_lower,
        "accepted_distance_lower": accepted_distance_lower,
        "accepted_distance_higher": accepted_distance_higher,
        "accepted_change_ratio": accepted_changed / total if total else 0.0,
    }

    return summary


def main():
    orig_records = load_jsonl(ORIG_LOG)
    qwen_records = load_jsonl(QWEN_LOG)

    print("orig_records:", len(orig_records))
    print("qwen_records:", len(qwen_records))

    summaries = []

    for policy in POLICIES:
        summary = evaluate_policy(policy, orig_records, qwen_records)
        summaries.append(summary)

    SUMMARY_JSON.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    lines.append("Qwen Guarded Rerank Policy Sweep")
    lines.append("=" * 80)

    for s in summaries:
        lines.append("")
        lines.append(f"Policy: {s['policy']}")
        lines.append(f"total_decisions: {s['total_decisions']}")
        lines.append(f"direct_changed: {s['direct_changed']}")
        lines.append(f"accepted_qwen_changes: {s['accepted_qwen_changes']}")
        lines.append(f"guarded_same_as_original: {s['guarded_same_as_original']}")
        lines.append(f"guard_fallback_count: {s['guard_fallback_count']}")
        lines.append(f"invalid_count: {s['invalid_count']}")
        lines.append(f"accepted_weighted_lower: {s['accepted_weighted_lower']}")
        lines.append(f"accepted_explore_lower: {s['accepted_explore_lower']}")
        lines.append(f"accepted_distance_lower: {s['accepted_distance_lower']}")
        lines.append(f"accepted_distance_higher: {s['accepted_distance_higher']}")
        lines.append(f"accepted_change_ratio: {s['accepted_change_ratio']:.4f}")
        lines.append(f"output_log: {s['output_log']}")

    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print()
    print("Wrote:", SUMMARY_TXT)
    print("Wrote:", SUMMARY_JSON)


if __name__ == "__main__":
    main()
