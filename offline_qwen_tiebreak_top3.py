import json
import re
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct"

BASE = Path(
    "results/Replica/office0/ActiveSem/"
    "run_fake_llm_full_night_20260526_140049/"
    "splatam/llm_logs"
)

INPUT_LOG = BASE / "planner_semantic_candidates.jsonl"
OUTPUT_LOG = BASE / "qwen_tiebreak_top3_results.jsonl"
SUMMARY_TXT = BASE / "qwen_tiebreak_top3_summary.txt"

TOP_N = 3

# Only call Qwen when top-1 and top-2 are close.
# Example: 0.10 means top2 score is within 10% of top1.
TIE_GAP_THRESHOLD = 0.10

# Guard constraints after Qwen selection.
MIN_WEIGHT_KEEP = 0.90
MIN_EXPLORE_KEEP = 0.80
MAX_DISTANCE_INCREASE = 0.50


def load_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def val(c, key):
    return float(c.get(key, 0.0))


def safe_ratio(new, old):
    if abs(old) < 1e-12:
        return 1.0 if new >= old else 0.0
    return new / old


def get_candidate(record, cand_id):
    for c in record["candidates"]:
        if int(c["id"]) == int(cand_id):
            return c
    return None


def top_candidates(record, n=3):
    candidates = list(record["candidates"])
    ranked = sorted(
        candidates,
        key=lambda c: (
            val(c, "weighted_score"),
            val(c, "explore_ig"),
        ),
        reverse=True,
    )
    return ranked[:n], ranked


def is_tie_case(topn):
    if len(topn) < 2:
        return False, None

    top1 = topn[0]
    top2 = topn[1]

    s1 = val(top1, "weighted_score")
    s2 = val(top2, "weighted_score")

    if abs(s1) < 1e-12:
        return True, 0.0

    gap = (s1 - s2) / abs(s1)
    return gap <= TIE_GAP_THRESHOLD, gap


def compact_candidate(c, weight_ref, explore_ref, rank):
    weighted = val(c, "weighted_score")
    explore = val(c, "explore_ig")

    return {
        "id": int(c["id"]),
        "rank_by_weighted_score": rank,
        "distance": round(val(c, "distance"), 4),
        "explore_ig": round(explore, 4),
        "explore_ig_norm_to_top1": round(safe_ratio(explore, explore_ref), 4),
        "semantic_entropy": round(val(c, "semantic_entropy"), 4),
        "weighted_score": round(weighted, 6),
        "weighted_score_norm_to_top1": round(safe_ratio(weighted, weight_ref), 4),
    }


def build_prompt(record, prompt_candidates, original_id, tie_gap):
    top1_weight = val(prompt_candidates[0], "weighted_score") if prompt_candidates else 0.0
    top1_explore = val(prompt_candidates[0], "explore_ig") if prompt_candidates else 0.0

    compact = [
        compact_candidate(c, top1_weight, top1_explore, i + 1)
        for i, c in enumerate(prompt_candidates)
    ]

    return (
        "You are helping a robot next-best-view planner.\n"
        "The planner is uncertain because the top candidates have similar weighted scores.\n"
        "Choose exactly one candidate id from the Top-3 list.\n"
        "Prefer candidates that preserve high weighted_score and explore_ig.\n"
        "If scores are close, prefer lower distance and useful semantic uncertainty.\n"
        "Do not choose a candidate with much lower weighted_score or explore_ig unless distance is clearly better.\n"
        "Return only one JSON object. No markdown. No extra text.\n\n"
        f"Step: {record.get('step')}\n"
        f"Planning state: {record.get('planning_state')}\n"
        f"Exploration stage: {record.get('exploration_stage')}\n"
        f"Original ActiveSGM selected id: {original_id}\n"
        f"Top1-top2 weighted_score relative gap: {tie_gap}\n\n"
        "Top-3 candidates:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
        "Required JSON format:\n"
        '{"selected_candidate_id": 0, "reason": "short reason"}\n'
    )


def extract_json_object(text):
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def generate_json(tok, model, prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a strict JSON generator for robot exploration planning. Output valid JSON only.",
        },
        {"role": "user", "content": prompt},
    ]

    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tok([text], return_tensors="pt").to("cuda:0")

    eos_ids = []
    if tok.eos_token_id is not None:
        eos_ids.append(tok.eos_token_id)

    im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_ids if eos_ids else None,
            pad_token_id=tok.eos_token_id,
        )

    generated = out[0][inputs.input_ids.shape[-1]:]
    return tok.decode(generated, skip_special_tokens=True)


def guard_accept(original_cand, qwen_cand):
    ow = val(original_cand, "weighted_score")
    qw = val(qwen_cand, "weighted_score")

    oe = val(original_cand, "explore_ig")
    qe = val(qwen_cand, "explore_ig")

    od = val(original_cand, "distance")
    qd = val(qwen_cand, "distance")

    w_ratio = safe_ratio(qw, ow)
    e_ratio = safe_ratio(qe, oe)
    d_delta = qd - od

    accept = (
        w_ratio >= MIN_WEIGHT_KEEP
        and e_ratio >= MIN_EXPLORE_KEEP
        and d_delta <= MAX_DISTANCE_INCREASE
    )

    return accept, {
        "weighted_ratio": w_ratio,
        "explore_ratio": e_ratio,
        "distance_delta": d_delta,
        "original_weighted_score": ow,
        "qwen_weighted_score": qw,
        "original_explore_ig": oe,
        "qwen_explore_ig": qe,
        "original_distance": od,
        "qwen_distance": qd,
    }


def main():
    print("Input log:", INPUT_LOG)
    print("Output log:", OUTPUT_LOG)
    print("TOP_N:", TOP_N)
    print("TIE_GAP_THRESHOLD:", TIE_GAP_THRESHOLD)
    print("MIN_WEIGHT_KEEP:", MIN_WEIGHT_KEEP)
    print("MIN_EXPLORE_KEEP:", MIN_EXPLORE_KEEP)
    print("MAX_DISTANCE_INCREASE:", MAX_DISTANCE_INCREASE)

    records = load_jsonl(INPUT_LOG)
    print("num_records:", len(records))

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
    )

    print("Loading model in float32...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cuda:0")
    model.eval()

    total = 0
    tie_cases = 0
    qwen_called = 0
    qwen_valid = 0
    accepted = 0
    fallback = 0
    skipped_confident = 0
    invalid = 0
    exceptions = 0

    accepted_distance_lower = 0
    accepted_distance_higher = 0
    accepted_weighted_lower = 0
    accepted_explore_lower = 0

    with OUTPUT_LOG.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            total += 1
            original_id = int(record["original_next_visit"])
            original_cand = get_candidate(record, original_id)

            topn, ranked = top_candidates(record, TOP_N)
            tie, tie_gap = is_tie_case(topn)

            final_id = original_id
            qwen_id = None
            qwen_reason = ""
            raw_response = ""
            guard_ok = False
            guard_metrics = {}
            mode = "keep_original"
            error_text = ""

            try:
                if not tie:
                    skipped_confident += 1
                    mode = "skip_qwen_confident_top1"
                elif original_cand is None:
                    fallback += 1
                    invalid += 1
                    mode = "fallback_original_missing"
                else:
                    tie_cases += 1
                    qwen_called += 1

                    prompt = build_prompt(record, topn, original_id, tie_gap)
                    raw_response = generate_json(tok, model, prompt)
                    obj = extract_json_object(raw_response)

                    valid_ids = {int(c["id"]) for c in topn}

                    if obj is None or not isinstance(obj.get("selected_candidate_id"), int):
                        fallback += 1
                        invalid += 1
                        mode = "fallback_invalid_json"
                        qwen_reason = "No valid JSON object found."
                    else:
                        qwen_id = int(obj["selected_candidate_id"])
                        qwen_reason = str(obj.get("reason", ""))
                        qwen_valid += 1

                        if qwen_id not in valid_ids:
                            fallback += 1
                            invalid += 1
                            mode = "fallback_invalid_candidate_id"
                        elif qwen_id == original_id:
                            final_id = original_id
                            mode = "qwen_same_as_original"
                        else:
                            qwen_cand = get_candidate(record, qwen_id)
                            guard_ok, guard_metrics = guard_accept(original_cand, qwen_cand)

                            if guard_ok:
                                final_id = qwen_id
                                accepted += 1
                                mode = "accept_qwen_tiebreak"

                                if guard_metrics["qwen_distance"] < guard_metrics["original_distance"]:
                                    accepted_distance_lower += 1
                                if guard_metrics["qwen_distance"] > guard_metrics["original_distance"]:
                                    accepted_distance_higher += 1
                                if guard_metrics["qwen_weighted_score"] < guard_metrics["original_weighted_score"]:
                                    accepted_weighted_lower += 1
                                if guard_metrics["qwen_explore_ig"] < guard_metrics["original_explore_ig"]:
                                    accepted_explore_lower += 1
                            else:
                                fallback += 1
                                mode = "fallback_guard_reject"

            except Exception as e:
                exceptions += 1
                fallback += 1
                invalid += 1
                final_id = original_id
                mode = "fallback_exception"
                error_text = repr(e)
                traceback.print_exc()

            out = {
                "record_index": idx,
                "step": record.get("step"),
                "planning_state": record.get("planning_state"),
                "exploration_stage": record.get("exploration_stage"),
                "num_candidates": record.get("num_candidates"),
                "original_next_visit": original_id,
                "top_ids": [int(c["id"]) for c in topn],
                "tie_case": tie,
                "tie_gap": tie_gap,
                "qwen_called": mode not in ["skip_qwen_confident_top1", "fallback_original_missing"],
                "qwen_selected_id": qwen_id,
                "guard_ok": guard_ok,
                "final_id": final_id,
                "changed_final": final_id != original_id,
                "mode": mode,
                "guard_metrics": guard_metrics,
                "qwen_reason": qwen_reason,
                "raw_response": raw_response,
                "error": error_text,
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

            print(
                f"[{idx+1}/{len(records)}] "
                f"step={record.get('step')} "
                f"tie={tie} gap={tie_gap} "
                f"orig={original_id} qwen={qwen_id} final={final_id} mode={mode}"
            )

            torch.cuda.empty_cache()

    summary = {
        "input_log": str(INPUT_LOG),
        "output_log": str(OUTPUT_LOG),
        "total_decisions": total,
        "tie_gap_threshold": TIE_GAP_THRESHOLD,
        "tie_cases": tie_cases,
        "skipped_confident": skipped_confident,
        "qwen_called": qwen_called,
        "qwen_valid": qwen_valid,
        "accepted_qwen_changes": accepted,
        "fallback_count": fallback,
        "invalid_count": invalid,
        "exception_count": exceptions,
        "accepted_distance_lower": accepted_distance_lower,
        "accepted_distance_higher": accepted_distance_higher,
        "accepted_weighted_lower": accepted_weighted_lower,
        "accepted_explore_lower": accepted_explore_lower,
        "accepted_change_ratio": accepted / total if total else 0.0,
    }

    SUMMARY_TXT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote:", SUMMARY_TXT)


if __name__ == "__main__":
    main()
