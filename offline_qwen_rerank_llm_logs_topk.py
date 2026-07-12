import json
import re
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct"

INPUT_LOG = Path(
    "results/Replica/office0/ActiveSem/"
    "run_fake_llm_full_night_20260526_140049/"
    "splatam/llm_logs/planner_semantic_candidates.jsonl"
)

OUTPUT_LOG = INPUT_LOG.with_name("qwen_offline_rerank_topk_results.jsonl")
SUMMARY_TXT = INPUT_LOG.with_name("qwen_offline_rerank_topk_summary.txt")

TOP_K = 12


def extract_json_object(text):
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def compact_candidate(c):
    return {
        "id": int(c["id"]),
        "distance": round(float(c.get("distance", 0.0)), 4),
        "explore_ig": round(float(c.get("explore_ig", 0.0)), 4),
        "semantic_entropy": round(float(c.get("semantic_entropy", 0.0)), 4),
        "weighted_score": round(float(c.get("weighted_score", 0.0)), 6),
    }


def select_topk_candidates(record):
    original = int(record["original_next_visit"])
    candidates = list(record["candidates"])

    # Sort by weighted_score first, then explore_ig.
    ranked = sorted(
        candidates,
        key=lambda c: (
            float(c.get("weighted_score", 0.0)),
            float(c.get("explore_ig", 0.0)),
        ),
        reverse=True,
    )

    selected = ranked[:TOP_K]

    # Always include original candidate for fair comparison.
    selected_ids = {int(c["id"]) for c in selected}
    if original not in selected_ids:
        for c in candidates:
            if int(c["id"]) == original:
                selected.append(c)
                break

    # Sort by id for readability.
    selected = sorted(selected, key=lambda c: int(c["id"]))
    return [compact_candidate(c) for c in selected]


def build_prompt(record, candidates):
    return (
        "You are a robot next-best-view planning assistant.\n"
        "Choose exactly one candidate id from the list.\n"
        "Consider weighted_score, exploration information gain, semantic entropy, and distance.\n"
        "Higher weighted_score and explore_ig are usually better. Lower distance is usually preferred if scores are similar.\n"
        "Return only one JSON object. No markdown. No extra text.\n\n"
        f"Step: {record.get('step')}\n"
        f"Planning state: {record.get('planning_state')}\n"
        f"Exploration stage: {record.get('exploration_stage')}\n\n"
        "Candidates:\n"
        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        "Required JSON format:\n"
        '{"selected_candidate_id": 0, "reason": "short reason"}\n'
    )


def generate_json(tok, model, prompt):
    messages = [
        {"role": "system", "content": "You are a strict JSON generator. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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


def main():
    print("Input log:", INPUT_LOG)
    print("Output log:", OUTPUT_LOG)
    print("TOP_K:", TOP_K)

    records = [json.loads(x) for x in INPUT_LOG.read_text(encoding="utf-8").splitlines() if x.strip()]
    print("num_records:", len(records))

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)

    print("Loading model in float32...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to("cuda:0")
    model.eval()

    total = 0
    same = 0
    changed = 0
    invalid = 0
    fallback = 0
    errors = 0

    with OUTPUT_LOG.open("w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            total += 1
            original = int(record["original_next_visit"])

            try:
                candidates = select_topk_candidates(record)
                valid_ids = {int(c["id"]) for c in candidates}

                prompt = build_prompt(record, candidates)
                raw = generate_json(tok, model, prompt)
                obj = extract_json_object(raw)

                qwen_selected = original
                qwen_reason = ""
                parse_ok = False
                used_fallback = False
                error_text = ""

                if obj is not None and isinstance(obj.get("selected_candidate_id"), int):
                    cand_id = int(obj["selected_candidate_id"])
                    if cand_id in valid_ids:
                        qwen_selected = cand_id
                        qwen_reason = str(obj.get("reason", ""))
                        parse_ok = True
                    else:
                        used_fallback = True
                        qwen_reason = f"Invalid candidate id {cand_id}; fallback to original."
                else:
                    used_fallback = True
                    qwen_reason = "No valid JSON object found; fallback to original."

            except Exception as e:
                errors += 1
                invalid += 1
                fallback += 1
                qwen_selected = original
                qwen_reason = "Exception occurred; fallback to original."
                raw = ""
                parse_ok = False
                used_fallback = True
                error_text = repr(e)
                traceback.print_exc()

            if not parse_ok:
                invalid += 1
            if used_fallback:
                fallback += 1

            same_as_original = qwen_selected == original
            if same_as_original:
                same += 1
            else:
                changed += 1

            out_record = {
                "record_index": idx,
                "step": record.get("step"),
                "planning_state": record.get("planning_state"),
                "exploration_stage": record.get("exploration_stage"),
                "num_candidates_original": record.get("num_candidates"),
                "num_candidates_prompt": len(candidates) if "candidates" in locals() else None,
                "original_next_visit": original,
                "qwen_selected_id": qwen_selected,
                "same_as_original": same_as_original,
                "parse_ok": parse_ok,
                "fallback": used_fallback,
                "qwen_reason": qwen_reason,
                "raw_response": raw,
                "error": error_text,
            }

            f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            f.flush()

            print(
                f"[{idx+1}/{len(records)}] "
                f"step={record.get('step')} "
                f"orig={original} qwen={qwen_selected} "
                f"same={same_as_original} parse_ok={parse_ok} fallback={used_fallback}"
            )

            torch.cuda.empty_cache()

    summary = {
        "input_log": str(INPUT_LOG),
        "output_log": str(OUTPUT_LOG),
        "top_k": TOP_K,
        "total_decisions": total,
        "same_as_original": same,
        "changed_decisions": changed,
        "invalid_outputs": invalid,
        "fallback_count": fallback,
        "exception_count": errors,
        "same_ratio": same / total if total else 0.0,
        "changed_ratio": changed / total if total else 0.0,
    }

    SUMMARY_TXT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
