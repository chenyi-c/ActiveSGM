import json
import os
import re
import traceback


_QWEN_TOKENIZER = None
_QWEN_MODEL = None


def _to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _safe_ratio(new, old):
    old = _to_float(old)
    new = _to_float(new)
    if abs(old) < 1e-12:
        return 1.0 if new >= old else 0.0
    return new / old


def _fake_keep_original(candidate_records, original_next_visit, reason_prefix="fake_llm"):
    return {
        "selected_candidate_id": int(original_next_visit),
        "guarded_final_id": int(original_next_visit),
        "reason": (
            f"{reason_prefix}: keep original ActiveSGM decision. "
            "This stage validates the planner-to-LLM interface only."
        ),
        "fallback": False,
        "mode": "fake_llm",
        "qwen_called": False,
        "tie_case": False,
        "tie_gap": None,
        "guard_accept_qwen": False,
        "guard_metrics": {},
        "raw_response": "",
    }


def _get_qwen_model():
    global _QWEN_TOKENIZER, _QWEN_MODEL

    if _QWEN_TOKENIZER is not None and _QWEN_MODEL is not None:
        return _QWEN_TOKENIZER, _QWEN_MODEL

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = os.environ.get(
        "QWEN_PLANNER_MODEL_PATH",
        "/data/run01/scxj889/models/Qwen2.5-1.5B-Instruct",
    )

    print(f"[QWEN-PLANNER] loading tokenizer from {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    dtype_name = os.environ.get("QWEN_PLANNER_DTYPE", "float32").lower()
    torch_dtype = torch.float32
    if dtype_name in ("float16", "fp16"):
        torch_dtype = torch.float16

    print(f"[QWEN-PLANNER] loading model from {model_path}, dtype={torch_dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to("cuda:0")
    model.eval()

    _QWEN_TOKENIZER = tok
    _QWEN_MODEL = model
    return _QWEN_TOKENIZER, _QWEN_MODEL


def _extract_json_object(text):
    m = re.search(r"\{.*?\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _rank_top_candidates(candidate_records, top_n):
    ranked = sorted(
        candidate_records,
        key=lambda c: (
            _to_float(c.get("weighted_score", 0.0)),
            _to_float(c.get("explore_ig", 0.0)),
        ),
        reverse=True,
    )
    return ranked[:top_n], ranked


def _is_tie_case(topn, threshold):
    if len(topn) < 2:
        return False, None

    s1 = _to_float(topn[0].get("weighted_score", 0.0))
    s2 = _to_float(topn[1].get("weighted_score", 0.0))

    if abs(s1) < 1e-12:
        return True, 0.0

    gap = (s1 - s2) / abs(s1)
    return gap <= threshold, gap


def _compact_candidate(c, weight_ref, explore_ref, rank):
    weighted = _to_float(c.get("weighted_score", 0.0))
    explore = _to_float(c.get("explore_ig", 0.0))

    return {
        "id": int(c["id"]),
        "rank_by_weighted_score": rank,
        "distance": round(_to_float(c.get("distance", 0.0)), 4),
        "explore_ig": round(explore, 4),
        "explore_ig_norm_to_top1": round(_safe_ratio(explore, explore_ref), 4),
        "semantic_entropy": round(_to_float(c.get("semantic_entropy", 0.0)), 4),
        "weighted_score": round(weighted, 6),
        "weighted_score_norm_to_top1": round(_safe_ratio(weighted, weight_ref), 4),
    }


def _build_prompt(candidate_records, original_next_visit, topn, tie_gap):
    top1_weight = _to_float(topn[0].get("weighted_score", 0.0)) if topn else 0.0
    top1_explore = _to_float(topn[0].get("explore_ig", 0.0)) if topn else 0.0

    compact = [
        _compact_candidate(c, top1_weight, top1_explore, i + 1)
        for i, c in enumerate(topn)
    ]

    return (
        "You are helping a robot next-best-view planner.\n"
        "The planner is uncertain because the top candidates have similar weighted scores.\n"
        "Choose exactly one candidate id from the Top-3 list.\n"
        "Prefer candidates that preserve high weighted_score and explore_ig.\n"
        "If scores are close, prefer lower distance and useful semantic uncertainty.\n"
        "Do not choose a candidate with much lower weighted_score or explore_ig unless distance is clearly better.\n"
        "Return only one JSON object. No markdown. No extra text.\n\n"
        f"Original ActiveSGM selected id: {int(original_next_visit)}\n"
        f"Top1-top2 weighted_score relative gap: {tie_gap}\n\n"
        "Top-3 candidates:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
        "Required JSON format:\n"
        '{"selected_candidate_id": 0, "reason": "short reason"}\n'
    )


def _generate_json(tok, model, prompt):
    import torch

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
            max_new_tokens=int(os.environ.get("QWEN_PLANNER_MAX_NEW_TOKENS", "96")),
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=eos_ids if eos_ids else None,
            pad_token_id=tok.eos_token_id,
        )

    generated = out[0][inputs.input_ids.shape[-1]:]
    return tok.decode(generated, skip_special_tokens=True)


def _get_candidate(candidate_records, cand_id):
    for c in candidate_records:
        if int(c["id"]) == int(cand_id):
            return c
    return None


def _guard_accept(original_cand, qwen_cand):
    min_weight_keep = float(os.environ.get("QWEN_PLANNER_MIN_WEIGHT_KEEP", "0.90"))
    min_explore_keep = float(os.environ.get("QWEN_PLANNER_MIN_EXPLORE_KEEP", "0.80"))
    max_distance_increase = float(os.environ.get("QWEN_PLANNER_MAX_DISTANCE_INCREASE", "0.0"))

    ow = _to_float(original_cand.get("weighted_score", 0.0))
    qw = _to_float(qwen_cand.get("weighted_score", 0.0))

    oe = _to_float(original_cand.get("explore_ig", 0.0))
    qe = _to_float(qwen_cand.get("explore_ig", 0.0))

    od = _to_float(original_cand.get("distance", 0.0))
    qd = _to_float(qwen_cand.get("distance", 0.0))

    w_ratio = _safe_ratio(qw, ow)
    e_ratio = _safe_ratio(qe, oe)
    d_delta = qd - od

    accept = (
        w_ratio >= min_weight_keep
        and e_ratio >= min_explore_keep
        and d_delta <= max_distance_increase
    )

    metrics = {
        "weighted_ratio": w_ratio,
        "explore_ratio": e_ratio,
        "distance_delta": d_delta,
        "original_weighted_score": ow,
        "qwen_weighted_score": qw,
        "original_explore_ig": oe,
        "qwen_explore_ig": qe,
        "original_distance": od,
        "qwen_distance": qd,
        "min_weight_keep": min_weight_keep,
        "min_explore_keep": min_explore_keep,
        "max_distance_increase": max_distance_increase,
    }

    return accept, metrics



def _rank_map_by_metric(candidate_records, key, reverse=True):
    ordered = sorted(
        candidate_records,
        key=lambda c: _to_float(c.get(key, 0.0)),
        reverse=reverse,
    )
    return {int(c["id"]): i + 1 for i, c in enumerate(ordered)}


def _stable_explore_efficiency(c):
    explore = _to_float(c.get("explore_ig", 0.0))
    distance = _to_float(c.get("distance", 0.0))
    return explore / (distance + 1.0)


def _enhanced_candidate(c, original_cand, rank_maps):
    cid = int(c["id"])

    weighted = _to_float(c.get("weighted_score", 0.0))
    explore = _to_float(c.get("explore_ig", 0.0))
    distance = _to_float(c.get("distance", 0.0))

    ow = _to_float(original_cand.get("weighted_score", 0.0))
    od = _to_float(original_cand.get("distance", 0.0))
    orig_eff = _stable_explore_efficiency(original_cand)
    cand_eff = _stable_explore_efficiency(c)

    weighted_ratio = _safe_ratio(weighted, ow)
    score_drop = 1.0 - weighted_ratio
    distance_saving = od - distance
    stable_eff_ratio = _safe_ratio(cand_eff, orig_eff)

    return {
        "id": cid,
        "weighted_score": round(weighted, 8),
        "weighted_rank": rank_maps["weighted"].get(cid),
        "weighted_ratio_vs_original": round(weighted_ratio, 4),
        "score_drop_ratio_vs_original": round(score_drop, 4),

        "explore_ig": round(explore, 4),
        "explore_rank": rank_maps["explore"].get(cid),

        "distance": round(distance, 4),
        "distance_rank": rank_maps["distance"].get(cid),
        "distance_saving_vs_original": round(distance_saving, 4),

        "semantic_entropy": round(_to_float(c.get("semantic_entropy", 0.0)), 4),

        "stable_explore_efficiency": round(cand_eff, 4),
        "stable_efficiency_ratio_vs_original": round(stable_eff_ratio, 4),
    }


def _build_rank_efficiency_prompt(candidate_records, original_next_visit, topn, tie_gap):
    original_cand = _get_candidate(candidate_records, original_next_visit)
    if original_cand is None:
        return _build_prompt(candidate_records, original_next_visit, topn, tie_gap)

    rank_maps = {
        "weighted": _rank_map_by_metric(candidate_records, "weighted_score", reverse=True),
        "explore": _rank_map_by_metric(candidate_records, "explore_ig", reverse=True),
        "distance": _rank_map_by_metric(candidate_records, "distance", reverse=False),
    }

    compact = [
        _enhanced_candidate(c, original_cand, rank_maps)
        for c in topn
    ]

    return (
        "You are helping an ActiveSGM next-best-view planner.\n"
        "The goal is to choose a candidate that is useful for exploration while avoiding unsafe trajectory changes.\n"
        "Choose exactly one candidate id from the provided Top candidates.\n\n"
        "Important derived metrics:\n"
        "- weighted_rank: rank by original ActiveSGM weighted_score. Lower rank is better.\n"
        "- explore_rank: rank by exploration information gain. Lower rank is better.\n"
        "- distance_rank: rank by movement distance. Lower rank is closer.\n"
        "- distance_saving_vs_original: positive means this candidate is closer than the original planner choice.\n"
        "- score_drop_ratio_vs_original: lower means less loss from the original planner score.\n"
        "- stable_efficiency_ratio_vs_original: >1 means better exploration gain per damped distance.\n\n"
        "Decision preference:\n"
        "1. Prefer weighted_rank <= 3.\n"
        "2. Avoid large score_drop_ratio_vs_original.\n"
        "3. Prefer clear distance_saving_vs_original.\n"
        "4. Prefer stable_efficiency_ratio_vs_original >= 1.0.\n"
        "5. Do not choose a candidate only because it is close if exploration value is weak.\n\n"
        "Return only one JSON object. No markdown. No extra text.\n\n"
        f"Original ActiveSGM selected id: {int(original_next_visit)}\n"
        f"Top1-top2 weighted_score relative gap: {tie_gap}\n\n"
        "Top candidates with enhanced metrics:\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
        "Required JSON format:\n"
        '{"selected_candidate_id": 0, "reason": "short reason using enhanced metrics"}\n'
    )


def _rank_efficiency_guard(original_cand, qwen_cand, candidate_records):
    max_weighted_rank = int(os.environ.get("QWEN_RANK_EFF_MAX_WEIGHTED_RANK", "3"))
    max_score_drop = float(os.environ.get("QWEN_RANK_EFF_MAX_SCORE_DROP", "0.09"))
    min_distance_saving = float(os.environ.get("QWEN_RANK_EFF_MIN_DISTANCE_SAVING", "0.5"))
    min_stable_eff_ratio = float(os.environ.get("QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO", "1.0"))

    weighted_ranks = _rank_map_by_metric(candidate_records, "weighted_score", reverse=True)
    explore_ranks = _rank_map_by_metric(candidate_records, "explore_ig", reverse=True)
    distance_ranks = _rank_map_by_metric(candidate_records, "distance", reverse=False)

    qid = int(qwen_cand["id"])
    oid = int(original_cand["id"])

    ow = _to_float(original_cand.get("weighted_score", 0.0))
    qw = _to_float(qwen_cand.get("weighted_score", 0.0))

    oe = _to_float(original_cand.get("explore_ig", 0.0))
    qe = _to_float(qwen_cand.get("explore_ig", 0.0))

    od = _to_float(original_cand.get("distance", 0.0))
    qd = _to_float(qwen_cand.get("distance", 0.0))

    weighted_ratio = _safe_ratio(qw, ow)
    explore_ratio = _safe_ratio(qe, oe)
    score_drop = 1.0 - weighted_ratio
    distance_saving = od - qd

    orig_eff = _stable_explore_efficiency(original_cand)
    qwen_eff = _stable_explore_efficiency(qwen_cand)
    stable_eff_ratio = _safe_ratio(qwen_eff, orig_eff)

    selected_weighted_rank = weighted_ranks.get(qid, 999999)
    selected_explore_rank = explore_ranks.get(qid, 999999)
    selected_distance_rank = distance_ranks.get(qid, 999999)

    accept = (
        selected_weighted_rank <= max_weighted_rank
        and score_drop <= max_score_drop
        and distance_saving >= min_distance_saving
        and stable_eff_ratio >= min_stable_eff_ratio
    )

    metrics = {
        "rank_efficiency_guard_accept": bool(accept),

        "selected_weighted_rank": selected_weighted_rank,
        "selected_explore_rank": selected_explore_rank,
        "selected_distance_rank": selected_distance_rank,
        "original_weighted_rank": weighted_ranks.get(oid, 999999),
        "original_explore_rank": explore_ranks.get(oid, 999999),
        "original_distance_rank": distance_ranks.get(oid, 999999),

        "weighted_ratio": weighted_ratio,
        "explore_ratio": explore_ratio,
        "score_drop_ratio_vs_original": score_drop,
        "distance_saving_vs_original": distance_saving,
        "distance_delta": qd - od,

        "original_stable_efficiency": orig_eff,
        "qwen_stable_efficiency": qwen_eff,
        "stable_efficiency_ratio_vs_original": stable_eff_ratio,

        "original_weighted_score": ow,
        "qwen_weighted_score": qw,
        "original_explore_ig": oe,
        "qwen_explore_ig": qe,
        "original_distance": od,
        "qwen_distance": qd,

        "max_weighted_rank": max_weighted_rank,
        "max_score_drop": max_score_drop,
        "min_distance_saving": min_distance_saving,
        "min_stable_eff_ratio": min_stable_eff_ratio,
    }

    return accept, metrics


def _level_from_score_drop(x):
    x = _to_float(x)
    if x <= 0.05:
        return "low"
    if x <= 0.09:
        return "medium"
    return "high"


def _level_from_distance_saving(x):
    x = _to_float(x)
    if x >= 1.0:
        return "high"
    if x >= 0.5:
        return "medium"
    return "low"


def _level_from_efficiency_gain(x):
    x = _to_float(x)
    if x >= 1.5:
        return "high"
    if x >= 1.1:
        return "medium"
    return "low"


def _stage_from_step_like(candidate_records):
    """
    Best-effort stage inference.

    The reranker interface currently does not receive the global step directly.
    Some future logs may include step inside candidate records; if not, keep
    unknown. The global-local prompt still benefits from candidate distribution
    and local risk summaries.
    """
    for c in candidate_records:
        for key in ("step", "frame_idx", "global_step"):
            if key in c:
                try:
                    step = int(c[key])
                    if step < 200:
                        return "early"
                    if step < 500:
                        return "middle"
                    return "late"
                except Exception:
                    pass
    return "unknown"


def _build_global_state_summary(candidate_records, original_next_visit, topn, tie_gap):
    distances = [_to_float(c.get("distance", 0.0)) for c in candidate_records]
    weights = [_to_float(c.get("weighted_score", 0.0)) for c in candidate_records]
    explores = [_to_float(c.get("explore_ig", 0.0)) for c in candidate_records]

    top1_weight = _to_float(topn[0].get("weighted_score", 0.0)) if len(topn) >= 1 else 0.0
    top2_weight = _to_float(topn[1].get("weighted_score", 0.0)) if len(topn) >= 2 else 0.0
    top3_weight = _to_float(topn[2].get("weighted_score", 0.0)) if len(topn) >= 3 else 0.0

    if distances:
        min_distance = min(distances)
        max_distance = max(distances)
        avg_distance = sum(distances) / len(distances)
    else:
        min_distance = max_distance = avg_distance = 0.0

    if weights:
        max_weight = max(weights)
        avg_weight = sum(weights) / len(weights)
    else:
        max_weight = avg_weight = 0.0

    if explores:
        max_explore = max(explores)
        avg_explore = sum(explores) / len(explores)
    else:
        max_explore = avg_explore = 0.0

    return {
        "exploration_stage": _stage_from_step_like(candidate_records),
        "candidate_count": len(candidate_records),
        "top_candidate_count": len(topn),
        "original_next_visit": int(original_next_visit),
        "top1_top2_weighted_score_gap": tie_gap,
        "top1_weighted_score": round(top1_weight, 8),
        "top2_weighted_score": round(top2_weight, 8),
        "top3_weighted_score": round(top3_weight, 8),
        "max_weighted_score": round(max_weight, 8),
        "avg_weighted_score": round(avg_weight, 8),
        "max_explore_ig": round(max_explore, 4),
        "avg_explore_ig": round(avg_explore, 4),
        "min_distance": round(min_distance, 4),
        "avg_distance": round(avg_distance, 4),
        "max_distance": round(max_distance, 4),
        "planner_uncertainty": "tie_case" if tie_gap is not None else "unknown",
    }


def _candidate_risk_summary(enhanced):
    score_drop = _to_float(enhanced.get("score_drop_ratio_vs_original", 0.0))
    distance_saving = _to_float(enhanced.get("distance_saving_vs_original", 0.0))
    eff_ratio = _to_float(enhanced.get("stable_efficiency_ratio_vs_original", 0.0))
    distance = _to_float(enhanced.get("distance", 0.0))

    return {
        "id": int(enhanced["id"]),
        "score_drop_level": _level_from_score_drop(score_drop),
        "distance_saving_level": _level_from_distance_saving(distance_saving),
        "efficiency_gain_level": _level_from_efficiency_gain(eff_ratio),
        "is_zero_distance_candidate": abs(distance) < 1e-6,
        "is_score_drop_high": score_drop > 0.09,
        "is_distance_saving_negative": distance_saving < 0.0,
        "is_efficiency_gain_weak": eff_ratio < 1.0,
    }


def _build_global_local_prompt(candidate_records, original_next_visit, topn, tie_gap):
    original_cand = _get_candidate(candidate_records, original_next_visit)
    if original_cand is None:
        return _build_rank_efficiency_prompt(candidate_records, original_next_visit, topn, tie_gap)

    rank_maps = {
        "weighted": _rank_map_by_metric(candidate_records, "weighted_score", reverse=True),
        "explore": _rank_map_by_metric(candidate_records, "explore_ig", reverse=True),
        "distance": _rank_map_by_metric(candidate_records, "distance", reverse=False),
    }

    enhanced_topn = [
        _enhanced_candidate(c, original_cand, rank_maps)
        for c in topn
    ]

    global_state = _build_global_state_summary(
        candidate_records=candidate_records,
        original_next_visit=original_next_visit,
        topn=topn,
        tie_gap=tie_gap,
    )

    local_risk = [_candidate_risk_summary(c) for c in enhanced_topn]

    payload = {
        "global_state": global_state,
        "original_active_sgm_choice": int(original_next_visit),
        "top_candidates_with_local_metrics": enhanced_topn,
        "local_risk_summary": local_risk,
        "decision_rules": [
            "The original ActiveSGM choice is the safe fallback.",
            "A candidate should not be selected only because it is closer.",
            "Prefer candidates with low score_drop, positive distance_saving, and stable_efficiency_ratio >= 1.0.",
            "If a candidate has high score_drop, negative distance_saving, or weak efficiency gain, mark trajectory_risk as high.",
            "For uncertain cases, prefer should_change_original=false.",
        ],
    }

    return (
        "You are a trajectory-aware judge for an ActiveSGM next-best-view planner.\n"
        "Your task is not only to pick a local candidate, but to judge whether changing the original planner choice is globally safe.\n"
        "Use both global_state and local candidate metrics.\n\n"
        "Definitions:\n"
        "- global_local_alignment: whether the local candidate is consistent with the current global exploration situation.\n"
        "- trajectory_risk: risk that changing the original planner choice may harm downstream trajectory, mapping, or localization.\n"
        "- decision_confidence: confidence in your own decision.\n"
        "- should_change_original: true only when the change is clearly beneficial and low risk.\n\n"
        "Return only one JSON object. No markdown. No extra text.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Required JSON format:\n"
        "{"
        "\"selected_candidate_id\": 0, "
        "\"should_change_original\": false, "
        "\"global_local_alignment\": \"low|medium|high\", "
        "\"trajectory_risk\": \"low|medium|high\", "
        "\"decision_confidence\": \"low|medium|high\", "
        "\"reason\": \"short reason\""
        "}\n"
    )

def qwen_tiebreak_top3_distance_strict_rerank(candidate_records, original_next_visit):
    """
    Online log-only Qwen reranker.

    It only recommends a guarded candidate. The planner can decide whether to
    apply it. Current ActiveSGM integration should keep next_visit as original
    for online log-only validation.
    """
    top_n = int(os.environ.get("QWEN_PLANNER_TOP_N", "3"))
    tie_gap_threshold = float(os.environ.get("QWEN_PLANNER_TIE_GAP", "0.10"))

    original_next_visit = int(original_next_visit)

    topn, _ = _rank_top_candidates(candidate_records, top_n)
    tie_case, tie_gap = _is_tie_case(topn, tie_gap_threshold)

    result = {
        "selected_candidate_id": original_next_visit,
        "guarded_final_id": original_next_visit,
        "reason": "keep original: not a tie case or Qwen not accepted.",
        "fallback": False,
        "mode": "qwen_tiebreak_top3_distance_strict",
        "qwen_called": False,
        "tie_case": bool(tie_case),
        "tie_gap": tie_gap,
        "guard_accept_qwen": False,
        "guard_metrics": {},
        "raw_response": "",
        "top_ids": [int(c["id"]) for c in topn],
    }

    if not tie_case:
        result["reason"] = "skip_qwen_confident_top1"
        return result

    original_cand = _get_candidate(candidate_records, original_next_visit)
    if original_cand is None:
        result["fallback"] = True
        result["reason"] = "fallback_original_candidate_not_found"
        return result

    try:
        tok, model = _get_qwen_model()
        prompt = _build_prompt(candidate_records, original_next_visit, topn, tie_gap)
        raw = _generate_json(tok, model, prompt)

        result["qwen_called"] = True
        result["raw_response"] = raw

        obj = _extract_json_object(raw)
        valid_ids = {int(c["id"]) for c in topn}

        if obj is None or not isinstance(obj.get("selected_candidate_id"), int):
            result["fallback"] = True
            result["reason"] = "fallback_invalid_json"
            return result

        qwen_id = int(obj["selected_candidate_id"])
        qwen_reason = str(obj.get("reason", ""))

        result["selected_candidate_id"] = qwen_id
        result["reason"] = qwen_reason

        if qwen_id not in valid_ids:
            result["fallback"] = True
            result["reason"] = f"fallback_invalid_candidate_id: {qwen_id}"
            result["guarded_final_id"] = original_next_visit
            return result

        if qwen_id == original_next_visit:
            result["guarded_final_id"] = original_next_visit
            result["guard_accept_qwen"] = False
            result["reason"] = qwen_reason or "qwen_same_as_original"
            return result

        qwen_cand = _get_candidate(candidate_records, qwen_id)
        if qwen_cand is None:
            result["fallback"] = True
            result["reason"] = f"fallback_qwen_candidate_not_found: {qwen_id}"
            result["guarded_final_id"] = original_next_visit
            return result

        guard_ok, guard_metrics = _guard_accept(original_cand, qwen_cand)
        result["guard_metrics"] = guard_metrics
        result["guard_accept_qwen"] = bool(guard_ok)

        if guard_ok:
            result["guarded_final_id"] = qwen_id
            result["fallback"] = False
            result["reason"] = qwen_reason or "accept_qwen_tiebreak"
        else:
            result["guarded_final_id"] = original_next_visit
            result["fallback"] = True
            result["reason"] = qwen_reason or "fallback_guard_reject"

        return result

    except Exception as e:
        result["fallback"] = True
        result["reason"] = f"qwen_tiebreak_exception: {repr(e)}"
        result["guarded_final_id"] = original_next_visit
        result["selected_candidate_id"] = original_next_visit
        traceback.print_exc()

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        return result



def qwen_rank_efficiency_logonly_rerank(candidate_records, original_next_visit):
    """
    Enhanced-metrics Qwen reranker.

    This mode is intentionally log-only:
    - Qwen sees derived planner metrics.
    - The function records Qwen's selected candidate and rank-efficiency guard metrics.
    - guarded_final_id remains original_next_visit.
    - guard_accept_qwen remains False to prevent accidental online apply.
    """
    top_n = int(os.environ.get("QWEN_PLANNER_TOP_N", "3"))
    tie_gap_threshold = float(os.environ.get("QWEN_PLANNER_TIE_GAP", "0.10"))

    original_next_visit = int(original_next_visit)

    topn, _ = _rank_top_candidates(candidate_records, top_n)
    tie_case, tie_gap = _is_tie_case(topn, tie_gap_threshold)

    result = {
        "selected_candidate_id": original_next_visit,
        "guarded_final_id": original_next_visit,
        "reason": "rank_efficiency_logonly: keep original trajectory.",
        "fallback": False,
        "mode": "qwen_rank_efficiency_logonly",
        "qwen_called": False,
        "tie_case": bool(tie_case),
        "tie_gap": tie_gap,
        "guard_accept_qwen": False,
        "guard_metrics": {},
        "raw_response": "",
        "top_ids": [int(c["id"]) for c in topn],
    }

    if not tie_case:
        result["reason"] = "rank_efficiency_logonly_skip_qwen_confident_top1"
        return result

    original_cand = _get_candidate(candidate_records, original_next_visit)
    if original_cand is None:
        result["fallback"] = True
        result["reason"] = "rank_efficiency_logonly_fallback_original_candidate_not_found"
        return result

    try:
        tok, model = _get_qwen_model()
        prompt = _build_rank_efficiency_prompt(candidate_records, original_next_visit, topn, tie_gap)
        raw = _generate_json(tok, model, prompt)

        result["qwen_called"] = True
        result["raw_response"] = raw

        obj = _extract_json_object(raw)
        valid_ids = {int(c["id"]) for c in topn}

        if obj is None or not isinstance(obj.get("selected_candidate_id"), int):
            result["fallback"] = True
            result["reason"] = "rank_efficiency_logonly_fallback_invalid_json"
            return result

        qwen_id = int(obj["selected_candidate_id"])
        qwen_reason = str(obj.get("reason", ""))

        result["selected_candidate_id"] = qwen_id
        result["reason"] = qwen_reason or "rank_efficiency_logonly_qwen_selected"

        if qwen_id not in valid_ids:
            result["fallback"] = True
            result["reason"] = f"rank_efficiency_logonly_fallback_invalid_candidate_id: {qwen_id}"
            result["guarded_final_id"] = original_next_visit
            result["guard_accept_qwen"] = False
            return result

        qwen_cand = _get_candidate(candidate_records, qwen_id)
        if qwen_cand is None:
            result["fallback"] = True
            result["reason"] = f"rank_efficiency_logonly_fallback_qwen_candidate_not_found: {qwen_id}"
            result["guarded_final_id"] = original_next_visit
            result["guard_accept_qwen"] = False
            return result

        rank_guard_ok, rank_guard_metrics = _rank_efficiency_guard(
            original_cand=original_cand,
            qwen_cand=qwen_cand,
            candidate_records=candidate_records,
        )

        # Important: this mode is log-only.
        # Store hypothetical guard result inside guard_metrics, but do not allow planner apply.
        result["guard_metrics"] = rank_guard_metrics
        result["guarded_final_id"] = original_next_visit
        result["guard_accept_qwen"] = False
        result["fallback"] = False

        if qwen_id == original_next_visit:
            result["reason"] = qwen_reason or "rank_efficiency_logonly_qwen_same_as_original"
        elif rank_guard_ok:
            result["reason"] = qwen_reason or "rank_efficiency_logonly_guard_would_accept"
        else:
            result["reason"] = qwen_reason or "rank_efficiency_logonly_guard_would_reject"

        return result

    except Exception as e:
        result["fallback"] = True
        result["reason"] = f"qwen_rank_efficiency_logonly_exception: {repr(e)}"
        result["guarded_final_id"] = original_next_visit
        result["selected_candidate_id"] = original_next_visit
        result["guard_accept_qwen"] = False
        traceback.print_exc()

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        return result





def _build_global_local_v2_prompt(candidate_records, original_next_visit, topn, tie_gap):
    """
    Stricter global-local prompt.

    v1 exposed a consistency problem:
    Qwen could mark trajectory_risk=high but still set should_change_original=true.
    v2 makes these consistency constraints explicit.
    """
    original_cand = _get_candidate(candidate_records, original_next_visit)
    if original_cand is None:
        return _build_global_local_prompt(candidate_records, original_next_visit, topn, tie_gap)

    rank_maps = {
        "weighted": _rank_map_by_metric(candidate_records, "weighted_score", reverse=True),
        "explore": _rank_map_by_metric(candidate_records, "explore_ig", reverse=True),
        "distance": _rank_map_by_metric(candidate_records, "distance", reverse=False),
    }

    enhanced_topn = [
        _enhanced_candidate(c, original_cand, rank_maps)
        for c in topn
    ]

    global_state = _build_global_state_summary(
        candidate_records=candidate_records,
        original_next_visit=original_next_visit,
        topn=topn,
        tie_gap=tie_gap,
    )

    local_risk = [_candidate_risk_summary(c) for c in enhanced_topn]

    max_weighted_rank = int(os.environ.get("QWEN_RANK_EFF_MAX_WEIGHTED_RANK", "3"))
    max_score_drop = float(os.environ.get("QWEN_RANK_EFF_MAX_SCORE_DROP", "0.09"))
    min_distance_saving = float(os.environ.get("QWEN_RANK_EFF_MIN_DISTANCE_SAVING", "0.8"))
    min_stable_eff_ratio = float(os.environ.get("QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO", "1.1"))

    payload = {
        "global_state": global_state,
        "original_active_sgm_choice": int(original_next_visit),
        "top_candidates_with_local_metrics": enhanced_topn,
        "local_risk_summary": local_risk,
        "deterministic_acceptance_thresholds": {
            "selected_weighted_rank_max": max_weighted_rank,
            "score_drop_ratio_vs_original_max": max_score_drop,
            "distance_saving_vs_original_min": min_distance_saving,
            "stable_efficiency_ratio_vs_original_min": min_stable_eff_ratio,
        },
        "strict_consistency_rules": [
            "The original ActiveSGM choice is the default safe fallback.",
            "If no candidate clearly passes all deterministic thresholds, select the original ActiveSGM choice.",
            "If trajectory_risk is high, should_change_original must be false.",
            "If selected_candidate_id is the original ActiveSGM choice, should_change_original must be false.",
            "If should_change_original is true, selected_candidate_id must be different from the original ActiveSGM choice.",
            "If should_change_original is true, the selected candidate must have trajectory_risk low, global_local_alignment high, and decision_confidence high or medium.",
            "If the selected candidate has score_drop above threshold, negative or weak distance_saving, or stable_efficiency_ratio below threshold, should_change_original must be false.",
            "Do not describe a metric as good if its numeric value violates the thresholds.",
        ],
    }

    return (
        "You are a strict trajectory-aware verifier for an ActiveSGM next-best-view planner.\n"
        "Your job is to decide whether it is safe to change the original ActiveSGM planner choice.\n"
        "Use the numeric metrics and the strict consistency rules. Do not rely on intuition alone.\n\n"
        "Critical rule:\n"
        "- A risky candidate can still be mentioned in the reason, but it must not set should_change_original=true.\n"
        "- When uncertain, select the original_active_sgm_choice and set should_change_original=false.\n\n"
        "Return only one JSON object. No markdown. No extra text.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Required JSON format:\n"
        "{"
        "\"selected_candidate_id\": 0, "
        "\"should_change_original\": false, "
        "\"global_local_alignment\": \"low|medium|high\", "
        "\"trajectory_risk\": \"low|medium|high\", "
        "\"decision_confidence\": \"low|medium|high\", "
        "\"reason\": \"short reason based on numeric thresholds\""
        "}\n"
    )


def _build_global_local_v3_prompt(candidate_records, original_next_visit, topn, tie_gap):
    """
    Soft-consistency global-local prompt.

    v2 is safe but overly conservative. v3 remains log-only, but asks Qwen to
    identify low-risk, high-efficiency alternatives under softer consistency rules.
    """
    original_cand = _get_candidate(candidate_records, original_next_visit)
    if original_cand is None:
        return _build_global_local_v2_prompt(candidate_records, original_next_visit, topn, tie_gap)

    rank_maps = {
        "weighted": _rank_map_by_metric(candidate_records, "weighted_score", reverse=True),
        "explore": _rank_map_by_metric(candidate_records, "explore_ig", reverse=True),
        "distance": _rank_map_by_metric(candidate_records, "distance", reverse=False),
    }

    enhanced_topn = [
        _enhanced_candidate(c, original_cand, rank_maps)
        for c in topn
    ]

    global_state = _build_global_state_summary(
        candidate_records=candidate_records,
        original_next_visit=original_next_visit,
        topn=topn,
        tie_gap=tie_gap,
    )

    local_risk = [_candidate_risk_summary(c) for c in enhanced_topn]

    max_weighted_rank = int(os.environ.get("QWEN_V3_SOFT_MAX_WEIGHTED_RANK", "3"))
    max_score_drop = float(os.environ.get("QWEN_V3_SOFT_MAX_SCORE_DROP", "0.09"))
    min_distance_saving = float(os.environ.get("QWEN_V3_SOFT_MIN_DISTANCE_SAVING", "0.5"))
    min_stable_eff_ratio = float(os.environ.get("QWEN_V3_SOFT_MIN_STABLE_EFF_RATIO", "1.0"))

    payload = {
        "global_state": global_state,
        "original_active_sgm_choice": int(original_next_visit),
        "top_candidates_with_local_metrics": enhanced_topn,
        "local_risk_summary": local_risk,
        "soft_acceptance_thresholds": {
            "selected_weighted_rank_max": max_weighted_rank,
            "score_drop_ratio_vs_original_max": max_score_drop,
            "distance_saving_vs_original_min": min_distance_saving,
            "stable_efficiency_ratio_vs_original_min": min_stable_eff_ratio,
            "allowed_global_local_alignment": ["medium", "high"],
            "allowed_trajectory_risk": ["low"],
            "allowed_decision_confidence": ["medium", "high"],
        },
        "soft_consistency_rules": [
            "This is diagnostic log-only analysis. You are not actually changing the trajectory.",
            "The original ActiveSGM choice is still the safe fallback.",
            "You may select a different candidate only if it is low trajectory risk.",
            "If the selected candidate has medium or high global_local_alignment and medium or high decision_confidence, it can be a diagnostic alternative.",
            "If trajectory_risk is high or medium, should_change_original must be false.",
            "If selected_candidate_id is the original ActiveSGM choice, should_change_original must be false.",
            "If should_change_original is true, selected_candidate_id must be different from the original ActiveSGM choice.",
            "If should_change_original is true, explain the numeric tradeoff using score_drop, distance_saving, and stable_efficiency.",
            "Do not describe a metric as good if its numeric value violates the thresholds.",
        ],
    }

    return (
        "You are a trajectory-aware diagnostic assistant for an ActiveSGM next-best-view planner.\n"
        "Your task is not to control the robot. Your task is to identify whether a low-risk alternative viewpoint is worth logging for later analysis.\n"
        "Use the numeric metrics and soft consistency rules. Do not rely on intuition alone.\n\n"
        "Important:\n"
        "- This is log-only. A different selection is only a hypothetical diagnostic signal.\n"
        "- Be less conservative than a strict safety verifier, but do not accept medium/high trajectory risk.\n"
        "- If no alternative satisfies the soft thresholds, select the original_active_sgm_choice and set should_change_original=false.\n\n"
        "Return only one JSON object. No markdown. No extra text.\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Required JSON format:\n"
        "{"
        "\"selected_candidate_id\": 0, "
        "\"should_change_original\": false, "
        "\"global_local_alignment\": \"low|medium|high\", "
        "\"trajectory_risk\": \"low|medium|high\", "
        "\"decision_confidence\": \"low|medium|high\", "
        "\"reason\": \"short reason based on numeric thresholds\""
        "}\n"
    )

def qwen_global_local_logonly_rerank(candidate_records, original_next_visit):
    """
    Global-local Qwen reranker.

    This mode is intentionally log-only:
    - Qwen sees global exploration summary, local candidate metrics, and risk hints.
    - Qwen returns selected_candidate_id, should_change_original, alignment, risk, and confidence.
    - guarded_final_id remains original_next_visit.
    - guard_accept_qwen remains False.
    """
    top_n = int(os.environ.get("QWEN_PLANNER_TOP_N", "3"))
    tie_gap_threshold = float(os.environ.get("QWEN_PLANNER_TIE_GAP", "0.10"))

    original_next_visit = int(original_next_visit)

    topn, _ = _rank_top_candidates(candidate_records, top_n)
    tie_case, tie_gap = _is_tie_case(topn, tie_gap_threshold)

    global_state = _build_global_state_summary(
        candidate_records=candidate_records,
        original_next_visit=original_next_visit,
        topn=topn,
        tie_gap=tie_gap,
    )

    result = {
        "selected_candidate_id": original_next_visit,
        "guarded_final_id": original_next_visit,
        "reason": "global_local_logonly: keep original trajectory.",
        "fallback": False,
        "mode": "qwen_global_local_logonly",
        "qwen_called": False,
        "tie_case": bool(tie_case),
        "tie_gap": tie_gap,
        "guard_accept_qwen": False,
        "guard_metrics": {
            "global_state": global_state,
            "global_local_alignment": "unknown",
            "trajectory_risk": "unknown",
            "decision_confidence": "unknown",
            "should_change_original": False,
        },
        "raw_response": "",
        "top_ids": [int(c["id"]) for c in topn],
    }

    if not tie_case:
        result["reason"] = "global_local_logonly_skip_qwen_confident_top1"
        return result

    original_cand = _get_candidate(candidate_records, original_next_visit)
    if original_cand is None:
        result["fallback"] = True
        result["reason"] = "global_local_logonly_fallback_original_candidate_not_found"
        return result

    try:
        tok, model = _get_qwen_model()
        prompt = _build_global_local_prompt(candidate_records, original_next_visit, topn, tie_gap)
        raw = _generate_json(tok, model, prompt)

        result["qwen_called"] = True
        result["raw_response"] = raw

        obj = _extract_json_object(raw)
        valid_ids = {int(c["id"]) for c in topn}

        if obj is None or not isinstance(obj.get("selected_candidate_id"), int):
            result["fallback"] = True
            result["reason"] = "global_local_logonly_fallback_invalid_json"
            return result

        qwen_id = int(obj["selected_candidate_id"])
        qwen_reason = str(obj.get("reason", ""))

        result["selected_candidate_id"] = qwen_id
        result["reason"] = qwen_reason or "global_local_logonly_qwen_selected"

        gm = result.get("guard_metrics") or {}
        gm["global_local_alignment"] = str(obj.get("global_local_alignment", "unknown")).lower()
        gm["trajectory_risk"] = str(obj.get("trajectory_risk", "unknown")).lower()
        gm["decision_confidence"] = str(obj.get("decision_confidence", "unknown")).lower()
        gm["should_change_original"] = bool(obj.get("should_change_original", False))

        if qwen_id not in valid_ids:
            result["fallback"] = True
            result["reason"] = f"global_local_logonly_fallback_invalid_candidate_id: {qwen_id}"
            result["guard_metrics"] = gm
            return result

        qwen_cand = _get_candidate(candidate_records, qwen_id)
        if qwen_cand is None:
            result["fallback"] = True
            result["reason"] = f"global_local_logonly_fallback_qwen_candidate_not_found: {qwen_id}"
            result["guard_metrics"] = gm
            return result

        rank_guard_ok, rank_guard_metrics = _rank_efficiency_guard(
            original_cand=original_cand,
            qwen_cand=qwen_cand,
            candidate_records=candidate_records,
        )

        # Store both deterministic metrics and global-local LLM judgments.
        rank_guard_metrics["global_state"] = global_state
        rank_guard_metrics["global_local_alignment"] = gm["global_local_alignment"]
        rank_guard_metrics["trajectory_risk"] = gm["trajectory_risk"]
        rank_guard_metrics["decision_confidence"] = gm["decision_confidence"]
        rank_guard_metrics["should_change_original"] = gm["should_change_original"]
        rank_guard_metrics["global_local_would_accept"] = bool(
            rank_guard_ok
            and gm["should_change_original"]
            and gm["trajectory_risk"] != "high"
            and gm["global_local_alignment"] in ("medium", "high")
        )

        result["guard_metrics"] = rank_guard_metrics
        result["guarded_final_id"] = original_next_visit
        result["guard_accept_qwen"] = False
        result["fallback"] = False

        return result

    except Exception as e:
        result["fallback"] = True
        result["reason"] = f"qwen_global_local_logonly_exception: {repr(e)}"
        result["guarded_final_id"] = original_next_visit
        result["selected_candidate_id"] = original_next_visit
        result["guard_accept_qwen"] = False
        traceback.print_exc()

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        return result



def qwen_global_local_v2_logonly_rerank(candidate_records, original_next_visit):
    """
    Global-local v2 Qwen reranker.

    This mode is still log-only, but adds stricter consistency checks:
    - risk=high cannot produce a would-accept decision.
    - should_change_original is post-filtered by deterministic rank-efficiency guard.
    - inconsistent Qwen outputs are recorded but not trusted.
    """
    top_n = int(os.environ.get("QWEN_PLANNER_TOP_N", "3"))
    tie_gap_threshold = float(os.environ.get("QWEN_PLANNER_TIE_GAP", "0.10"))

    original_next_visit = int(original_next_visit)

    topn, _ = _rank_top_candidates(candidate_records, top_n)
    tie_case, tie_gap = _is_tie_case(topn, tie_gap_threshold)

    global_state = _build_global_state_summary(
        candidate_records=candidate_records,
        original_next_visit=original_next_visit,
        topn=topn,
        tie_gap=tie_gap,
    )

    result = {
        "selected_candidate_id": original_next_visit,
        "guarded_final_id": original_next_visit,
        "reason": "global_local_v2_logonly: keep original trajectory.",
        "fallback": False,
        "mode": "qwen_global_local_v2_logonly",
        "qwen_called": False,
        "tie_case": bool(tie_case),
        "tie_gap": tie_gap,
        "guard_accept_qwen": False,
        "guard_metrics": {
            "global_state": global_state,
            "global_local_alignment": "unknown",
            "trajectory_risk": "unknown",
            "decision_confidence": "unknown",
            "should_change_original": False,
            "qwen_should_change_original_raw": False,
            "global_local_consistency_pass": False,
            "global_local_v2_would_accept": False,
        },
        "raw_response": "",
        "top_ids": [int(c["id"]) for c in topn],
    }

    if not tie_case:
        result["reason"] = "global_local_v2_logonly_skip_qwen_confident_top1"
        return result

    original_cand = _get_candidate(candidate_records, original_next_visit)
    if original_cand is None:
        result["fallback"] = True
        result["reason"] = "global_local_v2_logonly_fallback_original_candidate_not_found"
        return result

    try:
        tok, model = _get_qwen_model()
        prompt = _build_global_local_v2_prompt(candidate_records, original_next_visit, topn, tie_gap)
        raw = _generate_json(tok, model, prompt)

        result["qwen_called"] = True
        result["raw_response"] = raw

        obj = _extract_json_object(raw)
        valid_ids = {int(c["id"]) for c in topn}

        if obj is None or not isinstance(obj.get("selected_candidate_id"), int):
            result["fallback"] = True
            result["reason"] = "global_local_v2_logonly_fallback_invalid_json"
            return result

        qwen_id = int(obj["selected_candidate_id"])
        qwen_reason = str(obj.get("reason", ""))

        result["selected_candidate_id"] = qwen_id
        result["reason"] = qwen_reason or "global_local_v2_logonly_qwen_selected"

        gm = result.get("guard_metrics") or {}
        alignment = str(obj.get("global_local_alignment", "unknown")).lower()
        risk = str(obj.get("trajectory_risk", "unknown")).lower()
        confidence = str(obj.get("decision_confidence", "unknown")).lower()
        raw_should_change = bool(obj.get("should_change_original", False))

        gm["global_local_alignment"] = alignment
        gm["trajectory_risk"] = risk
        gm["decision_confidence"] = confidence
        gm["qwen_should_change_original_raw"] = raw_should_change

        if qwen_id not in valid_ids:
            result["fallback"] = True
            result["reason"] = f"global_local_v2_logonly_fallback_invalid_candidate_id: {qwen_id}"
            result["guard_metrics"] = gm
            return result

        qwen_cand = _get_candidate(candidate_records, qwen_id)
        if qwen_cand is None:
            result["fallback"] = True
            result["reason"] = f"global_local_v2_logonly_fallback_qwen_candidate_not_found: {qwen_id}"
            result["guard_metrics"] = gm
            return result

        rank_guard_ok, rank_guard_metrics = _rank_efficiency_guard(
            original_cand=original_cand,
            qwen_cand=qwen_cand,
            candidate_records=candidate_records,
        )

        selected_diff = int(qwen_id) != int(original_next_visit)

        # Post-hoc consistency gate. This prevents v1-style contradictions:
        # risk=high + should_change_original=true, or failed metrics + should_change=true.
        consistency_pass = bool(
            selected_diff
            and raw_should_change
            and rank_guard_ok
            and risk == "low"
            and alignment == "high"
            and confidence in ("medium", "high")
        )

        corrected_should_change = bool(consistency_pass)

        rank_guard_metrics["global_state"] = global_state
        rank_guard_metrics["global_local_alignment"] = alignment
        rank_guard_metrics["trajectory_risk"] = risk
        rank_guard_metrics["decision_confidence"] = confidence
        rank_guard_metrics["qwen_should_change_original_raw"] = raw_should_change
        rank_guard_metrics["should_change_original"] = corrected_should_change
        rank_guard_metrics["global_local_consistency_pass"] = consistency_pass
        rank_guard_metrics["global_local_v2_would_accept"] = consistency_pass
        rank_guard_metrics["global_local_v2_reject_reason"] = ""

        if not selected_diff:
            rank_guard_metrics["global_local_v2_reject_reason"] = "selected_original"
        elif not raw_should_change:
            rank_guard_metrics["global_local_v2_reject_reason"] = "qwen_should_change_false"
        elif not rank_guard_ok:
            rank_guard_metrics["global_local_v2_reject_reason"] = "rank_efficiency_guard_reject"
        elif risk != "low":
            rank_guard_metrics["global_local_v2_reject_reason"] = "trajectory_risk_not_low"
        elif alignment != "high":
            rank_guard_metrics["global_local_v2_reject_reason"] = "alignment_not_high"
        elif confidence not in ("medium", "high"):
            rank_guard_metrics["global_local_v2_reject_reason"] = "confidence_too_low"

        result["guard_metrics"] = rank_guard_metrics
        result["guarded_final_id"] = original_next_visit
        result["guard_accept_qwen"] = False
        result["fallback"] = False

        return result

    except Exception as e:
        result["fallback"] = True
        result["reason"] = f"qwen_global_local_v2_logonly_exception: {repr(e)}"
        result["guarded_final_id"] = original_next_visit
        result["selected_candidate_id"] = original_next_visit
        result["guard_accept_qwen"] = False
        traceback.print_exc()

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        return result


def qwen_global_local_v3_logonly_rerank(candidate_records, original_next_visit):
    """
    Global-local v3 soft-consistency Qwen reranker.

    This mode is log-only. It keeps the original ActiveSGM trajectory unchanged,
    but records whether Qwen's selected alternative passes a softer diagnostic
    consistency gate.
    """
    top_n = int(os.environ.get("QWEN_PLANNER_TOP_N", "3"))
    tie_gap_threshold = float(os.environ.get("QWEN_PLANNER_TIE_GAP", "0.10"))

    original_next_visit = int(original_next_visit)

    topn, _ = _rank_top_candidates(candidate_records, top_n)
    tie_case, tie_gap = _is_tie_case(topn, tie_gap_threshold)

    global_state = _build_global_state_summary(
        candidate_records=candidate_records,
        original_next_visit=original_next_visit,
        topn=topn,
        tie_gap=tie_gap,
    )

    result = {
        "selected_candidate_id": original_next_visit,
        "guarded_final_id": original_next_visit,
        "reason": "global_local_v3_logonly: keep original trajectory.",
        "fallback": False,
        "mode": "qwen_global_local_v3_logonly",
        "qwen_called": False,
        "tie_case": bool(tie_case),
        "tie_gap": tie_gap,
        "guard_accept_qwen": False,
        "guard_metrics": {
            "global_state": global_state,
            "global_local_alignment": "unknown",
            "trajectory_risk": "unknown",
            "decision_confidence": "unknown",
            "should_change_original": False,
            "qwen_should_change_original_raw": False,
            "global_local_consistency_pass": False,
            "global_local_v3_soft_gate_accept": False,
            "global_local_v3_would_accept": False,
            "global_local_v3_reject_reason": "",
        },
        "raw_response": "",
        "top_ids": [int(c["id"]) for c in topn],
    }

    if not tie_case:
        result["reason"] = "global_local_v3_logonly_skip_qwen_confident_top1"
        result["guard_metrics"]["global_local_v3_reject_reason"] = "not_tie_case"
        return result

    original_cand = _get_candidate(candidate_records, original_next_visit)
    if original_cand is None:
        result["fallback"] = True
        result["reason"] = "global_local_v3_logonly_fallback_original_candidate_not_found"
        result["guard_metrics"]["global_local_v3_reject_reason"] = "original_candidate_not_found"
        return result

    try:
        tok, model = _get_qwen_model()
        prompt = _build_global_local_v3_prompt(candidate_records, original_next_visit, topn, tie_gap)
        raw = _generate_json(tok, model, prompt)

        result["qwen_called"] = True
        result["raw_response"] = raw

        obj = _extract_json_object(raw)
        valid_ids = {int(c["id"]) for c in topn}

        if obj is None or not isinstance(obj.get("selected_candidate_id"), int):
            result["fallback"] = True
            result["reason"] = "global_local_v3_logonly_fallback_invalid_json"
            result["guard_metrics"]["global_local_v3_reject_reason"] = "invalid_json"
            return result

        qwen_id = int(obj["selected_candidate_id"])
        qwen_reason = str(obj.get("reason", ""))

        result["selected_candidate_id"] = qwen_id
        result["reason"] = qwen_reason or "global_local_v3_logonly_qwen_selected"

        gm = result.get("guard_metrics") or {}
        alignment = str(obj.get("global_local_alignment", "unknown")).lower()
        risk = str(obj.get("trajectory_risk", "unknown")).lower()
        confidence = str(obj.get("decision_confidence", "unknown")).lower()
        raw_should_change = bool(obj.get("should_change_original", False))

        gm["global_local_alignment"] = alignment
        gm["trajectory_risk"] = risk
        gm["decision_confidence"] = confidence
        gm["qwen_should_change_original_raw"] = raw_should_change

        if qwen_id not in valid_ids:
            result["fallback"] = True
            result["reason"] = f"global_local_v3_logonly_fallback_invalid_candidate_id: {qwen_id}"
            gm["global_local_v3_reject_reason"] = "invalid_candidate_id"
            result["guard_metrics"] = gm
            return result

        qwen_cand = _get_candidate(candidate_records, qwen_id)
        if qwen_cand is None:
            result["fallback"] = True
            result["reason"] = f"global_local_v3_logonly_fallback_qwen_candidate_not_found: {qwen_id}"
            gm["global_local_v3_reject_reason"] = "qwen_candidate_not_found"
            result["guard_metrics"] = gm
            return result

        _, rank_guard_metrics = _rank_efficiency_guard(
            original_cand=original_cand,
            qwen_cand=qwen_cand,
            candidate_records=candidate_records,
        )

        selected_diff = int(qwen_id) != int(original_next_visit)

        max_weighted_rank = int(os.environ.get("QWEN_V3_SOFT_MAX_WEIGHTED_RANK", "3"))
        max_score_drop = float(os.environ.get("QWEN_V3_SOFT_MAX_SCORE_DROP", "0.09"))
        min_distance_saving = float(os.environ.get("QWEN_V3_SOFT_MIN_DISTANCE_SAVING", "0.5"))
        min_stable_eff_ratio = float(os.environ.get("QWEN_V3_SOFT_MIN_STABLE_EFF_RATIO", "1.0"))

        selected_weighted_rank = int(rank_guard_metrics.get("selected_weighted_rank", 999))
        score_drop = float(rank_guard_metrics.get("score_drop_ratio_vs_original", 999.0))
        distance_saving = float(rank_guard_metrics.get("distance_saving_vs_original", -999.0))
        stable_eff_ratio = float(rank_guard_metrics.get("stable_efficiency_ratio_vs_original", 0.0))

        soft_rank_ok = bool(
            selected_weighted_rank <= max_weighted_rank
            and score_drop <= max_score_drop
            and distance_saving >= min_distance_saving
            and stable_eff_ratio >= min_stable_eff_ratio
        )

        risk_ok = risk == "low"
        alignment_ok = alignment in ("medium", "high")
        confidence_ok = confidence in ("medium", "high")

        # v3 soft consistency:
        # raw_should_change is not required, because v2 showed Qwen can be overly
        # conservative in that field. Instead, v3 records whether the selected
        # alternative satisfies structural and metric conditions.
        soft_consistency_pass = bool(
            selected_diff
            and soft_rank_ok
            and risk_ok
            and alignment_ok
            and confidence_ok
        )

        rank_guard_metrics["global_state"] = global_state
        rank_guard_metrics["global_local_alignment"] = alignment
        rank_guard_metrics["trajectory_risk"] = risk
        rank_guard_metrics["decision_confidence"] = confidence
        rank_guard_metrics["qwen_should_change_original_raw"] = raw_should_change
        rank_guard_metrics["should_change_original"] = False
        rank_guard_metrics["global_local_consistency_pass"] = soft_consistency_pass
        rank_guard_metrics["global_local_v3_soft_gate_accept"] = soft_rank_ok
        rank_guard_metrics["global_local_v3_would_accept"] = soft_consistency_pass
        rank_guard_metrics["global_local_v3_thresholds"] = {
            "selected_weighted_rank_max": max_weighted_rank,
            "score_drop_ratio_vs_original_max": max_score_drop,
            "distance_saving_vs_original_min": min_distance_saving,
            "stable_efficiency_ratio_vs_original_min": min_stable_eff_ratio,
            "allowed_global_local_alignment": ["medium", "high"],
            "allowed_trajectory_risk": ["low"],
            "allowed_decision_confidence": ["medium", "high"],
        }
        rank_guard_metrics["global_local_v3_reject_reason"] = ""

        if not selected_diff:
            rank_guard_metrics["global_local_v3_reject_reason"] = "selected_original"
        elif not soft_rank_ok:
            rank_guard_metrics["global_local_v3_reject_reason"] = "soft_rank_gate_reject"
        elif not risk_ok:
            rank_guard_metrics["global_local_v3_reject_reason"] = "trajectory_risk_not_low"
        elif not alignment_ok:
            rank_guard_metrics["global_local_v3_reject_reason"] = "alignment_too_low"
        elif not confidence_ok:
            rank_guard_metrics["global_local_v3_reject_reason"] = "confidence_too_low"

        result["guard_metrics"] = rank_guard_metrics
        result["guarded_final_id"] = original_next_visit
        result["guard_accept_qwen"] = False
        result["fallback"] = False

        return result

    except Exception as e:
        result["fallback"] = True
        result["reason"] = f"qwen_global_local_v3_logonly_exception: {repr(e)}"
        result["guarded_final_id"] = original_next_visit
        result["selected_candidate_id"] = original_next_visit
        result["guard_accept_qwen"] = False
        traceback.print_exc()

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

        return result

def qwen_rank_efficiency_apply_v2_rerank(candidate_records, original_next_visit):
    """
    Conservative apply-v2 mode for enhanced-metrics Qwen reranking.

    It reuses qwen_rank_efficiency_logonly_rerank to call Qwen and compute
    rank-efficiency guard metrics, then allows apply only when the
    rank-efficiency guard accepts the Qwen-selected candidate.

    This mode should be used together with:
      ACTIVE_SGM_LLM_APPLY=1
      QWEN_RANK_EFF_MIN_DISTANCE_SAVING=0.8
      QWEN_RANK_EFF_MIN_STABLE_EFF_RATIO=1.1
    """
    original_next_visit = int(original_next_visit)

    result = qwen_rank_efficiency_logonly_rerank(
        candidate_records=candidate_records,
        original_next_visit=original_next_visit,
    )

    result["mode"] = "qwen_rank_efficiency_apply_v2"

    selected_id = int(result.get("selected_candidate_id", original_next_visit))
    gm = result.get("guard_metrics") or {}
    rank_guard_ok = bool(gm.get("rank_efficiency_guard_accept"))

    if (
        result.get("qwen_called")
        and selected_id != original_next_visit
        and rank_guard_ok
    ):
        result["guarded_final_id"] = selected_id
        result["guard_accept_qwen"] = True
        result["fallback"] = False
        result["reason"] = (
            str(result.get("reason", "")) + " | apply_v2_rank_efficiency_accept"
        ).strip()
    else:
        result["guarded_final_id"] = original_next_visit
        result["guard_accept_qwen"] = False
        if result.get("qwen_called") and selected_id != original_next_visit:
            result["fallback"] = True
            result["reason"] = (
                str(result.get("reason", "")) + " | apply_v2_rank_efficiency_reject"
            ).strip()

    return result

def fake_llm_rerank(candidate_records, original_next_visit):
    """
    Backward-compatible entry point used by active_gs_planner_v2.py.

    Default mode keeps original ActiveSGM decision.
    Set:
      ACTIVE_SGM_LLM_MODE=qwen_tiebreak_top3_distance_strict
    to enable online Qwen Top-3 Tie-Break log-only reranking.
    """
    mode = os.environ.get("ACTIVE_SGM_LLM_MODE", "fake_llm").lower().strip()

    if mode == "qwen_global_local_v3_logonly":
        return qwen_global_local_v3_logonly_rerank(
            candidate_records=candidate_records,
            original_next_visit=original_next_visit,
        )

    if mode == "qwen_global_local_v2_logonly":
        return qwen_global_local_v2_logonly_rerank(
            candidate_records=candidate_records,
            original_next_visit=original_next_visit,
        )

    if mode == "qwen_global_local_logonly":
        return qwen_global_local_logonly_rerank(
            candidate_records=candidate_records,
            original_next_visit=original_next_visit,
        )

    if mode == "qwen_rank_efficiency_apply_v2":
        return qwen_rank_efficiency_apply_v2_rerank(
            candidate_records=candidate_records,
            original_next_visit=original_next_visit,
        )

    if mode == "qwen_rank_efficiency_logonly":
        return qwen_rank_efficiency_logonly_rerank(
            candidate_records=candidate_records,
            original_next_visit=original_next_visit,
        )

    if mode == "qwen_tiebreak_top3_distance_strict":
        return qwen_tiebreak_top3_distance_strict_rerank(
            candidate_records=candidate_records,
            original_next_visit=original_next_visit,
        )

    return _fake_keep_original(
        candidate_records=candidate_records,
        original_next_visit=original_next_visit,
        reason_prefix="fake_llm",
    )
