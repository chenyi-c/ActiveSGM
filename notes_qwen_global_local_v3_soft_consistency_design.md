# Qwen Global-Local V3 Soft-Consistency Design

## Motivation

The previous global-local experiments reveal a clear trade-off.

Global-local v1 allowed Qwen to reason about global trajectory risk and local candidate quality, but its structured outputs were sometimes inconsistent. For example, it could mark a candidate as high trajectory risk while still suggesting that the original ActiveSGM planner choice should be changed.

Global-local v2 fixed this issue by introducing stricter consistency rules and deterministic post-filtering. The repeat log-only experiments show that v2 is safe and structurally stable:
- apply_used = 0
- final_changed = 0
- global_local_v2_would_accept = 0
- qwen_called_unknown_or_missing_fields = 0

However, v2 is overly conservative. Across repeated runs, Qwen sometimes selects a different candidate from the original planner, but qwen_should_change_original_raw remains 0 and global_local_v2_would_accept remains 0. Therefore, v2 acts more like a risk auditor than an actionable planner assistant.

The purpose of v3 is to keep the safety and consistency of v2, while relaxing the decision rule enough to produce useful hypothetical accept candidates.

## Research Question

Can Qwen identify low-risk, high-efficiency alternative viewpoints under global-local consistency constraints, without directly changing the ActiveSGM trajectory?

## Key Principle

V3 remains log-only.

It does not change the actual camera trajectory. It only records whether a candidate would be acceptable under a softer consistency rule.

Therefore:
- final_next_visit must remain equal to original_next_visit.
- apply_used must remain false.
- guard_accept_qwen must remain false.
- v3_would_accept is only a diagnostic signal.

## V3 Decision Logic

V3 separates two concepts:

1. Qwen preference:
   Whether Qwen selects a different candidate and explains it as globally/local consistent.

2. Soft deterministic accept:
   Whether the selected candidate passes a relaxed but still safe metric gate.

A candidate can be marked as global_local_v3_would_accept only if all of the following are true:

### Required structural conditions

- selected_candidate_id differs from original_next_visit.
- trajectory_risk is low.
- global_local_alignment is medium or high.
- decision_confidence is medium or high.
- Qwen does not explicitly say the original is safer.
- Qwen reason has no detected consistency contradiction.

### Required deterministic metric conditions

The selected candidate should satisfy a soft rank-efficiency gate:

- selected_weighted_rank <= 3
- score_drop_ratio_vs_original <= 0.09
- distance_saving_vs_original >= 0.5
- stable_efficiency_ratio_vs_original >= 1.0

This is softer than v2:
- v2 required distance_saving >= 0.8
- v2 required stable_efficiency_ratio >= 1.1
- v2 required alignment = high

V3 allows:
- alignment = medium or high
- distance_saving >= 0.5
- stable_efficiency_ratio >= 1.0

## Why this is scientifically reasonable

The planner is a sequential decision system. A small local improvement is not guaranteed to improve the final trajectory. However, an overly strict rule prevents the system from collecting candidate alternatives for analysis.

V3 is not designed to immediately improve final metrics. It is designed to create a safe set of hypothetical alternative decisions. These decisions can later be inspected, compared with view-decision graph features, and used to decide whether a future apply-v3 experiment is justified.

## Expected Outcome

A successful v3 log-only run should satisfy:

- final_changed = 0
- apply_used = 0
- guard_accept_qwen = 0
- qwen_called_unknown_or_missing_fields = 0
- reason_flags = 0
- global_local_v3_would_accept > 0

If global_local_v3_would_accept remains 0, v3 is still too conservative.

If reason_flags > 0 or high-risk candidates are accepted, v3 is unsafe.

If v3 produces a small number of low-risk would-accept candidates, the next step is to inspect those candidate decisions before any real apply experiment.

## Planned Experiments

1. Implement qwen_global_local_v3_logonly.
2. Run no-GPU smoke test with mocked Qwen output.
3. Run real Qwen GPU unit test.
4. Run office0 full online log-only.
5. Analyze:
   - qwen_called
   - qwen_selected_diff_from_original
   - qwen_should_change_original_raw
   - global_local_v3_would_accept
   - reject_reason distribution
   - reason consistency flags
6. Compare v3 with v1/v2 and view-decision graph.
7. Decide whether apply-v3 is justified.
