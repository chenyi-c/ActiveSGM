# ActiveSGM + Qwen Planner: Stage-One Progress

## Research Question

Can an LLM assist ActiveSGM's next-best-view selection without replacing the
metric-based baseline planner or accepting unsafe low-quality choices?

## Implemented

- Candidate logging for id, distance, exploration information gain, semantic
  entropy, and weighted score.
- Qwen Top-3 selection for close-score cases.
- Numeric guard checks before an LLM recommendation can be applied.
- Separate log-only and apply-mode launch scripts.
- Offline reranking, guard simulation, decision inspection, and summary tools.
- Curated visual outputs for Replica `office0`.

The curated source release originates from branch
`feature_qwen25_15b_planner_v1` at commit `329c73aaaa27`.

## Planner Policy

ActiveSGM computes and ranks candidates first. Qwen can select only from the
Top-3 candidates in a tie-like case. In the recorded stage-one configuration,
an LLM recommendation is considered only when the metric guard accepts it:

- `weighted_ratio >= 0.90`
- `explore_ratio >= 0.80`
- `distance_delta <= 0.0`

In log-only mode the decision is stored for analysis. In apply mode the guarded
choice changes the next visit only when the apply switch is explicitly enabled.

## Stage-One Results

### Qwen Log-Only Reference

- 36 planner records, 21 Qwen calls, 21 tie cases.
- 5 guard-accepted Qwen recommendations.
- 0 applied changes by design.
- ATE RMSE: 122.69 cm; PSNR: 27.75; Depth RMSE: 0.53 cm; LPIPS: 0.092.

### Guarded Apply Run

- 42 planner records, 22 Qwen calls, 22 tie cases.
- 6 guard-accepted and applied planner changes.
- ATE RMSE: 120.37 cm; PSNR: 27.68; Depth RMSE: 0.83 cm; LPIPS: 0.088.

### Interpretation

The guarded policy demonstrably changes the trajectory and shows partial
improvement on ATE and LPIPS in this pair of runs. It also degrades the depth
metrics, so the evidence does not support a blanket quality-improvement claim.
The key stage-one outcome is a controllable, auditable LLM-assisted planning
mechanism rather than a finished optimal policy.

## Visual Assets

- `figures/office0_logonly_step_39.png`: log-only reference observation.
- `figures/office0_guarded_apply_step_39.png`: guarded-apply observation at a
  corresponding early step.
- `figures/office0_guarded_apply_step_539.png`: guarded-apply observation from
  a later changed-decision region.

These are selected presentation images. Full `results/` directories are
excluded because they are large run artifacts rather than source code.

## Next Steps

1. Tighten the guard, including a strictly improving distance criterion and a
   higher exploration threshold.
2. Audit each changed decision before another full GPU experiment.
3. Repeat controlled experiments across more scenes and seeds.
4. Migrate the environment and dependencies to the next GPU platform.
5. Integrate the validated planner interface into the second-stage drone system
   and prepare the research paper.
