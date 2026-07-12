# ActiveSGM Qwen Planner Integration Design

## Goal

Extend the existing ActiveSGM repository with the stage-one Qwen Planner work while preserving the baseline implementation, its attribution, and the current public history.

## Scope

- Preserve the existing ActiveSGM files and add the curated source package from `feature_qwen25_15b_planner_v1` at commit `329c73aaaa27`.
- Replace the top-level README with an accurate project overview that distinguishes the upstream baseline from the Qwen Planner extension.
- Add a concise progress report and selected presentation-ready figures.
- Keep large or non-redistributable assets out of Git: datasets, model weights, full run outputs, caches, environment directories, and third-party source trees.

## Public Research Claims

The Qwen component is an LLM-assisted, metric-guarded next-best-view extension. ActiveSGM scores candidates first; Qwen only breaks Top-3 ties; a numerical guard decides whether its recommendation can be applied. The release reports mixed stage-one evidence: the apply run changed six decisions and improved ATE/LPIPS against the logged baseline, while the depth metrics worsened. It must not claim an overall improvement.

## Repository Layout

- `src/`, `configs/`, `scripts/`, `envs/`: curated project source and runtime configuration.
- `docs/PROJECT_PROGRESS.md`: reproducible stage-one status and limitations.
- `docs/figures/`: a small set of presentation assets, each referenced from the README only after verification.
- `third_parties/`, `data/`, `results/`, weights, checkpoints, logs, and caches: intentionally absent from this public source release.

## Validation

Verify the source tree contains the Qwen Planner files, contains no prohibited large assets or credentials, and is clean after commit. Verify the published branch and merge commit are visible on GitHub.
