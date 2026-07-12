# ActiveSGM Qwen Planner Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the curated ActiveSGM + Qwen Planner source package, status documentation, and selected figures into the existing public ActiveSGM repository.

**Architecture:** Keep `main` as the baseline history and prepare the update on `qwen-planner-stage1`. Overlay only the curated source tree, use project documentation to explain the extension and mixed results, then verify the public release contains neither data nor model artifacts before merging.

**Tech Stack:** Git, Python source package, shell launch scripts, Markdown documentation.

---

### Task 1: Prepare the integration branch

**Files:**
- Create: `docs/superpowers/specs/2026-07-13-activesgm-qwen-planner-integration-design.md`
- Create: `docs/superpowers/plans/2026-07-13-activesgm-qwen-planner-integration.md`

- [x] Create `qwen-planner-stage1` from the current `main` commit.
- [x] Record the source branch `feature_qwen25_15b_planner_v1` and commit `329c73aaaa27` in the design document.
- [x] Verify `git status --short` is clean before importing source files.

### Task 2: Import curated Qwen Planner source

**Files:**
- Modify: `src/`
- Modify: `configs/`
- Modify: `scripts/`
- Modify: `envs/`
- Add: top-level `run_*.sh`, `test_*.sh`, `analysis_*.py`, `offline_*.py`, and experiment notes from the curated source package.

- [x] Copy only the curated main-code package into the repository.
- [x] Do not copy `third_parties/`, `data/`, `results/`, `.git/`, model weights, checkpoints, logs, caches, or environment directories.
- [x] Verify the imported tree includes Qwen planner, reranker, unit-test, and guarded-run entry points.

### Task 3: Add accurate project documentation and figures

**Files:**
- Modify: `README.md`
- Create: `docs/PROJECT_PROGRESS.md`
- Create: `docs/figures/`

- [x] Document the baseline ActiveSGM attribution and the guarded Top-3 Qwen Planner extension.
- [x] Report the stage-one result as mixed: ATE/LPIPS improved in the referenced apply run, while depth metrics worsened.
- [x] Add only selected, locally verified presentation figures and caption each figure with its role.

### Task 4: Verify and publish

**Files:**
- Verify: repository tree and Git history.

- [ ] Run a source-tree and forbidden-artifact scan.
- [ ] Review `git diff --check`, `git status --short`, and the staged file list.
- [ ] Commit the branch, push it to GitHub, merge it into `main`, and verify the resulting GitHub files and commit history.
