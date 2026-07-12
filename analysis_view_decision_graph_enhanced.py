import csv
from pathlib import Path
from collections import Counter, defaultdict

GRAPH_ALL = Path("analysis_view_decision_graph_all.csv")
DERIVED_CHANGED = Path("analysis_derived_planner_metrics_changed_steps.csv")

OUT_ALL = Path("analysis_view_decision_graph_all_enhanced.csv")
OUT_CHANGED = Path("analysis_view_decision_graph_changed_enhanced.csv")
OUT_SUMMARY = Path("analysis_view_decision_graph_enhanced_summary.txt")

METRIC_FIELDS = [
    "selected_weighted_rank",
    "selected_explore_rank",
    "selected_distance_rank",
    "weighted_ratio",
    "explore_ratio",
    "score_drop_ratio_vs_original",
    "distance_saving_vs_original",
    "distance_delta",
    "stable_efficiency_ratio_vs_original",
]

ALIASES = {
    "score_drop_ratio_vs_original": ["score_drop_ratio_vs_original", "score_drop", "score_drop_ratio"],
    "distance_saving_vs_original": ["distance_saving_vs_original", "distance_saving"],
    "stable_efficiency_ratio_vs_original": ["stable_efficiency_ratio_vs_original", "stable_eff_ratio_vs_original", "stable_eff_ratio", "stable_explore_efficiency_ratio_vs_original", "stable_efficiency_ratio"],
    "selected_weighted_rank": ["selected_weighted_rank", "weighted_rank"],
    "selected_explore_rank": ["selected_explore_rank", "explore_rank"],
    "selected_distance_rank": ["selected_distance_rank", "distance_rank"],
    "weighted_ratio": ["weighted_ratio"],
    "explore_ratio": ["explore_ratio"],
    "distance_delta": ["distance_delta"],
}

def read_csv(path):
    with path.open("r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def norm_run(x):
    x = str(x or "").strip().lower()
    if "apply-v1" in x or "apply_v1" in x:
        return "apply-v1"
    if "strict-v1" in x or "strict_v1" in x:
        return "strict-v1"
    if "apply-v2" in x or "apply_v2" in x:
        return "apply-v2"
    if "rank-eff" in x and "log" in x:
        return "rank-eff-logonly"
    if "log-only" in x or "log_only" in x:
        return "log-only"
    return x

def get_first(row, names):
    for name in names:
        if name in row and str(row.get(name, "")).strip() != "":
            return row.get(name)
    return ""

def is_zero_or_empty(x):
    s = str(x or "").strip()
    if s == "":
        return True
    try:
        return abs(float(s)) < 1e-12
    except Exception:
        return False

def is_true(x):
    return str(x).strip().lower() in {"true", "1", "yes"}

def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

if not GRAPH_ALL.exists():
    raise FileNotFoundError(f"Missing {GRAPH_ALL}")

if not DERIVED_CHANGED.exists():
    raise FileNotFoundError(f"Missing {DERIVED_CHANGED}")

graph_rows = read_csv(GRAPH_ALL)
derived_rows = read_csv(DERIVED_CHANGED)

derived_index = {}
for row in derived_rows:
    run = norm_run(get_first(row, ["run", "run_name", "experiment"]))
    step = str(get_first(row, ["step"])).strip()
    if not run or not step:
        continue

    packed = {}
    for field in METRIC_FIELDS:
        packed[field] = get_first(row, ALIASES.get(field, [field]))

    derived_index[(run, step)] = packed

enhanced_rows = []
merge_count = 0
changed_count = 0

for row in graph_rows:
    row = dict(row)
    run = norm_run(row.get("run"))
    step = str(row.get("step", "")).strip()
    final_changed = is_true(row.get("final_changed"))

    metric_source = "existing"

    if final_changed:
        changed_count += 1

    key = (run, step)
    derived = derived_index.get(key)

    if final_changed and derived:
        filled_any = False

        for field in METRIC_FIELDS:
            if field not in row:
                row[field] = ""

            if is_zero_or_empty(row.get(field)) and not is_zero_or_empty(derived.get(field)):
                row[field] = derived[field]
                filled_any = True

        if filled_any:
            metric_source = "derived_metrics_merge"
            merge_count += 1
        else:
            metric_source = "existing_or_no_better_derived"
    elif final_changed:
        # Some newer runs, e.g. apply-v2, already contain rank-efficiency metrics
        # in the original graph even if they are absent from the older derived table.
        has_existing_metrics = any(
            not is_zero_or_empty(row.get(field))
            for field in [
                "score_drop_ratio_vs_original",
                "distance_saving_vs_original",
                "stable_efficiency_ratio_vs_original",
            ]
        )
        metric_source = "existing_metrics" if has_existing_metrics else "missing_derived_metrics"
    else:
        metric_source = "not_changed"

    row["metric_source"] = metric_source
    enhanced_rows.append(row)

changed_rows = [r for r in enhanced_rows if is_true(r.get("final_changed"))]

write_csv(OUT_ALL, enhanced_rows)
write_csv(OUT_CHANGED, changed_rows)

lines = []
lines.append("Enhanced ActiveSGM View-Decision Graph Summary")
lines.append("=" * 80)
lines.append(f"Input graph: {GRAPH_ALL}")
lines.append(f"Derived changed metrics: {DERIVED_CHANGED}")
lines.append(f"Total graph records: {len(enhanced_rows)}")
lines.append(f"Changed records: {len(changed_rows)}")
lines.append(f"Changed records with merged derived metrics: {merge_count}")
lines.append("")

lines.append("Metric source counts:")
source_counter = Counter(r.get("metric_source") for r in enhanced_rows)
for k, v in source_counter.items():
    lines.append(f"  {k}: {v}")

lines.append("")
lines.append("Changed records by run and stage:")
group = defaultdict(list)
for r in changed_rows:
    group[(r.get("run"), r.get("stage"))].append(r)

for key in sorted(group.keys()):
    run, stage = key
    rows = group[key]
    lines.append(f"  run={run}, stage={stage}, count={len(rows)}, steps={[int(r['step']) for r in rows]}")

lines.append("")
lines.append("Enhanced changed record details:")
for r in changed_rows:
    lines.append(
        "  "
        f"run={r.get('run')}, "
        f"step={r.get('step')}, "
        f"stage={r.get('stage')}, "
        f"orig={r.get('original_next_visit')}, "
        f"selected={r.get('llm_selected_id')}, "
        f"final={r.get('final_next_visit')}, "
        f"score_drop={to_float(r.get('score_drop_ratio_vs_original')):.4f}, "
        f"distance_saving={to_float(r.get('distance_saving_vs_original')):.4f}, "
        f"stable_eff_ratio={to_float(r.get('stable_efficiency_ratio_vs_original')):.4f}, "
        f"metric_source={r.get('metric_source')}, "
        f"run_ate={r.get('run_ate_rmse_cm')}, "
        f"run_depth={r.get('run_depth_rmse_cm')}, "
        f"run_lpips={r.get('run_lpips')}"
    )

lines.append("")
lines.append("Interpretation:")
lines.append("  This enhanced graph fills older apply-v1 and strict-v1 changed-step metrics using analysis_derived_planner_metrics_changed_steps.csv.")
lines.append("  It makes the view-decision graph more suitable for trajectory-aware and stage-aware analysis.")
lines.append("  Remaining missing values indicate records for which older logs or derived analysis did not provide the requested metric.")

OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

print("\n".join(lines))
print("Wrote:", OUT_ALL)
print("Wrote:", OUT_CHANGED)
print("Wrote:", OUT_SUMMARY)
