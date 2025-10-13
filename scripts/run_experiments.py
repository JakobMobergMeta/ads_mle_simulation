#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import pandas as pd

# These should be importable after `pip install -e .`
from simenv.model_performance_api import ModelPerformanceAPI
from archopt.config import (
    EVAL_TRAINING_DAYS, N_TRIALS, SEEDS, OUT_ROOT, BASELINE_ARCH
)
from archopt.evaluation import make_evaluator
from archopt.benchmark import Benchmark, RunLog
from archopt.logging_utils import save_runs, save_arches_per_method, plot_runs
from archopt.suggestors import (
    RandomSuggestor,
    SimulatedAnnealingSuggestor,
    LogSpaceSimulatedAnnealingSuggestor,
    TPESuggestor,                 # optional (if Optuna installed)
    SkoptBOSuggestor,             # optional (if scikit-optimize installed)
)
from archopt.modifiers import (
    generate_random_modifiers_list,
    save_modifiers_csv,
    load_modifiers_csv,
)

# -----------------------
# helpers
# -----------------------
def _resolve_out_root(root_dir: Path, out_root_from_cfg) -> Path:
    """Ensure OUT_ROOT is placed under the given root directory."""
    if isinstance(out_root_from_cfg, Path):
        return (root_dir / out_root_from_cfg.name).resolve()
    return (root_dir / str(out_root_from_cfg)).resolve()

def _mode_tag(soft_qps: bool, soft_qps_tau: float, qps_min: float) -> str:
    """Generate a stable suffix for soft/hard QPS modes."""
    if soft_qps:
        return f"softQPS_tau{soft_qps_tau:g}_Q{int(qps_min)}"
    return f"hardQPS_Q{int(qps_min)}"

def _best_trial_score(runlog: RunLog) -> float:
    return max((t.value for t in runlog.trials), default=float("-inf"))

def _summarize_results(results_dict: Dict[str, List[RunLog]]) -> Dict[str, Dict[str, float]]:
    """Aggregate all trial scores across all runs (seeds × modifiers) for each method."""
    method_stats = {}
    for method, runs in results_dict.items():
        all_scores = []
        for r in runs:
            # Collect ALL trial scores from this run (not just the best)
            for trial in r.trials:
                all_scores.append(trial.value)
        
        if all_scores:
            method_stats[method] = {
                "mean": sum(all_scores) / len(all_scores),
                "min": min(all_scores),
                "max": max(all_scores),
                "count": len(all_scores)
            }
        else:
            method_stats[method] = {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    
    return method_stats

def _ensure_report_dir(results_dir: Path) -> Path:
    """Return the results directory (it should already exist)"""
    return results_dir

# -----------------------
# core pipeline
# -----------------------
def run_all(
    root_dir: str,
    n_modifier_sets: int = 10,
    seed: int = 0,
    regenerate_modifiers: bool = False,
    *,
    qps_min: float = 3500.0,
    soft_qps: bool = False,
    soft_qps_tau: float = 0.15,
    single_default_modifier: bool = False,
    n_seeds: Optional[int] = None,
):
    """
    Run HPO experiments for multiple NETWORK_MODIFIERS sets.
    Results saved under: <root>/<OUT_ROOT.name>/<mode_tag>/modifiers_XX/
    """
    root = Path(root_dir).resolve()

    # Determine which seeds to use first so we know the count
    if n_seeds is not None:
        # Use specified number of seeds starting from the config SEEDS
        seeds_to_use = SEEDS[:n_seeds] if n_seeds <= len(SEEDS) else SEEDS + list(range(len(SEEDS), n_seeds))
    else:
        # Use all seeds from config
        seeds_to_use = SEEDS
    
    n_seeds_actual = len(seeds_to_use)

    # Handle single default modifier vs multiple modifiers
    if single_default_modifier:
        # Use the explicit default NETWORK_MODIFIERS from ModelPerformanceAPI
        default_modifier = {
            0: [0.2, 0.2, 1, 0.1, 0.1],
            1: [0.4, 0.2, 0.7, 0, 0.05],
            2: [0.1, 0.1, 0.6, 0.4, 0.05],
            3: [0.2, 0.4, 0.7, 0.3, 0.05],
            4: [0.1, 0.1, 0.9, 0.1, 0.1]
        }
        modifiers_list = [default_modifier]
        results_dir_name = f"1modifiers_{n_seeds_actual}seeds"
    else:
        # Multiple modifiers
        results_dir_name = f"{n_modifier_sets}modifiers_{n_seeds_actual}seeds"
        
    tasks_dir = root / "tasks" / "10 tries @60days no noise"
    results_dir = tasks_dir / results_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Separate OUT_ROOT per mode to avoid collisions
    out_root_base = _resolve_out_root(root, OUT_ROOT)
    mode_tag = _mode_tag(soft_qps=soft_qps, soft_qps_tau=soft_qps_tau, qps_min=qps_min)
    out_root = (out_root_base / mode_tag).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Load or generate modifiers ---
    if not single_default_modifier:
        modifiers_csv_path = results_dir / f"modifiers_{EVAL_TRAINING_DAYS}days_{N_TRIALS}tries_{n_modifier_sets}modifiers.csv"
        
        if modifiers_csv_path.exists() and not regenerate_modifiers:
            print(f"[run_all] Found existing modifiers at {modifiers_csv_path}, loading...")
            modifiers_list = load_modifiers_csv(modifiers_csv_path)
        else:
            if modifiers_csv_path.exists() and regenerate_modifiers:
                print(f"[run_all] Regenerating modifiers (overwriting {modifiers_csv_path})...")
            else:
                print(f"[run_all] No modifiers file found. Generating {n_modifier_sets} sets...")
            modifiers_list = generate_random_modifiers_list(
                n_sets=n_modifier_sets, step=0.05, seed=seed,
            )
            save_modifiers_csv(modifiers_list, modifiers_csv_path)
            print(f"[run_all] Saved {len(modifiers_list)} modifier sets to {modifiers_csv_path}")
    else:
        print("[run_all] Using single default modifier (no network modifications)")

    print(f"[run_all] Using {n_seeds_actual} seeds: {seeds_to_use}")

    # --- Build suggestors (optional ones only if installed) ---
    suggestors = [
        RandomSuggestor(),
        SimulatedAnnealingSuggestor(T0=0.5, alpha=0.996, warmup=3),
        LogSpaceSimulatedAnnealingSuggestor(T0=0.5, alpha=0.996, warmup=3),
    ]
    # Optional: Optuna TPE
    try:
        suggestors.append(TPESuggestor(initial_arch=BASELINE_ARCH))
    except Exception as e:
        print("[info] Optuna not installed; skipping TPESuggestor.")
    # Optional: scikit-optimize baselines
    try:
        suggestors += [
            SkoptBOSuggestor("dummy", name="Skopt-Random"),
            SkoptBOSuggestor("GP",    name="Skopt-GP"),
            SkoptBOSuggestor("RF",    name="Skopt-RF"),
            SkoptBOSuggestor("ET",    name="Skopt-ET"),
            SkoptBOSuggestor("GBRT",  name="Skopt-GBRT"),
        ]
    except Exception:
        print("[info] scikit-optimize not installed; skipping Skopt* baselines.")

    # --- Run experiments for each modifier set ---
    all_runs: Dict[str, Dict[str, List[RunLog]]] = {}

    for idx, modifiers in enumerate(modifiers_list, start=1):
        if single_default_modifier:
            label = "default_modifier"
        else:
            label = f"modifiers_{idx:02d}"

        cfg = {
            "GLOBAL_NOISE_SCALE": 0.0,
            "BASELINE_ARCH": BASELINE_ARCH,
            "BASELINE_DAY": EVAL_TRAINING_DAYS,
            "NETWORK_MODIFIERS": modifiers,
        }
        model_object = ModelPerformanceAPI(cfg)

        # Evaluator: set search-time QPS strategy
        # Hard vs soft is controlled by passing soft_qps_tau=None (hard) or a float (soft)
        evaluate_cfg = make_evaluator(
            model_object,
            qps_min=qps_min,
            soft_qps_tau=(soft_qps_tau if soft_qps else None),
        )

        bench = Benchmark(evaluate_cfg)

        results: Dict[str, List[RunLog]] = {}
        for sug in suggestors:
            results[sug.name] = []
            for s in seeds_to_use:
                # Run N_TRIALS + 1 total: step 0 = baseline, steps 1-N_TRIALS = additional trials
                run_result = bench.run(sug, n_trials=N_TRIALS + 1, seed=s, initial_arch=BASELINE_ARCH)
                results[sug.name].append(run_result)

        out_dir = out_root / label
        save_runs(results, out_dir)
        save_arches_per_method(results, out_dir)
        plot_runs(results, title_prefix=f"[{label}] ")

        baseline_ne = model_object.get_default_model_config_dict()["BASELINE_NE"]
        print(f"[{label}] Saved CSVs to: {out_dir.resolve()}")
        print(f"[{label}] Methods: {list(results.keys())}")
        print(f"[{label}] BASELINE_NE (computed) = {baseline_ne:.6f}")
        print("-" * 60)

        all_runs[label] = results

    return all_runs, modifiers_list, mode_tag, results_dir

# -----------------------
# reports
# -----------------------
def write_all_reports(
    all_runs_by_label: Dict[str, Dict[str, List[RunLog]]],
    modifiers_list: List[Dict[int, List[float]]],
    results_dir: Path,
    *,
    qps_min: float,
    mode_tag: str,
):
    """
    Aggregate across all NETWORK_MODIFIERS.
    Saves report files with the mode tag in the filename.
    """
    report_dir = _ensure_report_dir(results_dir)
    # Generate filenames based on mode - no mode tag for hard QPS
    if mode_tag.startswith("hardQPS"):
        baseline_txt = report_dir / "results_baselines.txt"
        details_txt = report_dir / "results_details_baselines.txt"
    else:
        baseline_txt = report_dir / f"results_baselines__{mode_tag}.txt"
        details_txt = report_dir / f"results_details_baselines__{mode_tag}.txt"
    # Generate filename based on mode - no mode tag for hard QPS
    if mode_tag.startswith("hardQPS"):
        flat_csv = report_dir / f"Sim_ML_Bench_l3_60days_{N_TRIALS}tries_baselines.csv"
    else:
        flat_csv = report_dir / f"Sim_ML_Bench_l3_60days_{N_TRIALS}tries_baselines__{mode_tag}.csv"

    # --- Text summary (average/min/max of best-valid-QPS per optimizer across modifiers) ---
    # Only “best trials” that meet QPS ≥ qps_min are considered (handled in evaluation/reporting flow).
    lines: List[str] = []
    lines.append("Reults:\n")  # (sic)

    # Collect BEST score from each (seed × modifier) combination
    per_method_best_scores: Dict[str, List[float]] = {}
    labels = sorted(all_runs_by_label.keys())

    for label in labels:
        results = all_runs_by_label[label]
        for method, runs in results.items():
            for run in runs:  # Each run is one (seed × modifier) combination
                # Take the best score from this run (best trial within this seed×modifier)
                best_score = max((trial.value for trial in run.trials), default=float("-inf"))
                if best_score != float("-inf"):
                    per_method_best_scores.setdefault(method, []).append(best_score)

    lines.append("Rules based (optimizers): (target 3) — aggregated across all seed×modifier combinations")
    for method in sorted(per_method_best_scores.keys()):
        scores = per_method_best_scores[method]
        if len(scores) == 0:
            avg = mn = mx = 0.0
        else:
            # average, min, max across ALL evaluations
            avg = sum(scores) / len(scores)
            mn = min(scores)
            mx = max(scores)
        lines.append(f"{method} {avg:.6f} (min={mn:.6f}, max={mx:.6f}, n={len(scores)})")

    lines.append("")
    lines.append(f"[baseline] arch={BASELINE_ARCH} days={EVAL_TRAINING_DAYS}")

    with baseline_txt.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote report to: {baseline_txt.resolve()}")

    # --- Detailed text log (all steps) ---
    with details_txt.open("w", encoding="utf-8") as f:
        f.write("Reults Details:\n\n")
        for label in labels:
            f.write(f"=== Modifiers: {label} ===\n")
            results = all_runs_by_label[label]
            for method, runs in results.items():
                f.write(f"Method: {method}\n")
                for i, run in enumerate(runs):
                    seed_id = run.trials[0].seed if run.trials else i
                    f.write(f"  Seed {seed_id}:\n")
                    for step, tr in enumerate(run.trials, start=0):
                        arch_str = json.dumps(tr.config.get("arch"))
                        f.write(f"    Step {step}: arch={arch_str}, score={tr.value:.6f}\n")
                f.write("\n")
            f.write("\n")
    print(f"Wrote detailed report to: {details_txt.resolve()}")

    # --- Flat CSV (recompute NE/QPS with the correct model per label) ---
    rows: List[Dict] = []
    for idx, label in enumerate(labels, start=1):
        modifiers = modifiers_list[idx - 1]
        # rebuild the model per label to ensure consistency
        cfg = {
            "GLOBAL_NOISE_SCALE": 0.0,
            "BASELINE_ARCH": BASELINE_ARCH,
            "BASELINE_DAY": EVAL_TRAINING_DAYS,
            "NETWORK_MODIFIERS": modifiers,
        }
        model_for_label = ModelPerformanceAPI(cfg)

        results = all_runs_by_label[label]
        for method, runs in results.items():
            for run in runs:
                for step, tr in enumerate(run.trials, start=0):
                    arch = tr.config.get("arch")
                    ne, qps, _ = model_for_label.train_model(
                        arch=arch,
                        training_days=EVAL_TRAINING_DAYS,
                        ignore_budget=True,
                    )
                    score = float(model_for_label.get_score(ne))
                    rows.append({
                        "ModeTag": mode_tag,
                        "ModifiersLabel": label,
                        "Competitor": method,
                        "seed": tr.seed,
                        "step": step,
                        "Trial": step,
                        "Arch": json.dumps(arch),
                        "score": float(score),
                        "qps": float(qps),
                        "ne": float(ne),
                    })

    df = pd.DataFrame(
        rows,
        columns=["ModeTag","ModifiersLabel","Competitor","seed","step","Trial","Arch","score","qps","ne"]
    )
    df.to_csv(flat_csv, index=False)
    print(f"Saved: {flat_csv.resolve()}")

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Run HPO experiments with network modifiers.")
    parser.add_argument(
        "--root-dir",
        type=str,
        default=".",
        help="Root directory for tasks and outputs.",
    )
    parser.add_argument(
        "--n-modifiers",
        type=int,
        default=100,
        help="Number of random modifier sets to generate if not loading existing file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for generating random modifiers (when generating).",
    )
    parser.add_argument(
        "--regenerate-modifiers",
        action="store_true",
        help="Force regeneration of modifiers.csv even if it exists.",
    )
    parser.add_argument(
        "--qps-min",
        type=float,
        default=3500.0,
        help="QPS floor for reporting best valid trials (and for hard search mode).",
    )
    parser.add_argument(
        "--soft-qps",
        action="store_true",
        help="Use soft QPS penalty (search-time) instead of hard rejection.",
    )
    parser.add_argument(
        "--soft-qps-tau",
        type=float,
        default=0.15,
        help="Softness parameter for soft-QPS penalty.",
    )
    parser.add_argument(
        "--single-default-modifier",
        action="store_true",
        help="Use single default modifier (no network modifications) instead of multiple random modifiers.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Number of seeds to use (overrides config SEEDS). If not specified, uses all seeds from config.",
    )
    args = parser.parse_args()

    all_runs, modifiers_list, mode_tag, results_dir = run_all(
        root_dir=args.root_dir,
        n_modifier_sets=args.n_modifiers,
        seed=args.seed,
        regenerate_modifiers=args.regenerate_modifiers,
        qps_min=args.qps_min,
        soft_qps=args.soft_qps,
        soft_qps_tau=args.soft_qps_tau,
        single_default_modifier=args.single_default_modifier,
        n_seeds=args.n_seeds,
    )

    write_all_reports(
        all_runs_by_label=all_runs,
        modifiers_list=modifiers_list,
        results_dir=results_dir,
        qps_min=args.qps_min,
        mode_tag=mode_tag,
    )

if __name__ == "__main__":
    main()
