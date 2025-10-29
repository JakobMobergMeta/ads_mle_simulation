#!/usr/bin/env python3
import argparse
import sys
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
import archopt.suggestors as suggestors_module
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
from archopt.networks import (
    sample_arch_pow2,     # nested sampler (uses current width ladder)
    validate_arch_nested, # sanity checks
    set_width_mode,       # configure linear vs pow2 widths
)
import numpy as np

import random

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

def _derive_competition_name(folder_path: Path, root_dir: Path) -> str:
    """
    Derive competition name from folder path by taking the path relative to root_dir
    and joining directory names with "-".

    Example:
        folder = /path/to/tasks/20 tries @60days no noise/skewed_modifiers_10seeds
        root = /path/to
        returns: tasks-20 tries @60days no noise-skewed_modifiers_10seeds
    """
    try:
        # Get the path relative to root
        rel_path = folder_path.relative_to(root_dir)
        # Join parts with "-" instead of "/"
        return str(rel_path).replace("/", "-")
    except ValueError:
        # If folder is not relative to root, just use the folder name
        return folder_path.name

# def _generate_valid_baseline(model_api: ModelPerformanceAPI, qps_min: float, seed: int, max_attempts: int = 10000, verbose: bool = False) -> Optional[List[int]]:
#     """Generate a random valid baseline architecture that meets QPS >= qps_min."""
#     rng = random.Random(seed)

#     # Try to generate diverse architectures by rejecting the first few valid ones
#     # This ensures we explore more of the architecture space
#     valid_archs = []
#     valid_qps = []

#     for attempt in range(max_attempts):
#         # Generate random number of layers (e.g., 3-7 layers)
#         n_layers = rng.randint(3, 7)
#         # Generate random architecture with valid layer sizes
#         # Use a wider range and more options for diversity
#         arch = [rng.choice([32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]) for _ in range(n_layers)]
#         print(f"Attempt {attempt}: arch={arch}")

#         try:
#             # Test if this architecture meets QPS requirement
#             ne, qps, _ = model_api.train_model(
#                 arch=arch,
#                 training_days=EVAL_TRAINING_DAYS,
#                 ignore_budget=True,
#             )
#             if verbose and attempt % 100 == 0:
#                 print(f"  Attempt {attempt}: arch={arch}, QPS={qps:.2f} (min={qps_min})")

#             if qps >= qps_min:
#                 valid_archs.append(arch)
#                 valid_qps.append(qps)
#                 if verbose:
#                     print(f"  ✓ Found valid arch {len(valid_archs)}: {arch}, QPS={qps:.2f}")
#                 # Collect up to 10 valid architectures, then pick one randomly
#                 if len(valid_archs) >= 10:
#                     chosen_idx = rng.choice(range(len(valid_archs)))
#                     chosen_arch = valid_archs[chosen_idx]
#                     if verbose:
#                         print(f"  → Selected arch with QPS={valid_qps[chosen_idx]:.2f}: {chosen_arch}")
#                     return chosen_arch
#         except Exception as e:
#             # Skip invalid architectures
#             if verbose and attempt % 100 == 0:
#                 print(f"  Attempt {attempt}: arch={arch}, Error: {e}")
#             continue

#     # If we found at least one valid architecture, return a random one
#     if valid_archs:
#         chosen_idx = rng.choice(range(len(valid_archs)))
#         chosen_arch = valid_archs[chosen_idx]
#         if verbose:
#             print(f"  → Selected from {len(valid_archs)} valid archs, QPS={valid_qps[chosen_idx]:.2f}: {chosen_arch}")
#         return chosen_arch

#     # If we couldn't find any valid baseline, return None
#     print(f"Warning: Could not generate valid baseline after {max_attempts} attempts")
#     return None

def _generate_valid_baseline(
    model_api: ModelPerformanceAPI,
    qps_min: float,
    seed: int,
    max_attempts: int = 10000,
    verbose: bool = False
) -> Optional[List[List[int]]]:
    """
    Generate a random **nested** baseline architecture that meets QPS >= qps_min.
    - Uses powers-of-two widths and nested format: List[List[int]]
    - Samples via archopt.networks.sample_arch_pow2
    - Collects multiple valid candidates for diversity, then picks one.
    """
    rng = np.random.RandomState(seed)

    valid_archs: List[List[List[int]]] = []
    valid_qps: List[float] = []
    seen: set = set()  # avoid re-evaluating duplicates

    for attempt in range(max_attempts):
        # Sample a nested, powers-of-two arch (e.g., [[128,128,128],[256,256]])
        arch = sample_arch_pow2(rng)

        # De-duplicate by a tuple-ized key
        key = tuple(tuple(sub) for sub in arch)
        if key in seen:
            continue
        seen.add(key)

        # (Optional) print progress
        if verbose and attempt % 100 == 0:
            print(f"  Attempt {attempt}: arch={arch}")

        try:
            # Sanity check against global constraints
            validate_arch_nested(arch)

            # Evaluate QPS feasibility
            ne, qps, _ = model_api.train_model(
                arch=arch,
                training_days=EVAL_TRAINING_DAYS,
                ignore_budget=True,
            )

            if verbose and attempt % 100 == 0:
                print(f"    -> QPS={qps:.2f} (min={qps_min})")

            if qps >= qps_min:
                valid_archs.append(arch)
                valid_qps.append(qps)
                if verbose:
                    print(f"    ✓ Valid #{len(valid_archs)}: QPS={qps:.2f}, arch={arch}")

                # Gather up to 10 valid candidates to encourage diversity
                if len(valid_archs) >= 10:
                    idx = int(rng.randint(0, len(valid_archs)))
                    if verbose:
                        print(f"    → Selected (from 10) QPS={valid_qps[idx]:.2f}: {valid_archs[idx]}")
                    return valid_archs[idx]

        except Exception as e:
            # Skip invalid / failing samples; optionally log
            if verbose and attempt % 100 == 0:
                print(f"    ! Error on attempt {attempt}: {e}")
            continue

    # If no 10-candidate pool, return a random one among found valids
    if valid_archs:
        idx = int(rng.randint(0, len(valid_archs)))
        if verbose:
            print(f"  → Selected (from {len(valid_archs)} valids) QPS={valid_qps[idx]:.2f}: {valid_archs[idx]}")
        return valid_archs[idx]

    print(f"Warning: Could not generate valid baseline after {max_attempts} attempts")
    return None


def save_modifiers_and_baselines_csv(modifiers_list: List[Dict[int, List[float]]], baselines_list: List[List[int]], filepath: Path):
    """Save both modifiers and baseline architectures to a single CSV."""
    rows = []
    for idx, (modifiers, baseline) in enumerate(zip(modifiers_list, baselines_list), start=1):
        rows.append({
            "config_id": idx,
            "baseline_architecture": json.dumps(baseline),
            "network_modifier": json.dumps(modifiers)
        })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)

def load_modifiers_and_baselines_csv(filepath: Path) -> Tuple[List[Dict[int, List[float]]], List[List[int]]]:
    """Load both modifiers and baseline architectures from a single CSV."""
    df = pd.read_csv(filepath)
    modifiers_list = []
    baselines_list = []

    for _, row in df.iterrows():
        # Load baseline
        baseline = json.loads(row["baseline_architecture"])
        baselines_list.append(baseline)

        # Load modifiers - the JSON keys are strings, convert them back to integers
        modifiers_dict = json.loads(row["network_modifier"])
        modifiers = {int(k): v for k, v in modifiers_dict.items()}
        modifiers_list.append(modifiers)

    return modifiers_list, baselines_list

def load_full_config_csv(filepath: Path) -> Tuple[List[Dict[int, List[float]]], List[List[int]], int, int, int, float]:
    """Load modifiers, baselines, and experiment parameters from a single CSV.

    Returns:
        modifiers_list: List of network modifier dictionaries
        baselines_list: List of baseline architectures
        num_trials: Number of trials to run
        num_seeds: Number of seeds to use
        days: Number of training days
        qps_min: Minimum QPS requirement
    """
    df = pd.read_csv(filepath)
    modifiers_list = []
    baselines_list = []

    # Get experiment parameters from the first row (should be same for all rows)
    num_trials = int(df.iloc[0]["num_trials"])
    num_seeds = int(df.iloc[0]["num_seeds"])
    days = int(df.iloc[0]["days"])
    qps_min = float(df.iloc[0]["qps_min"]) if "qps_min" in df.columns else 3500.0

    for _, row in df.iterrows():
        # Load baseline
        baseline = json.loads(row["baseline_architecture"])
        baselines_list.append(baseline)

        # Load modifiers - the JSON keys are strings, convert them back to integers
        modifiers_dict = json.loads(row["network_modifier"])
        modifiers = {int(k): v for k, v in modifiers_dict.items()}
        modifiers_list.append(modifiers)

    return modifiers_list, baselines_list, num_trials, num_seeds, days, qps_min

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

def _save_command_file(folder_path: Path, command_args: argparse.Namespace):
    """Save the command that was run to a command.sh file in the folder."""
    command_parts = [sys.executable, "scripts/run_experiments.py"]

    # Add all arguments that were specified
    if command_args.folder:
        command_parts.append(f"--folder \"{command_args.folder}\"")
    if command_args.root_dir != ".":
        command_parts.append(f"--root-dir \"{command_args.root_dir}\"")
    if command_args.n_modifiers != 100:
        command_parts.append(f"--n-modifiers {command_args.n_modifiers}")
    if command_args.seed != 0:
        command_parts.append(f"--seed {command_args.seed}")
    if command_args.regenerate_modifiers:
        command_parts.append("--regenerate-modifiers")
    if command_args.qps_min != 3500.0:
        command_parts.append(f"--qps-min {command_args.qps_min}")
    if command_args.soft_qps:
        command_parts.append("--soft-qps")
    if command_args.soft_qps_tau != 0.15:
        command_parts.append(f"--soft-qps-tau {command_args.soft_qps_tau}")
    if command_args.single_default_modifier:
        command_parts.append("--single-default-modifier")
    if command_args.n_seeds is not None:
        command_parts.append(f"--n-seeds {command_args.n_seeds}")
    if command_args.verbose_baseline_gen:
        command_parts.append("--verbose-baseline-gen")
    if command_args.num_layers != 5:
        command_parts.append(f"--num-layers {command_args.num_layers}")
    if command_args.allow_variable_subarches:
        command_parts.append("--allow-variable-subarches")
    if command_args.use_pow2_widths:
        command_parts.append("--use-pow2-widths")

    command_str = " \\\n    ".join(command_parts)

    command_file = folder_path / "command.sh"
    with command_file.open("w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Command used to run this experiment\n")
        f.write(f"# Generated automatically by run_experiments.py\n\n")
        f.write(command_str + "\n")

    # Make the file executable
    command_file.chmod(0o755)
    print(f"[run_from_folder] Saved command to: {command_file.resolve()}")

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
    verbose_baseline_gen: bool = False,
    random_baselines: bool = False,
    num_layers: int = 5,
    fixed_depth: bool = True,
    use_pow2_widths: bool = False,
):
    """
    Run HPO experiments for multiple NETWORK_MODIFIERS sets.
    Results saved under: <root>/<OUT_ROOT.name>/<mode_tag>/modifiers_XX/
    """
    # Set global flags
    suggestors_module.FIXED_NUM_SUBARCHES_MODE = fixed_depth
    set_width_mode(use_pow2=use_pow2_widths)

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

    # --- Load or generate modifiers and baselines ---
    if single_default_modifier:
        n_configs = 1
    else:
        n_configs = n_modifier_sets

    config_csv_path = results_dir / f"configs_{EVAL_TRAINING_DAYS}days_{N_TRIALS}tries_{n_configs}configs.csv"

    if config_csv_path.exists() and not regenerate_modifiers:
        print(f"[run_all] Found existing configs at {config_csv_path}, loading...")
        modifiers_list, baselines_list = load_modifiers_and_baselines_csv(config_csv_path)
    else:
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
            baselines_list = [BASELINE_ARCH]  # Use default baseline for single default modifier
            print("[run_all] Using single default modifier (no network modifications)")
        else:
            if config_csv_path.exists() and regenerate_modifiers:
                print(f"[run_all] Regenerating configs (overwriting {config_csv_path})...")
            else:
                print(f"[run_all] No config file found. Generating {n_modifier_sets} sets...")

            modifiers_list = generate_random_modifiers_list(
                n_sets=n_modifier_sets, num_subarches=num_layers, step=0.05, seed=seed,
            )

            # Generate baselines for each modifier
            print(f"[run_all] Generating baselines for {len(modifiers_list)} modifier sets...")
            baselines_list = []
            for idx, modifiers in enumerate(modifiers_list):
                if verbose_baseline_gen:
                    print(f"\n[run_all] Generating baseline for modifier set {idx+1}/{len(modifiers_list)}...")
                # Create a temporary model API with this modifier to generate valid baseline
                cfg = {
                    "GLOBAL_NOISE_SCALE": 0.0,
                    "BASELINE_ARCH": BASELINE_ARCH,
                    "BASELINE_DAY": EVAL_TRAINING_DAYS,
                    "NETWORK_MODIFIERS": modifiers,
                }
                temp_model = ModelPerformanceAPI(cfg)
                # Use a large multiplier to create more variance in baseline seeds
                baseline_seed = seed * 10000 + idx * 123
                baseline_arch = _generate_valid_baseline(temp_model, qps_min, seed=baseline_seed, verbose=verbose_baseline_gen)
                if baseline_arch is None:
                    # Fallback to default BASELINE_ARCH if generation fails
                    print(f"[run_all] Using fallback BASELINE_ARCH for modifier set {idx+1}")
                    baseline_arch = BASELINE_ARCH
                baselines_list.append(baseline_arch)

        # Save combined config file
        save_modifiers_and_baselines_csv(modifiers_list, baselines_list, config_csv_path)
        print(f"[run_all] Saved {len(modifiers_list)} configs (modifiers + baselines) to {config_csv_path}")

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
    # try:
    suggestors += [
        SkoptBOSuggestor("dummy", name="Skopt-Random"),
        SkoptBOSuggestor("GP",    name="Skopt-GP"),
        SkoptBOSuggestor("RF",    name="Skopt-RF"),
        SkoptBOSuggestor("ET",    name="Skopt-ET"),
        SkoptBOSuggestor("GBRT",  name="Skopt-GBRT"),
    ]
    # except Exception:
    #     print("[info] scikit-optimize not installed; skipping Skopt* baselines.")

    # --- Run experiments for each modifier set ---
    all_runs: Dict[str, Dict[str, List[RunLog]]] = {}

    for idx, modifiers in enumerate(modifiers_list, start=1):
        if single_default_modifier:
            label = "default_modifier"
        else:
            label = f"modifiers_{idx:02d}"

        # Get the corresponding baseline for this modifier
        baseline_for_modifier = baselines_list[idx - 1]

        cfg = {
            "GLOBAL_NOISE_SCALE": 0.0,
            "BASELINE_ARCH": baseline_for_modifier,
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
                run_result = bench.run(sug, n_trials=N_TRIALS + 1, seed=s, initial_arch=baseline_for_modifier)
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

    return all_runs, modifiers_list, baselines_list, mode_tag, results_dir

# -----------------------
# reports
# -----------------------
def write_all_reports(
    all_runs_by_label: Dict[str, Dict[str, List[RunLog]]],
    modifiers_list: List[Dict[int, List[float]]],
    baselines_list: List[List[int]],
    results_dir: Path,
    *,
    qps_min: float,
    mode_tag: str,
    competition_name: str = "10tries_60days_no_noise",
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
        flat_csv = report_dir / "results_baselines.csv"
    else:
        flat_csv = report_dir / f"results_baselines__{mode_tag}.csv"

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
    lines.append(f"[baseline] days={EVAL_TRAINING_DAYS}")
    lines.append("Baseline architectures (one per modifier):")
    for idx, baseline in enumerate(baselines_list, start=1):
        lines.append(f"  Modifier {idx}: {baseline}")

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
        baseline = baselines_list[idx - 1]
        # rebuild the model per label to ensure consistency
        cfg = {
            "GLOBAL_NOISE_SCALE": 0.0,
            "BASELINE_ARCH": baseline,
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
                        "competition": competition_name,
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
        columns=["competition","ModifiersLabel","Competitor","seed","step","Trial","Arch","score","qps","ne"]
    )
    df.to_csv(flat_csv, index=False)
    print(f"Saved: {flat_csv.resolve()}")

    # --- Best results per config (averaged over seeds) CSV ---
    # Generate filename for best results
    if mode_tag.startswith("hardQPS"):
        best_csv = report_dir / "results_best_per_config.csv"
    else:
        best_csv = report_dir / f"results_best_per_config__{mode_tag}.csv"

    # Collect best results for each (config, method) combination, aggregated over seeds
    from collections import defaultdict

    # Dictionary to collect scores per (config, method)
    config_method_scores: Dict[tuple, List[float]] = defaultdict(list)

    for idx, label in enumerate(labels, start=1):
        modifiers = modifiers_list[idx - 1]
        baseline = baselines_list[idx - 1]
        # rebuild the model per label to ensure consistency
        cfg = {
            "GLOBAL_NOISE_SCALE": 0.0,
            "BASELINE_ARCH": baseline,
            "BASELINE_DAY": EVAL_TRAINING_DAYS,
            "NETWORK_MODIFIERS": modifiers,
        }
        model_for_label = ModelPerformanceAPI(cfg)

        results = all_runs_by_label[label]
        for method, runs in results.items():
            for run in runs:
                # Find the best trial in this run (config+seed+method combination)
                best_trial = max(run.trials, key=lambda t: t.value)

                # Store the score for aggregation
                config_method_scores[(idx, label, method)].append(best_trial.value)

    # Now aggregate the results per config+method
    best_rows: List[Dict] = []
    for (config_id, label, method), scores in config_method_scores.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            best_rows.append({
                "competition": competition_name,
                "config_id": config_id,
                "ModifiersLabel": label,
                "Competitor": method,
                "num_seeds": len(scores),
                "avg_score": float(avg_score),
                "min_score": float(min_score),
                "max_score": float(max_score),
                "score_range": f"[{min_score:.6f}, {max_score:.6f}]",
            })

    best_df = pd.DataFrame(
        best_rows,
        columns=["competition","config_id","ModifiersLabel","Competitor","num_seeds","avg_score","min_score","max_score","score_range"]
    )
    best_df = best_df.sort_values(["config_id", "Competitor"])
    best_df.to_csv(best_csv, index=False)
    print(f"Saved best results per config (averaged over seeds): {best_csv.resolve()}")

    # --- Best results per config+seed CSV ---
    # Generate filename for per-seed results
    if mode_tag.startswith("hardQPS"):
        best_per_seed_csv = report_dir / "results_best_per_config_seed.csv"
    else:
        best_per_seed_csv = report_dir / f"results_best_per_config_seed__{mode_tag}.csv"

    # Collect best results for each (config, seed, method) combination
    best_per_seed_rows: List[Dict] = []

    for idx, label in enumerate(labels, start=1):
        modifiers = modifiers_list[idx - 1]
        baseline = baselines_list[idx - 1]
        # rebuild the model per label to ensure consistency
        cfg = {
            "GLOBAL_NOISE_SCALE": 0.0,
            "BASELINE_ARCH": baseline,
            "BASELINE_DAY": EVAL_TRAINING_DAYS,
            "NETWORK_MODIFIERS": modifiers,
        }
        model_for_label = ModelPerformanceAPI(cfg)

        results = all_runs_by_label[label]
        for method, runs in results.items():
            for run in runs:
                # Find the best trial in this run (config+seed+method combination)
                best_trial = max(run.trials, key=lambda t: t.value)

                # Get the seed from the trial
                seed = best_trial.seed

                # Recompute metrics for the best architecture
                arch = best_trial.config.get("arch")
                ne, qps, _ = model_for_label.train_model(
                    arch=arch,
                    training_days=EVAL_TRAINING_DAYS,
                    ignore_budget=True,
                )

                best_per_seed_rows.append({
                    "competition": competition_name,
                    "config_id": idx,
                    "ModifiersLabel": label,
                    "Competitor": method,
                    "seed": seed,
                    "best_score": float(best_trial.value),
                    "best_qps": float(qps),
                    "best_ne": float(ne),
                    "best_arch": json.dumps(arch),
                })

    best_per_seed_df = pd.DataFrame(
        best_per_seed_rows,
        columns=["competition","config_id","ModifiersLabel","Competitor","seed","best_score","best_qps","best_ne","best_arch"]
    )
    best_per_seed_df = best_per_seed_df.sort_values(["config_id", "Competitor", "seed"])
    best_per_seed_df.to_csv(best_per_seed_csv, index=False)
    print(f"Saved best results per config+seed: {best_per_seed_csv.resolve()}")

def run_from_folder(
    folder_path: str,
    *,
    qps_min: float = 3500.0,
    soft_qps: bool = False,
    soft_qps_tau: float = 0.15,
    fixed_depth: bool = True,
    use_pow2_widths: bool = False,
):
    """
    Run experiments from a folder containing configs.csv.
    The config file should have columns: config_id, baseline_architecture, network_modifier, num_trials, num_seeds, days
    """
    # Set global flags
    suggestors_module.FIXED_NUM_SUBARCHES_MODE = fixed_depth
    set_width_mode(use_pow2=use_pow2_widths)

    folder = Path(folder_path).resolve()
    config_csv = folder / "configs.csv"

    if not config_csv.exists():
        raise FileNotFoundError(f"Config file not found: {config_csv}")

    # Derive competition name from folder path
    # Try to find the project root (containing tasks/ directory)
    root_dir = folder
    while root_dir.parent != root_dir:  # Stop at filesystem root
        if (root_dir / "tasks").exists():
            break
        root_dir = root_dir.parent
    competition_name = _derive_competition_name(folder, root_dir)

    print(f"[run_from_folder] Loading config from: {config_csv}")
    print(f"[run_from_folder] Competition name: {competition_name}")

    # Load the full config including experiment parameters
    modifiers_list, baselines_list, num_trials, num_seeds, days, qps_min_from_config = load_full_config_csv(config_csv)

    # Use QPS from config if available, otherwise fall back to the passed parameter
    qps_min_to_use = qps_min_from_config

    print(f"[run_from_folder] Loaded {len(modifiers_list)} configurations")
    print(f"[run_from_folder] Parameters: num_trials={num_trials}, num_seeds={num_seeds}, days={days}, qps_min={qps_min_to_use}")

    # Determine which seeds to use
    if num_seeds <= len(SEEDS):
        seeds_to_use = SEEDS[:num_seeds]
    else:
        # If we need more seeds than defined in config, extend with sequential seeds
        seeds_to_use = SEEDS + list(range(len(SEEDS), num_seeds))

    print(f"[run_from_folder] Using {len(seeds_to_use)} seeds: {seeds_to_use}")

    # Create output directory structure
    mode_tag = _mode_tag(soft_qps=soft_qps, soft_qps_tau=soft_qps_tau, qps_min=qps_min_to_use)

    # Output goes in the same folder or a subdirectory
    results_dir = folder
    out_root = (folder / "results" / mode_tag).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Build suggestors ---
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
    # try:
    suggestors += [
        SkoptBOSuggestor("dummy", name="Skopt-Random"),
        SkoptBOSuggestor("GP",    name="Skopt-GP"),
        SkoptBOSuggestor("RF",    name="Skopt-RF"),
        SkoptBOSuggestor("ET",    name="Skopt-ET"),
        SkoptBOSuggestor("GBRT",  name="Skopt-GBRT"),
    ]
    # except Exception:
    #     print("[info] scikit-optimize not installed; skipping Skopt* baselines.")

    # --- Run experiments for each configuration ---
    all_runs: Dict[str, Dict[str, List[RunLog]]] = {}

    for idx, (modifiers, baseline) in enumerate(zip(modifiers_list, baselines_list), start=1):
        label = f"config_{idx:02d}"

        print(f"\n[run_from_folder] Running configuration {idx}/{len(modifiers_list)}: {label}")

        cfg = {
            "GLOBAL_NOISE_SCALE": 0.0,
            "BASELINE_ARCH": baseline,
            "BASELINE_DAY": days,
            "NETWORK_MODIFIERS": modifiers,
        }
        model_object = ModelPerformanceAPI(cfg)

        # Evaluator: set search-time QPS strategy
        evaluate_cfg = make_evaluator(
            model_object,
            qps_min=qps_min_to_use,
            soft_qps_tau=(soft_qps_tau if soft_qps else None),
        )

        bench = Benchmark(evaluate_cfg)

        results: Dict[str, List[RunLog]] = {}
        for sug in suggestors:
            results[sug.name] = []
            for s in seeds_to_use:
                # Run num_trials + 1 total: step 0 = baseline, steps 1-num_trials = additional trials
                run_result = bench.run(sug, n_trials=num_trials + 1, seed=s, initial_arch=baseline)
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

    # --- Write reports ---
    print("\n[run_from_folder] Generating reports...")
    write_all_reports(
        all_runs_by_label=all_runs,
        modifiers_list=modifiers_list,
        baselines_list=baselines_list,
        results_dir=results_dir,
        qps_min=qps_min_to_use,
        mode_tag=mode_tag,
        competition_name=competition_name,
    )

    print(f"\n[run_from_folder] All experiments complete!")
    print(f"[run_from_folder] Results saved to: {results_dir}")

# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Run HPO experiments with network modifiers.")
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder containing configs.csv to run experiments from.",
    )
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
    parser.add_argument(
        "--verbose-baseline-gen",
        action="store_true",
        help="Print verbose output during baseline generation (shows QPS values and attempts).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=5,
        help="Number of layers (subarches) to use in network modifiers (default: 5).",
    )
    parser.add_argument(
        "--allow-variable-subarches",
        action="store_true",
        help="Allow suggestors to search architectures with variable number of subarchitectures (1 to MAX_SUBARCHES). By default, all architectures have exactly MAX_SUBARCHES (5) subarchitectures with variable depth per subarch.",
    )
    parser.add_argument(
        "--use-pow2-widths",
        action="store_true",
        help="Use power-of-2 width ladder (e.g., 64, 128, 256, ..., 4096) instead of the default linear ladder with step size 64 (e.g., 64, 128, 192, ..., 4096).",
    )
    args = parser.parse_args()

    # If folder is specified, run from config file in that folder
    if args.folder:
        # Save the command that was run
        _save_command_file(Path(args.folder), args)

        run_from_folder(
            folder_path=args.folder,
            qps_min=args.qps_min,
            soft_qps=args.soft_qps,
            soft_qps_tau=args.soft_qps_tau,
            fixed_depth=not args.allow_variable_subarches,
            use_pow2_widths=args.use_pow2_widths,
        )
    else:
        # Original behavior
        all_runs, modifiers_list, baselines_list, mode_tag, results_dir = run_all(
            root_dir=args.root_dir,
            n_modifier_sets=args.n_modifiers,
            seed=args.seed,
            regenerate_modifiers=args.regenerate_modifiers,
            qps_min=args.qps_min,
            soft_qps=args.soft_qps,
            soft_qps_tau=args.soft_qps_tau,
            single_default_modifier=args.single_default_modifier,
            n_seeds=args.n_seeds,
            verbose_baseline_gen=args.verbose_baseline_gen,
            num_layers=args.num_layers,
            fixed_depth=not args.allow_variable_subarches,
            use_pow2_widths=args.use_pow2_widths,
        )

        write_all_reports(
            all_runs_by_label=all_runs,
            modifiers_list=modifiers_list,
            baselines_list=baselines_list,
            results_dir=results_dir,
            qps_min=args.qps_min,
            mode_tag=mode_tag,
        )

if __name__ == "__main__":
    main()
