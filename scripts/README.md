# Experiment Scripts

This directory contains scripts for running neural architecture optimization experiments.

## Main Script: `run_experiments.py`

The primary script for running hyperparameter optimization (HPO) experiments with different network configurations.

### Prerequisites

Install optional dependencies for all optimization methods:
```bash
pip install -e ".[hpo]"
```

## Experiment Types

### 1. Multiple Network Modifiers Experiment

**Command:**
```bash
python scripts/run_experiments.py --n-seeds 1 --regenerate-modifiers
```

**What it does:**
- Generates 100 random network modifier sets (by default)
- Each modifier set creates different network performance characteristics
- Runs optimization methods on each modifier set
- Uses 10 seeds per method per modifier

**Output Location:**
```
tasks/10 tries @60days no noise/baselines_results_100modifiers/
```

**Files Generated:**
- `modifiers_60days_10tries_100modifiers.csv` - The 100 network modifier configurations
- `results_baselines.txt` - Summary results (average/min/max across all modifiers)
- `results_details_baselines.txt` - Detailed step-by-step results for all runs
- `Sim_ML_Bench_l3_60days_10tries_baselines.csv` - Complete CSV data for analysis

### 2. Single Default Modifier Experiment

**Command:**
```bash
python scripts/run_experiments.py --single-default-modifier
```

**What it does:**
- Uses only the default network configuration (no modifications)
- Runs optimization methods with 10 different random seeds
- Focuses on comparing optimization algorithm performance on the baseline network

**Output Location:**
```
tasks/10 tries @60days no noise/baselines_results_default_modifier/
```

**Files Generated:**
- `results_baselines.txt` - Summary results across 10 seeds
- `results_details_baselines.txt` - Detailed step-by-step results for all seeds
- `Sim_ML_Bench_l3_60days_10tries_baselines.csv` - Complete CSV data for analysis

## Optimization Methods Tested

Both experiments test these optimization algorithms:
- **Random** - Random architecture sampling
- **SimulatedAnnealing** - Standard simulated annealing
- **LogSpaceSimulatedAnnealing** - Improved SA with uniform log-space moves
- **TPE** - Tree-structured Parzen Estimator (if Optuna installed)
- **Skopt-*** - Bayesian optimization methods (if scikit-optimize installed):
  - Skopt-Random, Skopt-GP, Skopt-RF, Skopt-ET, Skopt-GBRT

## Width Search Space Configuration

The script supports two different width ladder modes for architecture search:

### 1. **Linear Step Size 64 (Default)**

By default, the script searches over widths with **equal step sizes of 64** from 64 to 4096:
```
64, 128, 192, 256, 320, 384, ..., 3968, 4032, 4096
```
This provides **64 discrete width levels** for fine-grained optimization.

**Example command:**
```bash
# Default mode - uses linear step size 64
python scripts/run_experiments.py --folder /path/to/experiment
```

### 2. **Power-of-2 Widths**

Use the `--use-pow2-widths` flag to search over **power-of-2 widths**:
```
64, 128, 256, 512, 1024, 2048, 4096
```
This provides **7 discrete width levels** with exponential spacing.

**Example command:**
```bash
# Power-of-2 mode
python scripts/run_experiments.py --folder /path/to/experiment --use-pow2-widths
```

### Running Experiments from a Folder

When you have a pre-configured experiment folder with a `configs.csv` file:

**Linear step size 64 (default):**
```bash
python scripts/run_experiments.py --folder /path/to/experiment/folder
```

**Power-of-2 widths:**
```bash
python scripts/run_experiments.py --folder /path/to/experiment/folder --use-pow2-widths
```

**With additional options:**
```bash
python scripts/run_experiments.py \
    --folder /path/to/experiment/folder \
    --use-pow2-widths \
    --qps-min 4000 \
    --soft-qps
```

**Command History:** Each run automatically saves a `command.sh` file in the experiment folder containing the exact command used, making experiments easy to reproduce.

## Additional Options

### Soft QPS Mode
Add `--soft-qps` for experiments with soft QPS constraints:
```bash
python scripts/run_experiments.py --soft-qps --soft-qps-tau 0.15
```
Results will include mode tag in filenames (e.g., `results_baselines__softQPS_tau0.15_Q3500.txt`)

### Custom Parameters
```bash
python scripts/run_experiments.py --n-modifiers 50 --qps-min 4000
```

## Results Structure

Each experiment creates:
1. **Summary files** - High-level performance comparisons
2. **Detailed logs** - Step-by-step optimization traces  
3. **CSV data** - Complete structured data for further analysis
4. **Modifier configurations** - Network modification parameters (for multi-modifier experiments)

The CSV files contain trial-by-trial data including:
- Architecture configurations
- Performance scores
- QPS (queries per second) values
- Network efficiency (NE) metrics

## Command-Line Reference

### All Available Flags

```bash
python scripts/run_experiments.py [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--folder` | string | None | Path to folder containing `configs.csv` for pre-configured experiments |
| `--root-dir` | string | `.` | Root directory for tasks and outputs |
| `--n-modifiers` | int | 100 | Number of random modifier sets to generate |
| `--seed` | int | 0 | Seed for generating random modifiers |
| `--regenerate-modifiers` | flag | False | Force regeneration of modifiers even if file exists |
| `--qps-min` | float | 3500.0 | Minimum QPS requirement |
| `--soft-qps` | flag | False | Use soft QPS penalty instead of hard rejection |
| `--soft-qps-tau` | float | 0.15 | Softness parameter for soft QPS penalty |
| `--single-default-modifier` | flag | False | Use single default modifier instead of multiple random ones |
| `--n-seeds` | int | None | Number of seeds to use (overrides config default) |
| `--verbose-baseline-gen` | flag | False | Print verbose output during baseline generation |
| `--num-layers` | int | 5 | Number of layers (subarches) in network modifiers |
| `--allow-variable-subarches` | flag | False | Allow variable number of subarchitectures (1 to MAX) |
| `--use-pow2-widths` | flag | False | Use power-of-2 widths instead of linear step 64 |

### Quick Examples

**Standard run with default settings (linear step 64):**
```bash
python scripts/run_experiments.py
```

**Run from folder with power-of-2 widths:**
```bash
python scripts/run_experiments.py --folder tasks/my_experiment --use-pow2-widths
```

**Generate 50 modifiers with 5 seeds each:**
```bash
python scripts/run_experiments.py --n-modifiers 50 --n-seeds 5
```

**Soft QPS mode with custom threshold:**
```bash
python scripts/run_experiments.py --soft-qps --qps-min 4000 --soft-qps-tau 0.2
```