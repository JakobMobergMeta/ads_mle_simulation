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
python scripts/run_experiments.py
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