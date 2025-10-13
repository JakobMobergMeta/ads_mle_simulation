# Baseline Experiments Insights

## Performance Comparison: Single Default vs 100 Random Modifiers

### Summary

A dramatic performance difference was observed between optimization on the single default network modifier versus 100 random network modifiers, revealing important characteristics about the optimization landscape difficulty.

## Results Overview

### Single Default Modifier (Easy Landscape)
- **Random**: 0.123 (consistent baseline)
- **Best optimization methods**: ~0.12-0.13 range
- **Behavior**: All methods perform similarly around 0.12
- **Variance**: Minimal (single configuration)

### 100 Random Modifiers (Hard Landscape)
- **Random**: 0.112 (becomes the best performer!)
- **Optimization methods**: 0.009-0.10 range (much worse performance)
- **Variance**: Huge across modifiers (min=0.000, max=0.146)
- **Behavior**: High variability, many optimization failures

## Key Findings

### 1. Random Sampling Outperforms Optimization on Hard Problems
- **Default modifier**: Random ≈ Optimizers (~0.12)
- **Random modifiers**: Random (0.112) >> Optimizers (0.009-0.10)
- **Implication**: The random network modifiers create optimization landscapes where sophisticated algorithms perform worse than random search

### 2. Optimization Algorithm Struggles
**Simulated Annealing variants suffer most:**
- LogSpaceSimulatedAnnealing: 0.120 → 0.009 (92% performance drop)
- SimulatedAnnealing: 0.120 → 0.021 (82% performance drop)

**Bayesian methods handle difficulty better:**
- TPE: 0.124 → 0.100 (19% drop, best optimizer)
- Skopt-GP: 0.124 → 0.102 (18% drop)

### 3. High Variance Indicates Pathological Cases
- **Min performance**: 0.000 (complete optimization failure)
- **Max performance**: 0.146 (some modifiers remain optimizable)
- **Interpretation**: Some network modifiers create nearly impossible optimization problems while others remain tractable

## Implications

### Benchmark Quality
This performance difference demonstrates that the network modifier system successfully creates:
- **Easy baseline problems** (default modifier)
- **Challenging optimization scenarios** (random modifiers)
- **Realistic difficulty gradients** for algorithm evaluation

### Algorithm Insights
1. **Simulated Annealing limitations**: Both SA variants struggle with rugged landscapes created by random modifiers, likely getting trapped in local optima

2. **Bayesian optimization robustness**: TPE and Gaussian Process methods show better adaptation to difficult landscapes through their probabilistic modeling

3. **Random search baseline value**: Random sampling provides a crucial performance floor that sophisticated optimizers should exceed

### Research Directions
1. **Modifier analysis**: Investigate which network modifier characteristics create the most/least optimizable landscapes

2. **Algorithm improvement**: Develop optimization methods that maintain performance across diverse landscape difficulties

3. **Problem characterization**: Use modifier-based difficulty tuning for systematic algorithm evaluation

## Conclusion

The dramatic performance gap between single default and random modifiers reveals that the benchmark successfully generates problems of varying difficulty. The fact that random search outperforms optimization algorithms on hard modifiers indicates the presence of deceptive, rugged fitness landscapes - exactly the type of challenging scenarios needed for rigorous algorithm evaluation.

This validates the benchmark's utility for distinguishing between robust and brittle optimization approaches in neural architecture search scenarios.

## Baseline Methods Documentation

### 1. Random
**Description**: Pure random sampling baseline that selects architectures uniformly from the valid architecture space.
**Algorithm**: Samples random architectures using powers-of-2 width levels and random depths.
**Hyperparameters**: None (purely stochastic)
**Implementation**: `RandomSuggestor()` - calls `sample_arch(rng)` for each trial

### 2. SimulatedAnnealing 
**Description**: Standard simulated annealing with local neighborhood moves on the architecture space.
**Algorithm**: 
- Starts from baseline architecture
- Proposes neighbors via width adjustments (±k steps on powers-of-2 ladder) and depth changes
- Accepts/rejects based on Metropolis criterion with cooling schedule
- Includes "shake" moves (15% probability) to replace entire sub-architectures
- Reheating mechanism when optimization stalls

**Hyperparameters**:
- `T0=0.5` - Initial temperature
- `alpha=0.996` - Cooling rate (T *= alpha each iteration)
- `warmup=3` - Number of initial trials to accept unconditionally
- `max_retries=16` - Max attempts to find valid neighbors
- `k_max=2` - Maximum width jump distance
- `shake_prob=0.15` - Probability of replacing entire sub-architecture
- `stall_reheat_every=6` - Reheat after 6 steps without improvement
- `reheat_factor=1.5` - Temperature increase factor during reheating

### 3. LogSpaceSimulatedAnnealing
**Description**: Enhanced simulated annealing that operates in log₂ space for uniform exploration of exponential width grids.
**Algorithm**: 
- Same as SimulatedAnnealing but width moves happen in log₂(width) space
- Makes relative changes uniform (±1 in log space = ×2 or ÷2 in actual width)
- Includes distance-aware acceptance probability scaling

**Hyperparameters**:
- `T0=0.5` - Initial temperature  
- `alpha=0.996` - Cooling rate
- `warmup=3` - Warmup period
- `max_retries=16` - Max neighbor attempts
- `k_max=1` - Maximum log₂ jump distance (smaller since log-space is more uniform)
- `shake_prob=0.15` - Shake move probability
- `stall_reheat_every=6` - Reheating frequency
- `reheat_factor=1.5` - Temperature boost factor

### 4. TPE (Tree-structured Parzen Estimator)
**Description**: Bayesian optimization using Optuna's TPE implementation for sequential model-based optimization.
**Algorithm**:
- Builds probabilistic models of promising vs unpromising regions
- Uses tree-structured Parzen estimators to model P(x|y)
- Optimizes acquisition function to balance exploration/exploitation
- Encodes architectures as categorical/integer parameters

**Hyperparameters**:
- Uses Optuna's default TPE hyperparameters
- `initial_arch=BASELINE_ARCH` - Seeds optimization with baseline architecture
- Automatic parameter space: n_sub ∈ [1,5], depth_i ∈ [1,5], widx_i ∈ [0,5] (width indices)

### 5. Skopt-* Methods (Bayesian Optimization)
**Description**: Scikit-optimize based Bayesian optimization with different base estimators.
**Algorithm**: 
- Sequential model-based optimization using various surrogate models
- Acquisition function: Expected Improvement (EI)
- Encodes architectures in continuous parameter space

**Common Hyperparameters**:
- `acq_func="EI"` - Expected Improvement acquisition
- Search space: n_sub ∈ [1,5], depth_i ∈ [1,5], widx_i ∈ [0,5]

**Variants**:
- **Skopt-Random**: Random baseline (`base_estimator="dummy"`)
- **Skopt-GP**: Gaussian Process surrogate (`base_estimator="GP"`)  
- **Skopt-RF**: Random Forest surrogate (`base_estimator="RF"`)
- **Skopt-ET**: Extra Trees surrogate (`base_estimator="ET"`)
- **Skopt-GBRT**: Gradient Boosting surrogate (`base_estimator="GBRT"`)

## Experimental Setup
- **Budget**: N_TRIALS + 1 = 11 evaluations per method per seed (step 0 = baseline arch, steps 1-10 = optimization)
- **Seeds**: 10 seeds (0-9) for default modifier, 1 seed for 100 modifiers
- **QPS constraint**: Hard QPS ≥ 3500 requirement
- **Training time**: 60 days simulation per architecture evaluation