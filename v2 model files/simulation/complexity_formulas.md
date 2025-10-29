# Model Performance Complexity Formulas

This document describes the mathematical formulas used for calculating model complexity metrics in the training simulation system.

## Overview

The model performance API uses two primary complexity functions to evaluate neural network architectures:
- `complexity_ne`: Complexity metric for Normalized Entropy (training performance)
- `complexity_qps`: Complexity metric for Queries Per Second (inference throughput)

## Configuration Constants

The following parameters are **constants defined in the config file** (`DEFAULT_CONFIG`):

### Core System Parameters
- **`MAX_TRAINING_DAYS`** = 60 (maximum training days representing 100% sufficiency)
- **`INPUT_DIMENSIONS`** = 512 (input vector dimensionality)
- **`OUTPUT_DIMENSIONS`** = 10 (output vector dimensionality)
- **`MACHINE_EFFICIENCY`** = 2.5 × 10¹² operations/second
- **`MIN_QPS`** = 500 (minimum queries per second)
- **`QPS_MODEL_MIN_FLOPS`** = 5 × 10⁷ (minimum computational complexity)
- **`QPS_MODEL_MAX_FLOPS`** = 5 × 10⁹ (maximum computational complexity)

### Architecture Range Limits
- **`DEPTH_RANGE`** = [1, 5] (min/max number of layers)
- **`FLAX_RANGE`** = [64, 2048] (min/max layer width)
- **`MIN_SUBARCHES`** = 1 (minimum sub-architectures)

### Reference Architectures
- **`SMALL_NETWORK`** = [[64]]
- **`MEDIUM_NETWORK`** = [[64, 64], [512, 512], [1024, 1024], [1024, 1024], [1024, 1024]]
- **`LARGE_NETWORK`** = [[2048, 2048], [2048, 2048], [2048, 2048], [2048, 2048], [2048, 2048]]

### Noise Parameters
- **`GLOBAL_NOISE_SCALE`** = 0.02 (base noise scale)
- **`NOISE_TRAINING_FACTOR_MIN`** = 0.01 (minimum noise at high training)
- **`NOISE_TRAINING_FACTOR_MAX`** = 0.7 (additional noise at low training)
- **`MIN_PERFORMANCE`** = 0.0 (performance bounds)
- **`MAX_PERFORMANCE`** = 1.0 (performance bounds)

### Network Modifiers (Config Constant)
**`NETWORK_MODIFIERS`** dictionary defines scaling factors for each sub-architecture:

| Index | λ_ne | λ_qps | λ_flops | λ_flax | λ_depth |
|-------|------|-------|---------|--------|---------|
| 0     | 0.2  | 0.2   | 1.0     | 0.1    | 0.1     |
| 1     | 0.4  | 0.2   | 0.7     | 0.0    | 0.05    |
| 2     | 0.1  | 0.1   | 0.6     | 0.4    | 0.05    |
| 3     | 0.2  | 0.4   | 0.7     | 0.3    | 0.05    |
| 4     | 0.1  | 0.1   | 0.9     | 0.1    | 0.1     |

## Function Definitions

### QPS Complexity Function

```
qps(complexity_qps) = machine_efficiency / model_complexity
```

Where:
- `complexity_qps := λ_qps_i × Δ_flops`
- `machine_efficiency` = **`MACHINE_EFFICIENCY`** = 2.5 × 10¹² operations/second *(config constant)*
- `model_complexity` = min_flops + complexity_qps × (max_flops - min_flops)
- `min_flops` = **`QPS_MODEL_MIN_FLOPS`** = 5 × 10⁷ *(config constant)*
- `max_flops` = **`QPS_MODEL_MAX_FLOPS`** = 5 × 10⁹ *(config constant)*

#### QPS Formula Components:
- **λ_qps_i**: QPS contribution weight for sub-architecture i (from NETWORK_MODIFIERS)
- **Δ_flops**: Normalized FLOPS delta = (flops - min_flops) / (max_flops - min_flops)

### Normalized Entropy Complexity Function

```
ne(complexity_ne) = interpolate(complexity_ne, training_sufficiency)
```

Where:
- `complexity_ne := Σᵢ λ_ne_i × (Δ_flops × λ_flops_i + Δ_flax × λ_flax_i + Δ_depth × λ_depth_i)`

#### NE Complexity Formula Components:
- **λ_ne_i**: NE contribution weight for sub-architecture i
- **Δ_flops**: Normalized FLOPS delta = (flops - min_flops) / (max_flops - min_flops)
- **Δ_flax**: Normalized layer width delta = (flax - min_flax) / (max_flax - min_flax)
- **Δ_depth**: Normalized depth delta = (depth - min_depth) / (max_depth - min_depth)
- **λ_flops_i, λ_flax_i, λ_depth_i**: Scaling factors for FLOPS, layer width, and depth

## Parameter Calculations

### Architecture to Parameters
```
params_millions = total_params / 10⁶

total_params = Σ (prev_layer_size × current_layer_size + current_layer_size)
```

### FLOPS Calculation
The FLOPS (Floating Point Operations Per Second) for each sub-architecture:
```
flops = arch_to_params(sub_arch, sub_input_size, sub_output_size)
```

### Architectural Metrics
- **Depth**: `depth = len(sub_arch)` (number of layers)
- **Flax**: `flax = min(sub_arch)` (minimum layer width)

## Scaling and Normalization

### Complexity Scaling
Both complexity metrics are scaled to [0,1] range:
```
scaled_complexity = (complexity - min_complexity) / (max_complexity - min_complexity)
```

### Training Sufficiency
```
training_sufficiency = min(training_days / MAX_TRAINING_DAYS, 1.0)
```
Where **`MAX_TRAINING_DAYS`** = 60 *(config constant)*

## Interpolation Functions

The NE complexity uses interpolation between three reference curves:

### Small Model Curve
```
small_curve(t) = 0.62 - 0.04 / (1 + exp(-10 × (t - 0.2)))
```


### Medium Model Curve
```
medium_curve(t) = 0.65 - 0.16 / (1 + exp(-8 × (t - 0.3)))
```


### Large Model Curve
```
large_curve(t) = 0.68 - 0.20 / (1 + exp(-8 × (t - 0.3)))
```



## Noise Application

When noise is enabled, both metrics receive stochastic perturbations:

### Training Factor
```
training_factor = noise_min + noise_max × (1 - training_sufficiency)
```
- `noise_min` = **`NOISE_TRAINING_FACTOR_MIN`** = 0.01 *(config constant)*
- `noise_max` = **`NOISE_TRAINING_FACTOR_MAX`** = 0.7 *(config constant)*

### Noise Scaling
```
current_noise_scale = global_noise_scale × training_factor
```
Where `global_noise_scale` = **`GLOBAL_NOISE_SCALE`** = 0.02 *(config constant)*

### NE Noise
```
ne_noisy = clip(ne + N(0, current_noise_scale), MIN_PERFORMANCE, MAX_PERFORMANCE)
```
Where bounds are **`MIN_PERFORMANCE`** = 0.0 and **`MAX_PERFORMANCE`** = 1.0 *(config constants)*

### QPS Noise
```
qps_noise_scale = qps × current_noise_scale
qps_noisy = max(qps + N(0, qps_noise_scale), min_qps)
```
Where `min_qps` = **`MIN_QPS`** = 500 *(config constant)*


## Key Relationships

1. **Higher complexity_qps** → Lower QPS (more computation required)
2. **Higher complexity_ne** → Different training dynamics (via interpolation)
3. **More training days** → Better performance (up to plateaus for smaller models)
4. **Larger architectures** → Higher complexity but potentially better final performance
