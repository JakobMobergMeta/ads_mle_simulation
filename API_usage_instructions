# Model Performance API Guide

## Overview

The `ModelPerformanceAPI` simulates machine learning model training with budget tracking and performance estimation. It provides realistic metrics for normalized error (NE) and queries per second (QPS) based on model architecture and training duration. It also returns the learning_curve for analysis of convergence and trendlines



## Quick Start

```python
from admarket.ads_copilot.common.training_simulation.model_performance_api import ModelPerformanceAPI

# Initialize API instance with default configuration
api = ModelPerformanceAPI()

# Train a model for 30 days with 1M parameters
training_ne, qps, learning_curve = api.train_model(
    training_days=30,
    params_millions=1.0
)

print(f"Final NE: {training_ne:.4f}, QPS: {qps:.0f}")
```
## Configuration

Default configuration includes:
- Total budget: 8,000 GPU days
- Max training days: 60
- GPUs per day: 8
- Global noise scale: 0.02 


-To get no-noise model you need to set global_noise_scale = 0 using the steps below
Override defaults by copying the config and creating a new API instance

```python
# Get default configuration
api = ModelPerformanceAPI()
default_config = api.get_default_model_config_dict()

# Modify the copied config
custom_config = default_config
custom_config.update({
    "TOTAL_BUDGET_GPU_DAYS": 5000,
    "GPUS_PER_DAY": 16,
    "GLOBAL_NOISE_SCALE":0,
    "INPUT_DIMENSIONS": 256,
    "OUTPUT_DIMENSIONS": 3
})

# Create new API instance with custom config
api_custom = ModelPerformanceAPI(model_config_dict=custom_config)
```


## Core Functions

### `train_model(training_days, params_millions=None, arch=None, ignore_budget=False)`

Trains a model and returns performance metrics.

**Parameters:**

in general use arch parameter over params_millions and ignore_budget = True unless explicitly told othewewise

- `training_days` (int): Number of training days (1-60)
- `arch` (list, optional): Architecture as layer sizes (e.g., `[512, 512]`)
- `ignore_budget` (bool): Skip budget validation

**Returns:**
- `training_ne` (float): Final normalized error (0.0-1.0)
- `qps` (float): Queries per second
- `learning_curve` (dict): Training progress by day

### `arch_to_params(arch, input_size=512, output_size=10)`

Converts architecture specification to parameter count.
Rememeber to consider if you need to update the config or use default input/output dimensions

**Parameters:**
- `arch` (list): Layer sizes (e.g., `[128, 256, 128]`)

**Returns:**
- Parameter count in millions

### Budget Management

```python
# Check remaining budget
remaining = api.get_remaining_budget()

# Calculate cost before training
cost = api.calculate_cost(training_days=10, gpus_per_day=8)

# Reset budget
api.reset_budget()
```

## Architecture Examples

```python
# Predefined architectures
small_arch = [128, 128]      # ~0.07M parameters
medium_arch = [1024, 1024]   # ~1.6M parameters
large_arch = [2048, 2048]    # ~6.3M parameters

# Train with architecture
ne, qps, curve = api.train_model(training_days=20, arch=medium_arch)

# Train with direct parameter count
ne, qps, curve = api.train_model(training_days=20, params_millions=2.5)
```


## Error Handling

```python
from admarket.ads_copilot.common.training_simulation.model_performance_api import BudgetExceededError

try:
    ne, qps, curve = api.train_model(training_days=50, params_millions=10.0)
except BudgetExceededError as e:
    print(f"Budget exceeded: {e}")
    # Handle budget constraint
```

## Key Features

- **Budget Tracking**: Automatic GPU-day budget management
- **Realistic Curves**: Performance improves with training time and model size
- **Noise Simulation**: Configurable noise for realistic variation
- **Caching**: Built-in caching for repeated configurations
- **Flexible Input**: Support for both architecture specs and direct parameter counts
