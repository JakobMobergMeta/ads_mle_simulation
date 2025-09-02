# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import time
from typing import Any, Dict

import numpy as np

from admarket.ads_copilot.common.training_simulation.simple_interpolation import (
    create_simple_interpolator,
)

# Global call counter for unique seeding
_call_counter = 0

# Centralized model configuration dictionary
model_config_dict: Dict[str, Any] = {
    # Shared performance parameters for consistent behavior across all graphs
    "MAX_TRAINING_DAYS": 60,  # Maximum training days (represents 100% training sufficiency)
    "SMALL_PARAMS": 0.089,  # millions
    "MEDIUM_PARAMS": 1.256,  # millions
    "LARGE_PARAMS": 4.41,  # millions
    "GLOBAL_NOISE_SCALE": 0.02,  # Global noise scale used consistently across all plots
    # Data and penalty configuration
    "DATA_AMOUNT_GB": 10.0,  # Amount of training data assumed in GB
    "OPTIMAL_DATA_PARAM_RATIO": 20.0,  # Optimal data per parameter ratio
    "DATA_PENALTY_FACTOR": 0.005,  # Factor for data penalty calculation
    "BASE_GENERALIZATION_GAP": 0.008,  # Base generalization gap
    "GENERALIZATION_GAP_FACTOR": 0.002,  # Generalization gap factor per parameter
    # QPS calculation configuration - Based on computational complexity
    "BASE_MODEL_FLOPS": 1e8,  # Base model complexity: 100M FLOP/sample (reduced for more sensitivity)
    "FLOPS_PER_PARAMETER": 500.0,  # Additional FLOP/sample per parameter (increased for more impact)
    "MACHINE_EFFICIENCY": 5e12,  # Machine efficiency: 5T FLOP/s (adjusted to maintain baseline ~50K QPS)
    "MIN_QPS_FLOOR": 3500,  # Minimum QPS floor for very large models
    # Noise configuration
    "NOISE_TRAINING_FACTOR_MIN": 0.3,  # Minimum noise factor at high training
    "NOISE_TRAINING_FACTOR_MAX": 0.7,  # Additional noise factor at low training
    "NOISE_PARAM_FACTOR": 1.0,  # Uniform noise factor across parameters
    "QPS_NOISE_RELATIVE_SCALE": 0.1,  # QPS noise as percentage of QPS value
    # Performance bounds
    "MIN_PERFORMANCE": 0.0,  # Minimum performance value
    "MAX_PERFORMANCE": 1.0,  # Maximum performance value
    "MIN_QPS": 0.0,  # Minimum QPS value
    "RANDOM_STATE": None,
    "SECONDS_PER_TRAINING_DAY": 1,
    # Default resource configuration
    "DEFAULT_GPUS_PER_DAY": 8,
    "DEFAULT_TOTAL_BUDGET_GPU_DAYS": 8000,
}

# Set computed values
model_config_dict["SATURATION_POINT"] = model_config_dict[
    "LARGE_PARAMS"
]  # Parameters in millions where benefits start to saturate
model_config_dict["ADD_NOICE"] = model_config_dict["GLOBAL_NOISE_SCALE"] > 0.0


class BudgetExceededError(Exception):
    """Exception raised when a call would exceed the training budget."""

    pass


class ModelPerformanceAPI:
    """
    API for getting model performance with budget tracking.

    Tracks training budget using global variables and class variable for spent budget.
    Throws error if get_model_performance would exceed the budget.
    """

    # Class variable for budget tracking
    spent_budget = 0.0

    @classmethod
    def arch_to_params(cls, arch: Any) -> float:
        """
        Convert architecture specification to parameter count in millions.

        Args:
            arch: List of layer sizes (e.g., [512, 512] for two 512-node layers)

        Returns:
            float: Parameter count in millions
        """
        if not isinstance(arch, (list, tuple)) or len(arch) == 0:
            raise ValueError(
                "Architecture must be a non-empty list or tuple of layer sizes"
            )

        total_params = 0
        prev_layer_size = arch[0]  # Assuming input size for first layer

        for layer_size in arch:
            # Each layer: weights (prev_size * current_size) + biases (current_size)
            if (
                total_params == 0
            ):  # First layer - assume input size equals first layer size
                layer_params = layer_size * layer_size + layer_size
            else:
                layer_params = prev_layer_size * layer_size + layer_size

            total_params += layer_params
            prev_layer_size = layer_size

        # Convert to millions
        return total_params / 1e6

    @classmethod
    def get_remaining_budget(cls, total_budget: Any = None) -> float:
        """Get the remaining budget."""
        if total_budget is None:
            total_budget = model_config_dict["DEFAULT_TOTAL_BUDGET_GPU_DAYS"]
        return total_budget - cls.spent_budget

    @classmethod
    def get_spent_budget(cls) -> float:
        """Get the amount of budget already spent."""
        return cls.spent_budget

    @classmethod
    def calculate_cost(cls, training_days: float, gpus_per_day: Any = None) -> float:
        """
        Calculate the cost of a get_model_performance call.

        Args:
            training_days: Number of training days
            gpus_per_day: GPUs per training day (default from config)

        Returns:
            Cost as gpus_per_day * training_days
        """
        if gpus_per_day is None:
            gpus_per_day = model_config_dict["DEFAULT_GPUS_PER_DAY"]
        return gpus_per_day * training_days

    @classmethod
    def _validate_and_set_defaults(
        cls, gpus_per_day: Any, total_budget: Any
    ) -> tuple[Any, Any]:
        """Validate inputs and set default values from config."""
        if gpus_per_day is None:
            gpus_per_day = model_config_dict["DEFAULT_GPUS_PER_DAY"]
        if total_budget is None:
            total_budget = model_config_dict["DEFAULT_TOTAL_BUDGET_GPU_DAYS"]
        return gpus_per_day, total_budget

    @classmethod
    def _check_budget_constraints(
        cls,
        training_days: int,
        gpus_per_day: Any,
        total_budget: Any,
        ignore_budget: bool,
    ) -> float:
        """Check budget constraints and return the calculated cost."""
        cost = cls.calculate_cost(training_days, gpus_per_day)
        if not ignore_budget:
            if cls.spent_budget + cost > total_budget:
                remaining = total_budget - cls.spent_budget
                raise BudgetExceededError(
                    f"Call would cost {cost} but only {remaining} budget remaining. "
                    f"Total budget: {total_budget}, Already spent: {cls.spent_budget}"
                )
        return cost

    @classmethod
    def _determine_params_count(cls, arch: Any, params_millions: Any) -> float:
        """Determine the parameter count from architecture or direct value."""
        if arch is not None:
            return cls.arch_to_params(arch)
        elif params_millions is None:
            raise ValueError("Either 'params_millions' or 'arch' must be provided")
        return params_millions

    @classmethod
    def _calculate_base_performance(
        cls, params_millions: float, training_sufficiency: float
    ) -> float:
        """Calculate base training performance using interpolation."""
        interpolator = create_simple_interpolator()
        return interpolator.interpolate_1d(
            params_millions=params_millions,
            training_sufficiency=training_sufficiency,
        )

    @classmethod
    def _calculate_eval_penalties(
        cls, params_millions: float, training_ne: float
    ) -> float:
        """Calculate eval NE by adding data and generalization penalties."""
        penalty_params = (
            model_config_dict["SATURATION_POINT"]
            if params_millions >= model_config_dict["SATURATION_POINT"]
            else params_millions
        )

        data_per_param = model_config_dict["DATA_AMOUNT_GB"] / max(penalty_params, 0.01)
        data_sufficiency = np.tanh(
            data_per_param / model_config_dict["OPTIMAL_DATA_PARAM_RATIO"]
        )
        data_penalty = (
            (1.0 - data_sufficiency)
            * penalty_params
            * model_config_dict["DATA_PENALTY_FACTOR"]
        )
        gen_gap = (
            model_config_dict["BASE_GENERALIZATION_GAP"]
            + penalty_params * model_config_dict["GENERALIZATION_GAP_FACTOR"]
        )

        return training_ne + data_penalty + gen_gap

    @classmethod
    def _calculate_qps(cls, params_millions: float) -> float:
        """Calculate QPS using computational complexity approach."""
        arch_params = params_millions * 1e6  # Convert millions to units
        model_complexity = (
            model_config_dict["BASE_MODEL_FLOPS"]
            + arch_params * model_config_dict["FLOPS_PER_PARAMETER"]
        )

        # QPS = Machine Efficiency / Complexity (samples/s)
        qps = model_config_dict["MACHINE_EFFICIENCY"] / model_complexity
        return max(qps, model_config_dict["MIN_QPS_FLOOR"])

    @classmethod
    def _create_random_state(cls) -> np.random.RandomState:
        """Create a random state for noise generation."""
        if model_config_dict["RANDOM_STATE"] is not None:
            return np.random.RandomState(model_config_dict["RANDOM_STATE"])
        else:
            # Use time + call counter + random component for unique seed each call
            import random

            global _call_counter
            _call_counter += 1

            random_component = random.randint(0, 10000)
            unique_seed = (
                int(time.time() * 1000000 + _call_counter + random_component) % 2**32
            )
            return np.random.RandomState(unique_seed)

    @classmethod
    def _apply_noise_to_metrics(
        cls,
        training_ne: float,
        eval_ne: float,
        qps: float,
        training_sufficiency: float,
        final_noise_scale: float,
    ) -> tuple[float, float, float]:
        """Apply noise to performance metrics."""
        rng = cls._create_random_state()

        # Calculate noise scaling factors
        training_factor = model_config_dict[
            "NOISE_TRAINING_FACTOR_MIN"
        ] + model_config_dict["NOISE_TRAINING_FACTOR_MAX"] * (
            1.0 - training_sufficiency
        )
        param_factor = model_config_dict["NOISE_PARAM_FACTOR"]
        current_noise_scale = final_noise_scale * training_factor * param_factor

        # Add noise to NE metrics
        training_ne_noise = rng.normal(0, current_noise_scale)
        eval_ne_noise = rng.normal(0, current_noise_scale)

        training_ne_noisy = np.clip(
            training_ne + training_ne_noise,
            model_config_dict["MIN_PERFORMANCE"],
            model_config_dict["MAX_PERFORMANCE"],
        )
        eval_ne_noisy = np.clip(
            eval_ne + eval_ne_noise,
            model_config_dict["MIN_PERFORMANCE"],
            model_config_dict["MAX_PERFORMANCE"],
        )

        # Add noise to QPS
        qps_noise_scale = qps * eval_ne_noisy * current_noise_scale
        qps_noise = rng.normal(0, qps_noise_scale)
        qps_noisy = max(qps + qps_noise, model_config_dict["MIN_QPS"])

        return training_ne_noisy, eval_ne_noisy, qps_noisy

    @classmethod
    def get_model_performance(
        cls,
        training_days: int,
        arch: Any = None,
        params_millions: Any = None,
        WAIT: bool = False,
        override_noise: Any = None,
        ignore_budget: bool = False,
        gpus_per_day: Any = None,
        total_budget: Any = None,
    ) -> Dict[str, float]:
        """
        Get model performance with optional budget tracking.

        Args:
            training_days: Number of training days
            arch: Architecture specification as list of layer sizes
            params_millions: Model parameter count in millions
            WAIT: Whether to wait during execution
            override_noise: Noise override settings
            ignore_budget: If True, bypass budget tracking (for visualization)
            gpus_per_day: GPUs per training day (default from config)
            total_budget: Total budget in GPU days (default from config)

        Returns:
            dict: Performance metrics

        Raises:
            BudgetExceededError: If the call would exceed the available budget (unless ignore_budget=True)
        """
        # Validate inputs and set defaults
        gpus_per_day, total_budget = cls._validate_and_set_defaults(
            gpus_per_day, total_budget
        )

        # Check budget constraints
        cost = cls._check_budget_constraints(
            training_days, gpus_per_day, total_budget, ignore_budget
        )

        # Determine parameter count
        params_millions = cls._determine_params_count(arch, params_millions)

        # Optional wait simulation
        if WAIT:
            time.sleep(
                model_config_dict["SECONDS_PER_TRAINING_DAY"]
                * training_days
                * np.random.uniform(0, 2)
            )

        # Convert training days to training sufficiency
        training_sufficiency = np.clip(
            training_days / model_config_dict["MAX_TRAINING_DAYS"], 0.0, 1.0
        )

        # Calculate performance metrics
        training_ne = cls._calculate_base_performance(
            params_millions, training_sufficiency
        )
        eval_ne = cls._calculate_eval_penalties(params_millions, training_ne)
        qps = cls._calculate_qps(params_millions)

        # Apply noise if requested
        final_noise_scale = (
            model_config_dict["GLOBAL_NOISE_SCALE"]
            if override_noise is None
            else float(override_noise)
        )

        if final_noise_scale > 0.0:
            training_ne, eval_ne, qps = cls._apply_noise_to_metrics(
                training_ne, eval_ne, qps, training_sufficiency, final_noise_scale
            )

        # Record the expense after successful completion (unless ignoring budget)
        if not ignore_budget:
            cls.spent_budget += cost

        return {
            "training_ne": float(training_ne),
            "eval_ne": float(eval_ne),
            "qps": float(qps),
            "gpu_days": float(cost),
        }
