# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from admarket.ads_copilot.common.training_simulation.simple_interpolation import (
    create_simple_interpolator,
)


class BudgetExceededError(Exception):
    """Exception raised when a call would exceed the training budget."""

    pass


class CompositeKeyCache:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[Any, ...], Any] = {}

    def _create_composite_key(self, key1: Any, key2: Any) -> Tuple[Any, Any]:
        return (key1, key2)  # Using a tuple as the composite key

    def set(self, key1: Any, key2: Any, value: Any) -> None:
        composite_key = self._create_composite_key(key1, key2)
        self._cache[composite_key] = value

    def get(self, key1: Any, key2: Any) -> Optional[Any]:
        composite_key = self._create_composite_key(key1, key2)
        return self._cache.get(composite_key)

    def delete(self, key1: Any, key2: Any) -> None:
        composite_key = self._create_composite_key(key1, key2)
        if composite_key in self._cache:
            del self._cache[composite_key]


class ModelPerformanceAPI:
    """
    API for getting model performance with budget tracking.

    Tracks training budget using global variables and class variable for spent budget.
    Throws error if get_model_performance would exceed the budget.
    """

    model_config_dict: Dict[str, Any] = {}
    _call_counter: int = 0
    spent_budget: float = 0.0
    use_noise: bool = True

    def __init__(self, model_config_dict: Optional[Dict[str, Any]] = None) -> None:
        if model_config_dict is None:
            # Centralized model configuration dictionary
            model_config_dict = {
                # Shared performance parameters for consistent behavior across all graphs
                "MAX_TRAINING_DAYS": 60,  # Maximum training days (represents 100% training sufficiency)
                "SMALL_NETWORK": [128, 128],
                "MEDIUM_NETWORK": [1024, 1024],
                "LARGE_NETWORK": [2048, 2048],
                "INPUT_DIMENSIONS": 512,  # Input dimensionality
                "OUTPUT_DIMENSIONS": 10,  # Output dimensionality
                "GLOBAL_NOISE_SCALE": 0.02,  # Global noise scale used consistently across all plots
                "BASE_MODEL_FLOPS": 1e8,  # Base model complexity: 100M FLOP/sample (reduced for more sensitivity)
                "FLOPS_PER_PARAMETER": 100.0,  # Additional FLOP/sample per parameter (increased for more impact)
                "MACHINE_EFFICIENCY": 5e12,  # Machine efficiency: 5T FLOP/s (adjusted to maintain baseline ~50K QPS)
                # Noise configuration
                "NOISE_TRAINING_FACTOR_MIN": 0.01,  # Minimum noise factor at high training
                "NOISE_TRAINING_FACTOR_MAX": 0.7,  # Additional noise factor at low training
                "NOISE_PARAM_FACTOR": 1.0,  # Parameter noise factor
                # Performance bounds
                "MIN_PERFORMANCE": 0.0,  # Minimum performance value
                "MAX_PERFORMANCE": 1.0,  # Maximum performance value
                "MIN_QPS": 500,  # Minimum QPS value
                "RANDOM_STATE": None,
                "GPUS_PER_DAY": 8,
                "TOTAL_BUDGET_GPU_DAYS": 8000,
                "WAIT": False,
                "WAIT_SECONDS_PER_TRAINING_DAY": 1,
            }

        # Set computed values
        model_config_dict["LARGE_PARAMS"] = self.arch_to_params(
            model_config_dict["LARGE_NETWORK"]
        )
        model_config_dict["MEDIUM_PARAMS"] = self.arch_to_params(
            model_config_dict["MEDIUM_NETWORK"]
        )
        model_config_dict["SMALL_PARAMS"] = self.arch_to_params(
            model_config_dict["SMALL_NETWORK"]
        )
        model_config_dict["SATURATION_POINT"] = model_config_dict["LARGE_PARAMS"]
        # Parameters in millions where benefits start to saturate

        self.model_config_dict = model_config_dict

        # Global call counter for unique seeding
        self._call_counter = 0
        # variable for budget tracking
        self.spent_budget = 0.0

        self.model_cache = CompositeKeyCache()

    def get_default_model_config_dict(self) -> Dict[str, Any]:
        """Get the default model configuration dictionary."""
        return self.model_config_dict.copy()

    def arch_to_params(
        self, arch: Any, input_size: int = 512, output_size: int = 10
    ) -> float:
        """
        Convert architecture specification to parameter count in millions.

        Args:
            arch: List of layer sizes (e.g., [512, 512] for two 512-node layers)
            input_size: Input size N (dimension of input vector)
            output_size: Output size K (dimension of output vector)

        Returns:
            float: Parameter count in millions
        """
        if not isinstance(arch, (list, tuple)) or len(arch) == 0:
            raise ValueError(
                "Architecture must be a non-empty list or tuple of layer sizes"
            )

        total_params = 0
        prev_layer_size = input_size
        end_to_end_network = list(arch) + [output_size]
        for layer_size in end_to_end_network:
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

    def get_remaining_budget(self, total_budget: Any = None) -> float:
        """Get the remaining budget."""
        return self.model_config_dict["TOTAL_BUDGET_GPU_DAYS"] - self.spent_budget

    def get_spent_budget(self) -> float:
        """Get the amount of budget already spent."""
        return self.spent_budget

    def reset_budget(self) -> None:
        """Reset the budget to 0."""
        self.spent_budget = 0.0

    def calculate_cost(self, training_days: float, gpus_per_day: Any = None) -> float:
        """
        Calculate the cost of a get_model_performance call.

        Args:
            training_days: Number of training days
            gpus_per_day: GPUs per training day

        Returns:
            Cost as gpus_per_day * training_days
        """
        if gpus_per_day is None:
            gpus_per_day = self.model_config_dict["GPUS_PER_DAY"]
        return gpus_per_day * training_days

    def _check_budget_constraints(
        self,
        training_days: int,
        gpus_per_day: float,
        total_budget: float,
        ignore_budget: bool,
    ) -> float:
        """Check budget constraints and return the calculated cost."""
        cost = self.calculate_cost(training_days, gpus_per_day)
        if not ignore_budget:
            if self.spent_budget + cost > total_budget:
                remaining = total_budget - self.spent_budget
                raise BudgetExceededError(
                    f"Call would cost {cost} but only {remaining} budget remaining. "
                    f"Total budget: {total_budget}, Already spent: {self.spent_budget}"
                )
        return cost

    def _determine_params_count(self, arch: Any, params_millions: Any) -> float:
        """Determine the parameter count from architecture or direct value."""
        if arch is not None:
            return self.arch_to_params(arch)
        elif params_millions is None:
            raise ValueError("Either 'params_millions' or 'arch' must be provided")
        return params_millions

    def _calculate_base_performance(
        self, params_millions: float, training_sufficiency: float
    ) -> float:
        """Calculate base training performance using interpolation."""
        interpolator = create_simple_interpolator(
            small_params=self.model_config_dict["SMALL_PARAMS"],
            medium_params=self.model_config_dict["MEDIUM_PARAMS"],
            large_params=self.model_config_dict["LARGE_PARAMS"],
            saturation_point=self.model_config_dict["SATURATION_POINT"],
        )
        return interpolator.interpolate_1d(
            params_millions=params_millions,
            training_sufficiency=training_sufficiency,
        )

    def _calculate_qps(self, params_millions: float) -> float:
        """Calculate QPS using computational complexity approach."""
        arch_params = params_millions * 1e6  # Convert millions to units
        model_complexity = (
            self.model_config_dict["BASE_MODEL_FLOPS"]
            + arch_params * self.model_config_dict["FLOPS_PER_PARAMETER"]
        )

        # QPS = Machine Efficiency / Complexity (samples/s)
        qps = self.model_config_dict["MACHINE_EFFICIENCY"] / model_complexity
        return max(qps, self.model_config_dict["MIN_QPS"])

    def _create_random_state(self) -> np.random.RandomState:
        """Create a random state for noise generation."""
        if self.model_config_dict["RANDOM_STATE"] is not None:
            return np.random.RandomState(self.model_config_dict["RANDOM_STATE"])
        else:
            # Use time + call counter + random component for unique seed each call
            import random

            self._call_counter += 1

            random_component = random.randint(0, 10000)
            unique_seed = (
                int(time.time() * 1000000 + self._call_counter + random_component)
                % 2**32
            )
            return np.random.RandomState(unique_seed)

    def _apply_noise_to_metrics(
        self,
        training_ne: float,
        qps: float,
        training_sufficiency: float,
        final_noise_scale: float,
    ) -> tuple[float, float]:
        """Apply noise to performance metrics."""
        rng = self._create_random_state()

        # Calculate noise scaling factors
        training_factor = self.model_config_dict[
            "NOISE_TRAINING_FACTOR_MIN"
        ] + self.model_config_dict["NOISE_TRAINING_FACTOR_MAX"] * (
            1.0 - training_sufficiency
        )

        current_noise_scale = final_noise_scale * training_factor

        # Add noise to NE metrics
        training_ne_noise = rng.normal(0, current_noise_scale)

        training_ne_noisy = np.clip(
            training_ne + training_ne_noise,
            self.model_config_dict["MIN_PERFORMANCE"],
            self.model_config_dict["MAX_PERFORMANCE"],
        )

        # Add noise to QPS
        qps_noise_scale = qps * current_noise_scale
        qps_noise = rng.normal(0, qps_noise_scale)
        qps_noisy = max(qps + qps_noise, self.model_config_dict["MIN_QPS"])

        return training_ne_noisy, qps_noisy

    def train_model(
        self,
        training_days: int,
        params_millions: Any = None,
        arch: Any = None,
        ignore_budget: bool = False,
    ) -> tuple[float, float, Dict[str, float]]:
        # Check budget constraints
        cost = self._check_budget_constraints(
            training_days,
            self.model_config_dict["GPUS_PER_DAY"],
            self.model_config_dict["TOTAL_BUDGET_GPU_DAYS"],
            ignore_budget,
        )

        # Determine parameter count
        params_millions = self._determine_params_count(arch, params_millions)

        # Optional wait simulation
        if self.model_config_dict["WAIT"]:
            time.sleep(
                self.model_config_dict["WAIT_SECONDS_PER_TRAINING_DAY"]
                * training_days
                * np.random.uniform(0, 2)
            )

        curve = {}
        training_ne = 0.0
        qps = 0.0

        for i in range(1, training_days + 1):
            if self.model_cache.get(i, params_millions) is not None:
                training_ne, qps = self.model_cache.get(i, params_millions)
            else:
                results = self._get_model_performance(i, params_millions)
                training_ne = results["training_ne"]
                qps = results["qps"]
                self.model_cache.set(i, params_millions, (training_ne, qps))

            curve[i] = training_ne

        # Record the expense after successful completion (unless ignoring budget)
        if not ignore_budget:
            self.spent_budget += cost

        return training_ne, qps, curve

    def _get_model_performance(
        self,
        training_days: int,
        params_millions: Any = None,
    ) -> Dict[str, float]:
        """
        Get model performance with optional budget tracking.

        Args:
            training_days: Number of training days
            params_millions: Model parameter count in millions
            gpus_per_day: GPUs per training day
            total_budget: Total budget in GPU days

        Returns:
            dict: Performance metrics

        Raises:
            BudgetExceededError: If the call would exceed the available budget (unless ignore_budget=True)
        """

        # Convert training days to training sufficiency
        training_sufficiency = np.clip(
            training_days / self.model_config_dict["MAX_TRAINING_DAYS"], 0.0, 1.0
        )

        # Calculate performance metrics
        training_ne = self._calculate_base_performance(
            params_millions, training_sufficiency
        )
        qps = self._calculate_qps(params_millions)

        if self.model_config_dict["GLOBAL_NOISE_SCALE"] > 0.0:
            training_ne, qps = self._apply_noise_to_metrics(
                training_ne,
                qps,
                training_sufficiency,
                self.model_config_dict["GLOBAL_NOISE_SCALE"],
            )

        return {
            "training_ne": float(training_ne),
            "qps": float(qps),
        }
