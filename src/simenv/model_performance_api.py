# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .simple_interpolation import (
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

    DEFAULT_CONFIG: Dict[str, Any] = {
        "MAX_TRAINING_DAYS": 60,  # Maximum training days (represents 100% training sufficiency)
        "SMALL_NETWORK": [[64]],
        "MEDIUM_NETWORK": [
            [64, 64],
            [512, 512],
            [1024, 1024],
            [1024, 1024],
            [1024, 1024],
        ],
        "LARGE_NETWORK": [
            [4096, 4096, 4096, 4096],
            [4096, 4096, 4096, 4096],
            [4096, 4096, 4096, 4096],
            [4096, 4096, 4096, 4096],
            [4096, 4096, 4096, 4096],
        ],
        "INPUT_DIMENSIONS": 512,  # Input dimensionality
        "OUTPUT_DIMENSIONS": 10,  # Output dimensionality
        "GLOBAL_NOISE_SCALE": 0.02,  # Global noise scale used consistently across all plots
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
        #        "BASELINE_NE": 0.559786328135504,
        "BASELINE_ARCH": [[1024, 1024], [512, 512], [1024, 1024], [1024], [1024]],
        "BASELINE_DAY": 60,
        "MIN_SUBARCHES": 1,  # Minimum number of sub-architectures
        "DEPTH_RANGE": [1, 5],
        "FLAX_RANGE": [64, 4096],
        "NETWORK_MODIFIERS": {
            0: [0.05, 0.23, 1, 0.1, 0.1],
            1: [0.05, 0.2, 1, 0.1, 0.1],
            2: [0.8, 0.01, 1, 0.1, 0.1],
            3: [0.05, 0.23, 1, 0.1, 0.1],
            4: [0.05, 0.23, 1, 0.1, 0.1],
            #           0: [0.2, 0.2, 1, 0.1, 0.1],
            #           1: [0.4, 0.2, 0.7, 0, 0.05],
            #           2: [0.1, 0.1, 0.6, 0.4, 0.05],
            #           3: [0.2, 0.4, 0.7, 0.3, 0.05],
            #           4: [0.1, 0.1, 0.9, 0.1, 0.1],
            # format
            # { network_idx:
            #   [
            #       ne_contrib_weight_idx,
            #       qps_contrib_weight_idx,
            #       flops_scale_idx,
            #       flax_scale_idx,
            #       depth_scale_idx
            #   ]
            # }
        },
        "MACHINE_EFFICIENCY": 2.5e12,
        "QPS_MODEL_MIN_FLOPS": 5e7,  # 2.5e12 / 50000 = 5e7 for max QPS of 50000
        "QPS_MODEL_MAX_FLOPS": 5e9,  # 2.5e12 / 500 = 5e9 for min QPS of 500
    }

    def post_process_config_update(self) -> None:
        self.model_config_dict["MIN_ARCHITECTURE"] = self.get_min_arch()
        self.model_config_dict["MAX_ARCHITECTURE"] = self.get_max_arch()

        self.model_config_dict["MIN_COMPLEXITIES"] = self._arch_to_complexities(
            self.model_config_dict["MIN_ARCHITECTURE"]
        )

        self.model_config_dict["MAX_COMPLEXITIES"] = self._arch_to_complexities(
            self.model_config_dict["MAX_ARCHITECTURE"]
        )

        self.model_config_dict["SMALL_COMPLEXITIES_SCALED"] = (
            self.arch_to_scaled_complexities(self.model_config_dict["SMALL_NETWORK"])
        )
        self.model_config_dict["MEDIUM_COMPLEXITIES_SCALED"] = (
            self.arch_to_scaled_complexities(self.model_config_dict["MEDIUM_NETWORK"])
        )

        self.model_config_dict["LARGE_COMPLEXITIES_SCALED"] = (
            self.arch_to_scaled_complexities(self.model_config_dict["LARGE_NETWORK"])
        )
        # Global call counter for unique seeding
        self._call_counter = 0
        # variable for budget tracking
        self.spent_budget = 0.0

        self.model_cache = CompositeKeyCache()
        self.model_config_dict["BASELINE_NE"] = self._get_model_performance(
            self.model_config_dict["BASELINE_DAY"],
            self.model_config_dict["BASELINE_ARCH"],
        )["training_ne"]

    def __init__(self, model_config_dict: Optional[Dict[str, Any]] = None) -> None:
        cfg = dict(
            self.DEFAULT_CONFIG
        )  # shallow copy is enough since values are primitives/lists
        if model_config_dict:
            cfg.update(model_config_dict)
        model_config_dict = cfg

        arches_range = [1, len(model_config_dict["NETWORK_MODIFIERS"])]
        model_config_dict["ARCHES_RANGE"] = arches_range

        self.model_config_dict = model_config_dict
        self.post_process_config_update()

    def get_delta(self, x_val: float, min_val: float, max_val: float) -> float:
        return (x_val - min_val) / (max_val - min_val)

    def get_delta_depth(self, depth: int) -> float:
        return self.get_delta(
            depth,
            self.model_config_dict["DEPTH_RANGE"][0],
            self.model_config_dict["DEPTH_RANGE"][1],
        )

    def get_delta_flax(self, flax: int) -> float:
        return self.get_delta(
            flax,
            self.model_config_dict["FLAX_RANGE"][0],
            self.model_config_dict["FLAX_RANGE"][1],
        )

    def get_delta_nr_subarches(self, nr_subarches: int) -> float:
        return self.get_delta(
            nr_subarches,
            self.model_config_dict["ARCHES_RANGE"][0],
            self.model_config_dict["FLAX_RANGEARCHES_RANGE"][1],
        )

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
            # Ensure layer_size is an integer (handle nested structures)
            if isinstance(layer_size, (list, tuple)):
                raise ValueError(
                    f"Expected integer layer size, got {type(layer_size)}: {layer_size}. "
                    f"Architecture should be a flat list of integers, not nested."
                )

            if not isinstance(layer_size, int) or layer_size <= 0:
                raise ValueError(
                    f"Layer size must be a positive integer, got: {layer_size}"
                )

            # Each layer: weights (prev_size * current_size) + biases (current_size)
            layer_params = prev_layer_size * layer_size + layer_size
            total_params += layer_params
            prev_layer_size = layer_size

        # Convert to millions
        return total_params / 1e6

    def arch_to_scaled_complexities(
        self, arch: Any, input_size: int = 512, output_size: int = 10
    ) -> Tuple[float, float]:
        ne_complexity, qps_complexity = self._arch_to_complexities(
            arch, input_size, output_size
        )
        min_complexities = self.model_config_dict["MIN_COMPLEXITIES"]
        max_complexities = self.model_config_dict["MAX_COMPLEXITIES"]

        #  print("min_comp")
        #  print(min_complexities)
        #  print("max_comp")
        #  print(max_complexities)

        scaled_ne_complexity = self.get_delta(
            ne_complexity, min_complexities[0], max_complexities[0]
        )
        scaled_qps_complexity = self.get_delta(
            qps_complexity, min_complexities[1], max_complexities[1]
        )
        return scaled_ne_complexity, scaled_qps_complexity

    def _arch_to_complexities(
        self, arch: Any, input_size: int = 512, output_size: int = 10
    ) -> Tuple[float, float]:
        """
        Convert architecture specification to parameter count in millions.

        Args:
            arch: List of layer sizes (e.g., [512, 512] for two 512-node layers)
            input_size: Input size N (dimension of input vector)
            output_size: Output size K (dimension of output vector)

        Returns:
            float:"""

        # ne complexity
        ne_complexity = 0.0
        # qps complexity
        qps_complexity = 0.0

        for i in range(len(arch)):
            sub_input_size = arch[i - 1][-1] if i > 0 else input_size
            sub_output_size = arch[i + 1][0] if i < len(arch) - 1 else output_size

            sub_arch = arch[i]
            lambda_ne_i, lambda_qps_i, lambda_flops_i, lambda_flax_i, lambda_depth_i = (
                self.model_config_dict["NETWORK_MODIFIERS"][i]
            )

            flops = self.arch_to_params(sub_arch, sub_input_size, sub_output_size)

            depth = len(sub_arch)
            flax = min(sub_arch)

            min_sub_arch = self.model_config_dict["MIN_ARCHITECTURE"][0]
            max_sub_arch = self.model_config_dict["MAX_ARCHITECTURE"][i]

            min_flops = self.arch_to_params(
                min_sub_arch, sub_input_size, sub_output_size
            )
            max_flops = self.arch_to_params(
                max_sub_arch, sub_input_size, sub_output_size
            )

            delta_depth = self.get_delta_depth(depth)
            delta_flax = self.get_delta_flax(flax)
            delta_flops = self.get_delta(flops, min_flops, max_flops)

            ne_complexity += lambda_ne_i * (
                delta_flops * lambda_flops_i
                + delta_flax * lambda_flax_i
                + delta_depth * lambda_depth_i
            )
            qps_complexity += lambda_qps_i * (delta_flops)
        #     print("loop step")
        #     print(delta_depth, delta_flax, delta_flops)
        #     print(arch, ne_complexity, qps_complexity)
        return ne_complexity, qps_complexity

    def get_min_arch(self) -> List[List[int]]:
        min_dim = self.model_config_dict["DEPTH_RANGE"][0]
        min_layer = self.model_config_dict["FLAX_RANGE"][0]
        min_subarchs = self.model_config_dict["MIN_SUBARCHES"]
        min_arch = []
        for _ in range(min_subarchs):
            min_arch.append([min_layer] * min_dim)
        return min_arch

    def get_max_arch(self) -> List[List[int]]:
        max_dim = self.model_config_dict["DEPTH_RANGE"][1]
        max_layer = self.model_config_dict["FLAX_RANGE"][1]
        max_subarchs = len(self.model_config_dict["NETWORK_MODIFIERS"])
        max_arch = []
        for _ in range(max_subarchs):
            max_arch.append([max_layer] * max_dim)
        return max_arch

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

    def _calculate_base_performance(
        self,
        complexity_scaled: float,
        training_sufficiency: float,
    ) -> float:
        """Calculate base training performance using interpolation."""

        interpolator = create_simple_interpolator(
            small_complexity=self.model_config_dict["SMALL_COMPLEXITIES_SCALED"][0],
            medium_complexity=self.model_config_dict["MEDIUM_COMPLEXITIES_SCALED"][0],
            large_complexity=self.model_config_dict["LARGE_COMPLEXITIES_SCALED"][0],
        )
        interpolated_ne = interpolator.interpolate_1d(
            complexity=complexity_scaled,
            training_sufficiency=training_sufficiency,
        )

        return interpolated_ne

    def _calculate_qps(self, qps_complexity_scaled: float) -> float:
        #        """Calculate QPS using computational complexity approach."""
        # min_complexity_qps = self.model_config_dict["MIN_COMPLEXITIES"][1]
        # max_complexity_qps = self.model_config_dict["MAX_COMPLEXITIES"][1]

        # delta_qps_flops = self.get_delta(
        #     qps_complexity, min_complexity_qps, max_complexity_qps
        # )
        model_complexity = self.model_config_dict[
            "QPS_MODEL_MIN_FLOPS"
        ] + qps_complexity_scaled * (
            self.model_config_dict["QPS_MODEL_MAX_FLOPS"]
            - self.model_config_dict["QPS_MODEL_MIN_FLOPS"]
        )

        # QPS = Machine Efficiency / Complexity (samples/s)
        qps = self.model_config_dict["MACHINE_EFFICIENCY"] / model_complexity
        return max(qps, self.model_config_dict["MIN_QPS"])

        # QPS = Machine Efficiency / Complexity (samples/s)

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
        arch: List[List[int]],
        ignore_budget: bool = False,
    ) -> Tuple[float, float, Dict[int, Dict[str, float]]]:
        # Check budget constraints
        cost = self._check_budget_constraints(
            training_days,
            self.model_config_dict["GPUS_PER_DAY"],
            self.model_config_dict["TOTAL_BUDGET_GPU_DAYS"],
            ignore_budget,
        )
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

        arch_str = str(arch)  # Use arch as cache key
        for i in range(1, training_days + 1):
            if self.model_cache.get(i, arch_str) is not None:
                training_ne, qps = self.model_cache.get(i, arch_str)
            else:
                # Use arch for the actual performance calculation
                results = self._get_model_performance(i, arch)
                training_ne = results["training_ne"]
                qps = results["qps"]
                qps = max(qps, self.model_config_dict["MIN_QPS"])

                self.model_cache.set(i, arch_str, (training_ne, qps))

            curve[i] = {"ne": training_ne, "qps": qps}

        # Record the expense after successful completion (unless ignoring budget)
        if not ignore_budget:
            self.spent_budget += cost

        return training_ne, qps, curve

    def _get_model_performance(
        self, training_days: int, arch: List[List[int]]
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

        ne_complexity, qps_complexity = self.arch_to_scaled_complexities(arch)

        training_ne = 0.0
        qps = 0.0

        training_ne = self._calculate_base_performance(
            ne_complexity, training_sufficiency
        )
        qps = self._calculate_qps(qps_complexity)

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

    def get_score(self, ne: float) -> float:
        """
        Get the score for a given architecture.

        Args:
            arch: Architecture configuration (defaults to [64, 64] if None)

        Returns:
            float: Score based on normalized error reduction from baseline
        """
        baseline_ne = self.model_config_dict["BASELINE_NE"]
        return (baseline_ne - ne) / baseline_ne
