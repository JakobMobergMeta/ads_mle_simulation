# src/archopt/suggestors.py
import math, random
from typing import Any, Dict, Optional, List
import numpy as np

from .networks import (
    # sampling and neighbor functions
    sample_arch_pow2,
    sample_subarch_pow2,  # needed for LogSpaceSimulatedAnnealingSuggestor
    neighbor_arch_pow2,
    validate_arch_nested,
    # width ladder access
    get_width_levels,  # dynamically get current width ladder
    USE_POW2_WIDTHS,   # check current mode
    MIN_SUBARCHES, MAX_SUBARCHES,
    MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB,
    # encoding helpers
    arch_to_params, params_to_arch, params_to_x, x_to_params,
)

# Global flag to control architecture constraints
# When True: forces exactly MAX_SUBARCHES subarchitectures, allows variable depth per subarch
# When False: allows variable number of subarchitectures (1 to MAX_SUBARCHES), allows variable depth
FIXED_NUM_SUBARCHES_MODE = True

def sample_arch(rng: np.random.RandomState) -> List[List[int]]:
    """Wrapper that respects FIXED_NUM_SUBARCHES_MODE"""
    return sample_arch_pow2(rng, fixed_num_subarches=FIXED_NUM_SUBARCHES_MODE)

def neighbor_arch(
    rng: np.random.RandomState,
    arch: List[List[int]],
    *,
    k_max: int = 1,
    force_replace: bool = False,
) -> List[List[int]]:
    """Wrapper that respects FIXED_NUM_SUBARCHES_MODE"""
    return neighbor_arch_pow2(rng, arch, k_max=k_max, force_replace=force_replace, fixed_num_subarches=FIXED_NUM_SUBARCHES_MODE)

class Suggestor:
    name: str = "Suggestor"
    def reset(self, seed: int): ...
    def ask(self, rng: np.random.RandomState, state: Optional[Dict[str, Any]]): ...
    def tell(self, config: Dict[str, Any], value: float): ...

class RandomSuggestor(Suggestor):
    name = "Random"
    def reset(self, seed: int): random.seed(seed)
    def ask(self, rng, state):  # always sample powers-of-two architectures
        return {"arch": sample_arch(rng)}
    def tell(self, config, value): pass

class SimulatedAnnealingSuggestor(Suggestor):
    """
    Simulated annealing over nested architectures (powers-of-two widths) with:
      - occasional 'shake' move (replace one sub-arch)
      - multi-step width jumps (±k_max)
      - reheating when stalled
      - retries to avoid no-op neighbors
    """
    name = "SimulatedAnnealing"

    def __init__(
        self,
        T0: float = 0.8,
        alpha: float = 0.995,
        warmup: int = 2,
        max_retries: int = 16,
        k_max: int = 2,
        shake_prob: float = 0.15,
        stall_reheat_every: int = 6,
        reheat_factor: float = 1.5,
    ):
        self.T0 = T0
        self.alpha = alpha
        self.warmup = warmup
        self.max_retries = max_retries
        self.k_max = k_max
        self.shake_prob = shake_prob
        self.stall_reheat_every = stall_reheat_every
        self.reheat_factor = reheat_factor

        self._x: Optional[Dict[str, Any]] = None
        self._fx: Optional[float] = None
        self._best: Optional[Dict[str, Any]] = None
        self._fbest: float = -float("inf")
        self._T: float = T0
        self._n_evals: int = 0
        self._since_improve: int = 0

    def reset(self, seed: int):
        random.seed(seed)
        self._x = None
        self._fx = None
        self._best = None
        self._fbest = -float("inf")
        self._T = self.T0
        self._n_evals = 0
        self._since_improve = 0

    def _propose_neighbor(self, rng: np.random.RandomState, arch: List[List[int]]) -> List[List[int]]:
        # 1) shake (replace one sub-arch) with small prob
        if rng.rand() < self.shake_prob:
            neigh = neighbor_arch(rng, arch, k_max=self.k_max, force_replace=True)
            if neigh != arch:
                return neigh

        # 2) otherwise try local edits with retries to avoid no-op
        for _ in range(self.max_retries):
            neigh = neighbor_arch(rng, arch, k_max=self.k_max, force_replace=False)
            if neigh != arch:
                return neigh

        # 3) fallback: random different arch
        for _ in range(self.max_retries):
            alt = sample_arch(rng)
            if alt != arch:
                return alt
        return arch

    def ask(self, rng, state):
        if (self._x is None) or (self._n_evals < self.warmup):
            return state if state is not None else {"arch": sample_arch(rng)}
        arch_cur = self._x["arch"]
        arch_next = self._propose_neighbor(rng, arch_cur)
        try:
            validate_arch_nested(arch_next)
        except AssertionError:
            arch_next = sample_arch(rng)
        return {"arch": arch_next}

    def tell(self, config, value):
        self._n_evals += 1
        val = float(value)

        if self._fx is None:
            self._x, self._fx = config, val
            self._best, self._fbest = config, val
            self._since_improve = 0
            return

        # track global best
        if val > self._fbest:
            self._best, self._fbest = config, val
            self._since_improve = 0
        else:
            self._since_improve += 1

        # Metropolis accept (maximize)
        delta = val - self._fx
        accept = (delta >= 0) or (random.random() < math.exp(delta / max(self._T, 1e-12)))
        if accept:
            self._x, self._fx = config, val

        # reheat on stall
        if self._since_improve >= self.stall_reheat_every:
            self._T *= self.reheat_factor
            self._since_improve = 0

        # cool
        self._T *= self.alpha

class LogSpaceSimulatedAnnealingSuggestor(Suggestor):
    """
    Simulated annealing in log-space for uniform exploration of exponential grids.
    Works on log₂(width) values to make steps uniform, then maps back to powers-of-2.
    Note: This suggestor is designed for power-of-2 width ladders and may not work well with linear ladders.
    """
    name = "LogSpaceSimulatedAnnealing"

    def __init__(
        self,
        T0: float = 0.8,
        alpha: float = 0.995,
        warmup: int = 2,
        max_retries: int = 16,
        k_max: int = 1,  # smaller steps work better in log-space
        shake_prob: float = 0.15,
        stall_reheat_every: int = 6,
        reheat_factor: float = 1.5,
    ):
        self.T0 = T0
        self.alpha = alpha
        self.warmup = warmup
        self.max_retries = max_retries
        self.k_max = k_max
        self.shake_prob = shake_prob
        self.stall_reheat_every = stall_reheat_every
        self.reheat_factor = reheat_factor

        # Get width levels dynamically (will be set at reset time based on mode)
        self._width_levels = None
        self.log2_values = None
        self.min_log2 = None
        self.max_log2 = None

        self._x: Optional[Dict[str, Any]] = None
        self._fx: Optional[float] = None
        self._best: Optional[Dict[str, Any]] = None
        self._fbest: float = -float("inf")
        self._T: float = T0
        self._n_evals: int = 0
        self._since_improve: int = 0

    def reset(self, seed: int):
        random.seed(seed)
        # Update width levels from current mode
        self._width_levels = get_width_levels()
        # Compute log2 values (or use linear indices for linear mode)
        if USE_POW2_WIDTHS:
            self.log2_values = [int(math.log2(w)) for w in self._width_levels]
            self.min_log2 = min(self.log2_values)
            self.max_log2 = max(self.log2_values)
        else:
            # For linear mode, use indices directly
            self.log2_values = list(range(len(self._width_levels)))
            self.min_log2 = 0
            self.max_log2 = len(self._width_levels) - 1

        self._x = None
        self._fx = None
        self._best = None
        self._fbest = -float("inf")
        self._T = self.T0
        self._n_evals = 0
        self._since_improve = 0

    def _width_to_log2(self, width: int) -> int:
        """Convert width to log₂ value or index"""
        if USE_POW2_WIDTHS:
            return int(math.log2(width))
        else:
            # For linear mode, return the index in the width ladder
            try:
                return self._width_levels.index(width)
            except ValueError:
                # Snap to nearest
                return int(np.argmin([abs(width - w) for w in self._width_levels]))

    def _log2_to_width(self, log2_val: int) -> int:
        """Convert log₂ value or index to width, clamped to valid range"""
        log2_val = max(self.min_log2, min(self.max_log2, log2_val))
        if USE_POW2_WIDTHS:
            return 2 ** log2_val
        else:
            # For linear mode, use as direct index
            return self._width_levels[log2_val]

    def _propose_neighbor_logspace(self, rng: np.random.RandomState, arch: List[List[int]]) -> List[List[int]]:
        """Propose neighbor using uniform moves in log₂ space"""
        new_arch = [sub[:] for sub in arch]  # deep copy

        # 1) shake (replace one sub-arch) with small prob
        if rng.rand() < self.shake_prob:
            sub_idx = rng.randint(len(new_arch))
            new_arch[sub_idx] = sample_subarch_pow2(rng, fixed_depth=False)
            return new_arch

        # 2) uniform log-space moves with retries
        for _ in range(self.max_retries):
            # Choose random sub-arch and operation
            sub_idx = rng.randint(len(new_arch))
            sub = new_arch[sub_idx]

            if rng.rand() < 0.7:  # width change (70% prob)
                if len(sub) > 0:
                    current_width = sub[0]  # assuming uniform width per sub
                    current_log2 = self._width_to_log2(current_width)

                    # Uniform step in log₂ space
                    delta = rng.randint(-self.k_max, self.k_max + 1)
                    if delta != 0:  # ensure we make a change
                        new_log2 = current_log2 + delta
                        new_width = self._log2_to_width(new_log2)

                        if new_width != current_width:  # valid change
                            new_arch[sub_idx] = [new_width] * len(sub)
                            return new_arch

            else:  # depth change (30% prob)
                if rng.rand() < 0.5 and len(sub) < MAX_LAYERS_PER_SUB:
                    # add layer
                    new_arch[sub_idx] = sub + [sub[0]] if sub else [self._width_levels[rng.randint(len(self._width_levels))]]
                    return new_arch
                elif len(sub) > MIN_LAYERS_PER_SUB:
                    # remove layer
                    new_arch[sub_idx] = sub[:-1]
                    return new_arch

        # 3) fallback: random different arch
        for _ in range(self.max_retries):
            alt = sample_arch(rng)
            if alt != arch:
                return alt

        return arch

    def ask(self, rng, state):
        if (self._x is None) or (self._n_evals < self.warmup):
            return state if state is not None else {"arch": sample_arch(rng)}
        
        arch_cur = self._x["arch"]
        arch_next = self._propose_neighbor_logspace(rng, arch_cur)
        
        try:
            validate_arch_nested(arch_next)
        except AssertionError:
            arch_next = sample_arch(rng)
        
        return {"arch": arch_next}

    def tell(self, config, value):
        self._n_evals += 1
        val = float(value)

        if self._fx is None:
            self._x, self._fx = config, val
            self._best, self._fbest = config, val
            self._since_improve = 0
            return

        # track global best
        if val > self._fbest:
            self._best, self._fbest = config, val
            self._since_improve = 0
        else:
            self._since_improve += 1

        # Metropolis accept (maximize) - using log-space distance for better scaling
        delta = val - self._fx
        
        # Optional: scale acceptance by architectural distance in log-space
        arch_distance = self._compute_log_distance(config["arch"], self._x["arch"])
        scaled_delta = delta / (1 + 0.1 * arch_distance)  # dampening factor
        
        accept = (scaled_delta >= 0) or (random.random() < math.exp(scaled_delta / max(self._T, 1e-12)))
        if accept:
            self._x, self._fx = config, val

        # reheat on stall
        if self._since_improve >= self.stall_reheat_every:
            self._T *= self.reheat_factor
            self._since_improve = 0

        # cool
        self._T *= self.alpha

    def _compute_log_distance(self, arch1: List[List[int]], arch2: List[List[int]]) -> float:
        """Compute architectural distance in log₂ space"""
        if len(arch1) != len(arch2):
            return 10.0  # large penalty for different structure
        
        total_dist = 0.0
        for sub1, sub2 in zip(arch1, arch2):
            if len(sub1) != len(sub2):
                total_dist += 2.0  # penalty for different depths
            else:
                # Width distance in log₂ space
                if sub1 and sub2:
                    log1 = self._width_to_log2(sub1[0])
                    log2 = self._width_to_log2(sub2[0])
                    total_dist += abs(log1 - log2)
                
                # Depth difference
                total_dist += abs(len(sub1) - len(sub2))
        
        return total_dist

# ----- Optional TPE (Optuna) -----
try:
    import optuna  # type: ignore

    class TPESuggestor(Suggestor):
        name = "TPE"
        def __init__(self, initial_arch: Optional[List[List[int]]] = None):
            self._optuna = optuna
            self._study = None
            self._last_trial = None
            self._initial_arch = initial_arch

        def reset(self, seed: int):
            sampler = self._optuna.samplers.TPESampler(seed=seed)
            self._study = self._optuna.create_study(direction="maximize", sampler=sampler)
            self._last_trial = None
            if self._initial_arch is not None:
                try:
                    validate_arch_nested(self._initial_arch)
                    self._study.enqueue_trial(arch_to_params(self._initial_arch))
                except Exception:
                    pass

        def ask(self, rng: np.random.RandomState, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            trial = self._study.ask()
            width_levels = get_width_levels()
            if FIXED_NUM_SUBARCHES_MODE:
                # Force exactly MAX_SUBARCHES subarchitectures
                n_sub = MAX_SUBARCHES
            else:
                n_sub = trial.suggest_int("n_sub", MIN_SUBARCHES, MAX_SUBARCHES)
            for i in range(MAX_SUBARCHES):
                trial.suggest_int(f"depth_{i}", MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB)
                trial.suggest_int(f"widx_{i}", 0, len(width_levels) - 1)
            params = dict(trial.params)
            params["n_sub"] = n_sub
            cfg = {"arch": params_to_arch(params)}
            self._last_trial = trial
            return cfg

        def tell(self, config: Dict[str, Any], value: float):
            if self._last_trial is not None:
                self._study.tell(self._last_trial, float(value))
                self._last_trial = None

except Exception:
    # Safe stub so importing TPESuggestor doesn't explode when Optuna isn't installed.
    class TPESuggestor(Suggestor):  # type: ignore
        name = "TPE"
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Optuna not installed; TPESuggestor unavailable.")

# ---- scikit-optimize wrappers ----
class SkoptBOSuggestor(Suggestor):
    """ base can be one of: "GP", "RF", "ET", "GBRT", "dummy" """
    def __init__(self, base: str = "GP", name: Optional[str] = None):
        try:
            import skopt  # noqa: F401
            from skopt import Optimizer
            from skopt.space import Integer
        except Exception as e:
            raise RuntimeError("scikit-optimize not available: pip install scikit-optimize") from e

        self._skopt = __import__("skopt")
        self._Optimizer = self._skopt.Optimizer
        self._Integer = self._skopt.space.Integer
        self.base = base
        self.name = name or f"Skopt-{base}"
        self.opt = None
        self._last_x = None

        # Search space: n_sub + per-sub depth + per-sub width-index (over width ladder)
        # Get width levels at initialization time
        width_levels = get_width_levels()
        if FIXED_NUM_SUBARCHES_MODE:
            # When number of subarches is fixed, we don't need n_sub dimension in the search space
            # We'll set it to MAX_SUBARCHES when converting to architecture
            depth_dims = [self._Integer(MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB, name=f"depth_{i}")
                          for i in range(MAX_SUBARCHES)]
            self._space = [
                *depth_dims,
                *[self._Integer(0, len(width_levels) - 1, name=f"widx_{i}")
                  for i in range(MAX_SUBARCHES)],
            ]
            self._fixed_num_subarches = True
        else:
            depth_dims = [self._Integer(MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB, name=f"depth_{i}")
                          for i in range(MAX_SUBARCHES)]
            self._space = [
                self._Integer(MIN_SUBARCHES, MAX_SUBARCHES, name="n_sub"),
                *depth_dims,
                *[self._Integer(0, len(width_levels) - 1, name=f"widx_{i}")
                  for i in range(MAX_SUBARCHES)],
            ]
            self._fixed_num_subarches = False

    def reset(self, seed: int):
        self.opt = self._Optimizer(
            dimensions=self._space,
            base_estimator=self.base,
            random_state=seed,
            acq_func="EI",
        )
        self._last_x = None

    def ask(self, rng: np.random.RandomState, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        x = self.opt.ask()

        if self._fixed_num_subarches:
            # When in fixed num subarches mode, x only contains: [depth_0, ..., depth_N, widx_0, ..., widx_N]
            # We need to construct params with fixed n_sub
            params = {"n_sub": MAX_SUBARCHES}
            # Set depths from x (first MAX_SUBARCHES values)
            for i in range(MAX_SUBARCHES):
                params[f"depth_{i}"] = int(x[i])
            # Set width indices from x (next MAX_SUBARCHES values)
            for i in range(MAX_SUBARCHES):
                params[f"widx_{i}"] = int(x[MAX_SUBARCHES + i])
        else:
            # Normal mode: x contains [n_sub, depth_0, ..., depth_N, widx_0, ..., widx_N]
            params = x_to_params(x)

        cfg = {"arch": params_to_arch(params)}
        self._last_x = x
        return cfg

    def tell(self, config: Dict[str, Any], value: float):
        # skopt minimizes -> pass negative objective
        y = -float(value)
        if self._last_x is not None:
            x = self._last_x
            self._last_x = None
        else:
            # Encode the provided architecture back to x
            params = arch_to_params(config["arch"])
            if self._fixed_num_subarches:
                # When in fixed num subarches mode, x only contains: [depth_0, ..., depth_N, widx_0, ..., widx_N]
                x = []
                for i in range(MAX_SUBARCHES):
                    x.append(params[f"depth_{i}"])
                for i in range(MAX_SUBARCHES):
                    x.append(params[f"widx_{i}"])
            else:
                # Normal mode
                x = params_to_x(params)
        self.opt.tell(x, y)
