# src/archopt/suggestors.py
import math, random
from typing import Any, Dict, Optional, List
import numpy as np

from .networks import (
    # use powers-of-two ladder
    sample_arch_pow2 as sample_arch,
    neighbor_arch_pow2 as neighbor_arch,
    validate_arch_nested,
    POW2_WIDTHS as WIDTH_LEVELS,  # list of allowed widths (2^n)
    MIN_SUBARCHES, MAX_SUBARCHES,
    MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB,
    # encoding helpers
    arch_to_params, params_to_arch, params_to_x, x_to_params,
)

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
            n_sub = trial.suggest_int("n_sub", MIN_SUBARCHES, MAX_SUBARCHES)
            for i in range(MAX_SUBARCHES):
                trial.suggest_int(f"depth_{i}", MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB)
                trial.suggest_int(f"widx_{i}", 0, len(WIDTH_LEVELS) - 1)
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

        # Search space: n_sub + per-sub depth + per-sub width-index (over POW2 ladder)
        self._space = [
            self._Integer(MIN_SUBARCHES, MAX_SUBARCHES, name="n_sub"),
            *[self._Integer(MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB, name=f"depth_{i}")
              for i in range(MAX_SUBARCHES)],
            *[self._Integer(0, len(WIDTH_LEVELS) - 1, name=f"widx_{i}")
              for i in range(MAX_SUBARCHES)],
        ]

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
            x = params_to_x(params)
        self.opt.tell(x, y)
