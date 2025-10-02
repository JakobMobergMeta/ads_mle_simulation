import time, random
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

@dataclass
class Trial:
    config: Dict[str, Any]
    value: float
    elapsed: float
    seed: int

@dataclass
class RunLog:
    method: str
    trials: List[Trial]
    def best_curve(self) -> List[float]:
        best, out = -float("inf"), []
        for t in self.trials:
            best = max(best, t.value); out.append(best)
        return out

class Benchmark:
    def __init__(self, evaluate_fn: Callable[[Dict[str, Any]], float]):
        self.evaluate = evaluate_fn
    def run(self, suggestor, n_trials: int, seed: int, initial_arch=None) -> RunLog:
        rng = np.random.RandomState(seed); random.seed(seed)
        suggestor.reset(seed)
        trials, used, state = [], 0, None
        if initial_arch is not None:
            cfg0 = {"arch": initial_arch}
            t0 = time.time(); val0 = float(self.evaluate(cfg0)); dt0 = time.time() - t0
            suggestor.tell(cfg0, val0); trials.append(Trial(cfg0, val0, dt0, seed))
            state, used = cfg0, 1
        for _ in range(n_trials - used):
            cfg = suggestor.ask(rng, state)
            t0 = time.time(); val = float(self.evaluate(cfg)); dt = time.time() - t0
            suggestor.tell(cfg, val); trials.append(Trial(cfg, val, dt, seed))
            state = cfg
        return RunLog(method=getattr(suggestor, "name", "Suggestor"), trials=trials)
