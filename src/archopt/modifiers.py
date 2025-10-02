# src/archopt/modifiers.py
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional


# ----------------------------
# Helpers
# ----------------------------
def _random_partition_on_grid(
    rng: random.Random,
    n: int,
    step: float = 0.05,
    positive: bool = True,
) -> List[float]:
    """
    Return n parts (each a multiple of `step`) summing to exactly 1.0.

    If positive=True, each part is >= step (strictly positive).
    If positive=False, parts are >= 0 but still multiples of step.

    Implementation: sample a random composition of total_steps (= 1/step).
    """
    if n <= 0:
        return []

    total_steps = int(round(1.0 / step))
    if positive:
        # Give each part 1 step, then distribute the rest
        base = [1] * n
        remaining = total_steps - n
    else:
        base = [0] * n
        remaining = total_steps

    for _ in range(remaining):
        i = rng.randrange(n)
        base[i] += 1

    return [round(x * step, 2) for x in base]


def _rand_step(rng: random.Random, lo: float, hi: float, step: float) -> float:
    """
    Uniformly sample from the grid {lo, lo+step, ..., hi}, inclusive.
    """
    n_steps = int(round((hi - lo) / step))
    idx = rng.randint(0, n_steps)
    return round(lo + idx * step, 2)


# ----------------------------
# Public API
# ----------------------------
def generate_random_network_modifiers(
    num_subarches: int = 5,
    step: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[int, List[float]]:
    """
    Generate one NETWORK_MODIFIERS dict with raw integer indices:
      {0: [c1, c2, c3, c4, c5], 1: [...], ...}

    Constraints:
      1) First column (c1) sums to 1 across rows, each value ∈ {0.05, 0.10, ..., 1.00} and >0
      2) Second column (c2) sums to 1 across rows, same grid and >0
      3) Third  column (c3) ∈ [0.50, 1.00], step=0.05
      4) Fourth column (c4) ∈ [0.05, 0.50], step=0.05
      5) Fifth  column (c5) ∈ [0.05, 0.50], step=0.05

    All numbers are on a 0.05 grid and within [0, 1].
    """
    rng = random.Random(seed)

    # Columns 1 & 2: partitions on the 0.05 grid summing to 1
    col1 = _random_partition_on_grid(rng, num_subarches, step=step, positive=True)
    col2 = _random_partition_on_grid(rng, num_subarches, step=step, positive=True)

    out: Dict[int, List[float]] = {}
    for i in range(num_subarches):
        c3 = _rand_step(rng, 0.50, 1.00, step)   # flops weight
        c4 = _rand_step(rng, 0.05, 0.50, step)   # flax  weight
        c5 = _rand_step(rng, 0.05, 0.50, step)   # depth/“how long”
        out[i] = [col1[i], col2[i], c3, c4, c5]
    return out


def generate_random_modifiers_list(
    n_sets: int,
    num_subarches: int = 5,
    step: float = 0.05,
    seed: Optional[int] = None,
) -> List[Dict[int, List[float]]]:
    """
    Generate `n_sets` independent NETWORK_MODIFIERS dicts, deterministically if `seed` is given.
    """
    master = random.Random(seed)
    seeds = [master.randrange(0, 2**31 - 1) for _ in range(n_sets)]
    return [
        generate_random_network_modifiers(num_subarches=num_subarches, step=step, seed=s)
        for s in seeds
    ]


def save_modifiers_csv(
    modifiers_list: List[Dict[int, List[float]]],
    path: str | Path,
) -> None:
    """
    Save modifiers to a compact CSV with two columns:
      idx, network_modifiers

    where network_modifiers is a JSON object with raw integer keys and 5-element lists.
    Example cell value:
      {"0":[0.25,0.30,0.90,0.15,0.10],"1":[0.35,0.25,0.80,0.05,0.05],...}
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "network_modifiers"])
        for idx, modifiers in enumerate(modifiers_list, start=1):
            # Ensure keys are ints in Python; json will serialize them to strings,
            # which we’ll restore to ints in load_modifiers_csv.
            writer.writerow([idx, json.dumps(modifiers, separators=(",", ":"))])


def load_modifiers_csv(path: str | Path) -> List[Dict[int, List[float]]]:
    """
    Load modifiers saved by save_modifiers_csv (two columns: idx, network_modifiers(JSON)).
    Coerces JSON keys back to ints.
    """
    path = Path(path)
    out: List[Dict[int, List[float]]] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "network_modifiers" not in reader.fieldnames:
            raise ValueError(
                f"Expected columns 'idx,network_modifiers' in {path}, "
                f"found: {reader.fieldnames}"
            )
        for row in reader:
            raw = row["network_modifiers"]
            if not raw:
                continue
            obj = json.loads(raw)
            # Convert keys back to ints (JSON object keys are strings)
            fixed: Dict[int, List[float]] = {int(k): v for k, v in obj.items()}
            out.append(fixed)

    return out
