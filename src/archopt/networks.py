# src/archopt/networks.py
from __future__ import annotations
import math, json, csv
import numpy as np
from pathlib import Path
from typing import Dict, List

from .config import (
    WIDTH_MIN, WIDTH_MAX, WIDTH_STEP,
    MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB,
    MIN_SUBARCHES,      MAX_SUBARCHES,
)

# ------------------------------------------------------------
# Width ladders: Linear (default) and Powers-of-two (optional)
# ------------------------------------------------------------
def _linear_levels(lo: int, hi: int, step: int) -> List[int]:
    """Generate width levels with equal step size."""
    levels: List[int] = []
    w = lo
    while w <= hi:
        levels.append(w)
        w += step
    return levels

def _pow2_levels(lo: int, hi: int) -> List[int]:
    """Generate width levels as powers of two."""
    if lo < 1: lo = 1
    levels: List[int] = []
    w = 1
    while w < lo:
        w <<= 1
    while w <= hi:
        levels.append(w)
        w <<= 1
    # ensure we stay within [WIDTH_MIN, WIDTH_MAX]
    levels = [v for v in levels if WIDTH_MIN <= v <= WIDTH_MAX]
    if not levels:
        # fallback to clamped ends if something odd happens
        levels = [max(1, WIDTH_MIN)]
        while levels[-1] < WIDTH_MAX:
            nxt = levels[-1] << 1
            if nxt > WIDTH_MAX: break
            levels.append(nxt)
    return levels

# Default: Linear widths with step size 64 (e.g. [64, 128, 192, ..., 4096])
LINEAR_WIDTHS: List[int] = _linear_levels(WIDTH_MIN, WIDTH_MAX, WIDTH_STEP)
# Optional: Powers-of-two widths (e.g. [64, 128, 256, 512, 1024, 2048, 4096])
POW2_WIDTHS: List[int] = _pow2_levels(WIDTH_MIN, WIDTH_MAX)

# Global variable to control which width ladder to use
USE_POW2_WIDTHS = False  # Default is linear widths

def get_width_levels() -> List[int]:
    """Return the currently active width ladder."""
    return POW2_WIDTHS if USE_POW2_WIDTHS else LINEAR_WIDTHS

def set_width_mode(use_pow2: bool = False):
    """Set the width ladder mode (linear or power-of-2)."""
    global USE_POW2_WIDTHS
    USE_POW2_WIDTHS = use_pow2

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def validate_arch_nested(arch: List[List[int]]) -> None:
    assert isinstance(arch, list), "arch must be a list of sub-architectures"
    assert MIN_SUBARCHES <= len(arch) <= MAX_SUBARCHES, f"#subarches out of bounds"
    width_levels = get_width_levels()
    for sub in arch:
        assert isinstance(sub, list), "each sub-arch must be a list"
        assert MIN_LAYERS_PER_SUB <= len(sub) <= MAX_LAYERS_PER_SUB, "depth out of bounds"
        for w in sub:
            assert isinstance(w, int), "widths must be ints"
            assert WIDTH_MIN <= w <= WIDTH_MAX, "width outside global bounds"
            # enforce width is in the current width ladder
            assert w in width_levels, f"width {w} is not in the width ladder"

# ------------------------------------------------------------
# Sampling (using current width ladder)
# ------------------------------------------------------------
def sample_subarch_pow2(rng: np.random.RandomState, fixed_depth: bool = False) -> List[int]:
    """Sample a sub-architecture using the current width ladder."""
    width_levels = get_width_levels()
    if fixed_depth:
        depth = MAX_LAYERS_PER_SUB
    else:
        depth = int(rng.randint(MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB + 1))
    wi = int(rng.randint(0, len(width_levels)))
    w = int(width_levels[wi])
    return [w] * depth

def sample_arch_pow2(rng: np.random.RandomState, fixed_depth: bool = False, fixed_num_subarches: bool = False) -> List[List[int]]:
    if fixed_num_subarches:
        s = MAX_SUBARCHES
    else:
        s = int(rng.randint(MIN_SUBARCHES, MAX_SUBARCHES + 1))
    return [sample_subarch_pow2(rng, fixed_depth=fixed_depth) for _ in range(s)]

# ------------------------------------------------------------
# Neighborhood moves (support k_max & force_replace)
# ------------------------------------------------------------
def neighbor_arch_pow2(
    rng: np.random.RandomState,
    arch: List[List[int]],
    *,
    k_max: int = 1,
    force_replace: bool = False,
    fixed_depth: bool = False,
    fixed_num_subarches: bool = False,
) -> List[List[int]]:
    """
    Local edits on powers-of-two ladder:
      - width: move width index by ±k where 1<=k<=k_max
      - depth: +/- one layer (disabled if fixed_depth=True)
      - add/remove: add or remove a sub-arch (disabled if fixed_num_subarches=True)
      - replace: swap one sub-arch with a fresh random one (used when force_replace=True)

    Returns a valid architecture; falls back to a random sample if needed.
    """
    if not arch:
        return sample_arch_pow2(rng, fixed_depth=fixed_depth, fixed_num_subarches=fixed_num_subarches)

    new_arch = [list(sub) for sub in arch]  # deep copy

    if force_replace:
        # Replace one sub-arch entirely
        i = int(rng.randint(0, len(new_arch)))
        new_arch[i] = sample_subarch_pow2(rng, fixed_depth=fixed_depth)
    else:
        if fixed_num_subarches:
            # When fixed_num_subarches is True, only allow width and depth changes
            if fixed_depth:
                # Only width changes
                ops = ["width"]
                p   = [1.0]
            else:
                # Width and depth changes
                ops = ["width", "depth"]
                p   = [0.60,    0.40]
        elif fixed_depth:
            # When fixed_depth is True, only allow width changes and add/remove sub-arches
            ops = ["width", "add", "remove"]
            p   = [0.80,    0.10,  0.10]
        else:
            # All operations allowed
            ops = ["width", "depth", "add", "remove"]
            p   = [0.45,    0.35,    0.10,  0.10]
        op = rng.choice(ops, p=p)

        if op == "width":
            width_levels = get_width_levels()
            i = int(rng.randint(0, len(new_arch)))
            cur_w = new_arch[i][0]
            try:
                idx = width_levels.index(cur_w)
            except ValueError:
                # if somehow not on ladder, snap to nearest by log distance (for pow2) or absolute distance (for linear)
                if USE_POW2_WIDTHS:
                    idx = int(np.argmin([abs(math.log(max(1, cur_w)) - math.log(w)) for w in width_levels]))
                else:
                    idx = int(np.argmin([abs(cur_w - w) for w in width_levels]))
            # jump by ±k where k in [1, k_max]
            k = int(rng.randint(1, max(1, k_max) + 1))
            step = int(rng.choice([-k, k]))
            idx = max(0, min(len(width_levels) - 1, idx + step))
            new_w = int(width_levels[idx])
            new_arch[i] = [new_w] * len(new_arch[i])

        elif op == "depth":
            i = int(rng.randint(0, len(new_arch)))
            d = len(new_arch[i])
            if rng.rand() < 0.5 and d > MIN_LAYERS_PER_SUB:
                new_arch[i] = new_arch[i][:-1]
            elif d < MAX_LAYERS_PER_SUB:
                new_arch[i] = new_arch[i] + [new_arch[i][0]]

        elif op == "add" and len(new_arch) < MAX_SUBARCHES:
            new_arch.append(sample_subarch_pow2(rng, fixed_depth=fixed_depth))

        elif op == "remove" and len(new_arch) > MIN_SUBARCHES:
            i = int(rng.randint(0, len(new_arch)))
            new_arch.pop(i)

    # Validate, else fallback
    try:
        validate_arch_nested(new_arch)
        return new_arch
    except AssertionError:
        return sample_arch_pow2(rng, fixed_depth=fixed_depth, fixed_num_subarches=fixed_num_subarches)

# ------------------------------------------------------------
# Encoding helpers for skopt/optuna (index over width ladder)
# ------------------------------------------------------------
def arch_to_params(arch: List[List[int]]) -> Dict[str, int]:
    validate_arch_nested(arch)
    width_levels = get_width_levels()
    n_sub = len(arch)
    params: Dict[str, int] = {"n_sub": n_sub}
    for i, sub in enumerate(arch):
        d = len(sub)
        w = sub[0]
        try:
            k = width_levels.index(w)
        except ValueError:
            # snap to nearest ladder idx
            if USE_POW2_WIDTHS:
                k = int(np.argmin([abs(math.log(max(1, w)) - math.log(v)) for v in width_levels]))
            else:
                k = int(np.argmin([abs(w - v) for v in width_levels]))
        params[f"depth_{i}"] = int(d)
        params[f"widx_{i}"]  = int(k)
    # fill the rest (fixed-size param space)
    for i in range(n_sub, MAX_SUBARCHES):
        params[f"depth_{i}"] = MIN_LAYERS_PER_SUB
        params[f"widx_{i}"]  = 0
    return params

def params_to_arch(params: Dict[str, int]) -> List[List[int]]:
    width_levels = get_width_levels()
    n_sub = int(params["n_sub"])
    n_sub = max(MIN_SUBARCHES, min(MAX_SUBARCHES, n_sub))
    arch: List[List[int]] = []
    for i in range(n_sub):
        d = max(MIN_LAYERS_PER_SUB, min(MAX_LAYERS_PER_SUB, int(params[f"depth_{i}"])))
        k = max(0, min(len(width_levels) - 1, int(params[f"widx_{i}"])))
        w = int(width_levels[k])
        arch.append([w] * d)
    validate_arch_nested(arch)
    return arch

def params_to_x(params: Dict[str, int]) -> List[int]:
    x: List[int] = [int(params["n_sub"])]
    for i in range(MAX_SUBARCHES):
        x.append(int(params.get(f"depth_{i}", MIN_LAYERS_PER_SUB)))
    for i in range(MAX_SUBARCHES):
        x.append(int(params.get(f"widx_{i}", 0)))
    return x

def x_to_params(x: List[int]) -> Dict[str, int]:
    idx = 0
    n_sub = int(x[idx]); idx += 1
    params: Dict[str, int] = {"n_sub": n_sub}
    for i in range(MAX_SUBARCHES):
        params[f"depth_{i}"] = int(x[idx]); idx += 1
    for i in range(MAX_SUBARCHES):
        params[f"widx_{i}"] = int(x[idx]); idx += 1
    return params

# ------------------------------------------------------------
# (Optional) CSV helper kept for compatibility; not used by suggestors
# ------------------------------------------------------------
def save_global_networks_csv(path: str, n_trials: int, seed: int):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(1, n_trials + 1):
        rows.append((i, sample_arch_pow2(rng)))
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Idx", "Arch"])
        for idx, arch in rows:
            w.writerow([idx, json.dumps(arch)])
