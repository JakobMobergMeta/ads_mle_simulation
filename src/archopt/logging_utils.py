import json, csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from .benchmark import RunLog

def save_runs(results: Dict[str, List[RunLog]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, runs in results.items():
        for i, run in enumerate(runs):
            fp = out_dir / f"{method}_seed{i}.csv"
            with fp.open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["step","best_so_far","value","elapsed","config_json"])
                best = -float("inf")
                for t, tr in enumerate(run.trials, start=1):
                    best = max(best, tr.value)
                    w.writerow([t, best, tr.value, tr.elapsed, json.dumps(tr.config)])

def save_arches_per_method(results: Dict[str, List[RunLog]], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, runs in results.items():
        fp = out_dir / f"{method}_arches.csv"
        with fp.open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["seed","step","arch_json","score"])
            for i, run in enumerate(runs):
                seed_id = run.trials[0].seed if run.trials else i
                for step, tr in enumerate(run.trials, start=1):
                    w.writerow([seed_id, step, json.dumps(tr.config.get("arch")), tr.value])

def plot_runs(results: Dict[str, List[RunLog]], title_prefix: str):
    for method, runs in results.items():
        if not runs: continue
        L = min(len(r.trials) for r in runs)
        curves = np.array([r.best_curve()[:L] for r in runs], dtype=float)
        mean, std = curves.mean(0), curves.std(0)
        x = np.arange(1, L+1)
        plt.figure(); plt.plot(x, mean, label=f"{method} (mean)")
        plt.fill_between(x, mean-std, mean+std, alpha=0.2)
        plt.xlabel("Trials"); plt.ylabel("Best-so-far SCORE (higher is better)")
        plt.title(f"{title_prefix}Anytime: {method}")
        plt.legend(); plt.tight_layout(); plt.show()
