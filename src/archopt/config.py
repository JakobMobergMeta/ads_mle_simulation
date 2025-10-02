from pathlib import Path
from typing import List

EVAL_TRAINING_DAYS = 60
N_TRIALS = 10
SEEDS = [0]

WIDTH_MIN, WIDTH_MAX = 64, 2048
MIN_LAYERS_PER_SUB, MAX_LAYERS_PER_SUB = 1, 5
MIN_SUBARCHES, MAX_SUBARCHES = 1, 5
N_WIDTH_LEVELS = 7
QPS_MIN = 3500.0

BASELINE_ARCH: List[List[int]] = [[256,256],[512,512],[256,256],[64],[64]]
OUT_ROOT = Path("benchmark_results_nested_arch")
NETWORKS_CSV_PATH = "../tasks/10 tries @60days no noise/networks.csv"
