# Arch Optimization Benchmarking

## Run
Run the script from root directory.
```
python scripts/run_experiments.py --regenerate-modifiers
```

Results are then saved in `tasks/10 tries @60days no noise/results_baselines.txt`.

### Soft vs. Hard QPS
Sometimes if we set the hard mininum QPS, the optimizers (e.g. simulated annealing) can run into situations where the result is not valid and fail to keep exploring. We thus also try soft gating of QPS while still reporting only the best results when gated with hard QPS requirements.

```
# Hard QPS
python scripts/run_experiments.py \
  --root-dir /shared_data0/weiqiuy/ads_mle_simulation \
  --n-modifiers 10 \
  --seed 0 \
  --qps-min 3500

# Soft QPS
python scripts/run_experiments.py \
  --root-dir /shared_data0/weiqiuy/ads_mle_simulation \
  --n-modifiers 10 \
  --seed 0 \
  --soft-qps \
  --soft-qps-tau 0.15 \
  --qps-min 3500
```

Currently soft QPS is not actually helping, so please just default to hard QPS.

## Neighbor generation (local search move)

When Simulated Annealing (and any local-search routine) proposes a new architecture, it applies **one small edit** to the current nested arch. The move is sampled from:

- **width** (40%)
- **depth** (35%)
- **add** sub-arch (15%)
- **remove** sub-arch (10%)

### Constraints & ladders
- **Widths** restricted to powers of two: `{64, 128, 256, 512, 1024, 2048}`.
- **Depth per sub-arch** ∈ `[1, 5]`.
- **Number of sub-arches** ∈ `[1, 5]`.
- All layers within a sub-arch share the same width.

### Edits
1. **Width tweak (40%)**
   - Pick a random sub-arch `i`.
   - Locate current width index on the power-of-two ladder.
   - Move **±1 step** (clamped to bounds).
   - Set all layers in sub-arch `i` to the new width.

2. **Depth change (35%)**
   - Pick a random sub-arch `i`.
   - With 50% chance (and if depth > 1), **remove** the last layer.
   - Otherwise, if depth < 5, **add** one layer (same width).

3. **Add sub-arch (15%)**
   - If #sub-arches < 5, append a new sub-arch:
     - Depth uniform in `[1, 5]`.
     - Width sampled from the power-of-two ladder.
     - All layers use that width.

4. **Remove sub-arch (10%)**
   - If #sub-arches > 1, remove one random sub-arch.

### Validity & no-ops
- Every candidate is validated against the constraints above.
- To avoid **no-op** moves (e.g., trying to add depth at max depth), we:
  - Retry up to **16** times to produce a neighbor **different** from the input.
  - If still identical (rare), fall back to a random valid sample different from the input.
