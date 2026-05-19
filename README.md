# Shannon Capacity — Largest Independent Set in C₇⁵

Construct the largest independent set in the 5th strong power of C₇ (C₇⁵). Target: 367, matching the best-known lower bound for the Shannon capacity of C₇.

## Files

| File | Description |
|------|-------------|
| `best_program.py` | Deterministic version (fixed iteration counts), fully reproducible |
| `best_program_origin.py` | Original evolved version (time-based), semi-reproducible |
| `verifier.py` | Standalone verification and reproduction tool |
| `related_files/shannon_capacity/config-standard.yaml` | Evolution configuration |
| `related_files/shannon_capacity/evaluator.py` | Project-specific eval entry point for the OpenEvolve framework |
| `related_files/openevolve/hpo.py` | HPO module from an unpublished OpenEvolve fork (reference only) |
| `eval_results/` | Pre-computed independent sets |

## Requirements

- Python ≥ 3.10
- NumPy (for `verifier.py`, standalone reproduction)

The full evolution pipeline requires the unpublished OpenEvolve fork. `verifier.py` works without it — it includes a `MockHPO` that replaces the framework's `tunable()`.

## Reproduction

### Default mode

```bash
python verifier.py best_program.py
```

Parameters: `greedy_iters=1525`, `local_iters=32182`, `random_seed=42`.

Each run produces identical output — same 367 nodes, same score.

### Reproduce `eval_results/`

```bash
python verifier.py best_program.py spectrum
```

Uses `local_iters=200000` across 9 `greedy_iters` values (1474–1600). All produce Score 367. Saved as `eval_results/nodes_g*.npy`, shape `(367, 5)`, int64.

## Algorithm

Two-phase:

1. **Greedy**: permute node order randomly, pick nodes greedily, repeat for `greedy_iters` rounds, keep the best.
2. **Local search**: remove one node, try to insert 1–2 replacements, repeat.

Different `greedy_iters` give different starting points, but all reach 367 — there are many distinct maximum independent sets in this graph.

## HPO Parameters

`best_program.py` uses fixed values from HPO:

| Parameter | Value | Role |
|-----------|-------|------|
| `rev_prob` | 0.55 | Probability of reversing the node order |
| `perm_prob` | 0.05 | Probability of permuting coordinate axes |
| `greedy_iters` | 1525 | Greedy rounds |
| `local_iters` | 32182 | Local search iterations |

## HPO

`related_files/openevolve/hpo.py` is from an **unpublished fork** of OpenEvolve, included for reference only. It is not importable from this directory.

Design contract:

- Programs declare tunable parameters via `tunable(name, range, default)`. During evolution, Optuna samples the range; during standalone verification with `verifier.py`, the default is used.
- **hpo_mode**: enabled during HPO search, disabled for final evaluation. In hpo_mode the evaluator skips expensive validation (conflict detection, feature extraction) and uses raw point count as the objective. This keeps HPO trials fast. The contract is that a program should produce a valid independent set regardless of hpo_mode — the hpo_mode score (raw count) must match the validated score. In other words, the program is expected to be conflict-free across the parameter distribution, not just at the final fixed values.

Both `related_files/shannon_capacity/evaluator.py` and `related_files/openevolve/hpo.py` require the unpublished fork and will not run standalone.

## Acknowledgments

- Feishu (Shanghai) Technology Co., Ltd — LLM API credits and compute resources for this project
- OpenEvolve framework contributors, for the evolution infrastructure — <https://github.com/algorithmicsuperintelligence/openevolve>
