# Filtering Bias Analysis

Monte Carlo estimates of bias sources in `filter_new_token` (kvpress
`FilteringPress`). 16,384 decode steps, 0.5 compression ratio, random
keys on GPU, single runs.

## Bias Sources

### 1. Threshold parameter: `num_scored` vs `total_tokens_seen`

| Scoring | `num_scored` | `total_tokens_seen` |
|---|---|---|
| Random | 50.0% | 70.5% |
| KeyDiff | 0.1% | 50.3% |

With any scoring function, cached tokens are self-selected survivors
with skewed-high scores. A new token drawn from the full population
competes against this elite subset. `num_scored` calibrates the
threshold to the subset, making it too strict. `total_tokens_seen`
compensates by setting the threshold as if skipped tokens were still
competing — the fair comparison. With random scores no self-selection
occurs, so `total_tokens_seen` inflates the rate instead.

### 2. `int()` vs `round()`

| Rounding | KeyDiff + total_tokens_seen |
|---|---|
| `round()` | 50.3% |
| `int()` | 54.8% |

`int()` truncates down, making `n_kept` systematically smaller and
the threshold easier to pass.

### 3. Even-head majority vote

With 4 heads, `>= 0.5` resolves ties (2/4) to KEEP. P(KEEP) =
68.75% instead of 50%. Odd head counts eliminate ties entirely and
produce the same rate as a single head.

## Combined Results (1 head)

| Configuration | Random | KeyDiff |
|---|---|---|
| num_scored + round | **50.0%** | 0.1% |
| total_tokens_seen + round | 70.5% | **50.3%** |
| total_tokens_seen + int | 70.5% | 54.8% |

## Additional Findings

- **Prefill does not affect the steady-state rate.** Random prefill
  does not change random-score experiments; top-k prefill does not
  change KeyDiff + `total_tokens_seen` experiments. The system
  converges to the same equilibrium regardless.
- **Odd head count = single head.** Any odd number of heads (1, 3, 5,
  7) gives the same keep rate once ties are removed.

## Conclusion

kvpress's `total_tokens_seen` + `int()` gives 54.8% with KeyDiff.
`total_tokens_seen` is essential (without it: 0.1%). Switching to
`round()` would bring it to 50.3%. Even-head tie bias is an
additional concern. Results are Monte Carlo estimates with synthetic
keys; real model keys may differ.
