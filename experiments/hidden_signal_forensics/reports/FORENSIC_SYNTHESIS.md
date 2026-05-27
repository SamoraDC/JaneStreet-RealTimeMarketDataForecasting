# Hidden Signal Forensics: First Evidence Synthesis

## Objective

Pause model-improvement work and test a different hypothesis: the data may be
noisy financial data with hidden planted patterns, or synthetic/randomized data
with low-amplitude predictive bias. This report covers the first controlled
screen under `experiments/hidden_signal_forensics/`.

## Runs

| Run | Rows | Dates | Split | Best simple rule | Valid R2 | Permuted-search max | MP upper outliers |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| `global_stride200_perm10` | 235,637 | 0-1698 | 1274 | `feature_04::z` | `0.001227595` | `0.000269216` | 19 |
| `recent1200_stride50_perm10` | 300,000 | 1200-1608 | 1506 | `feature_16::square_z` | `0.001029502` | `0.000099635` | 19 |

Implementation check: `uv run pytest -q` passed with `150 passed`.

The best simple rules are above the small permutation nulls. This says there is
detectable signal in simple transforms. It does **not** say the data is
synthetic, nor that we found a planted key.

## What Looks Real-Market-Like

- The target has nonzero autocorrelation in squared returns/responses:
  volatility clustering is visible in both runs.
- The Marchenko-Pastur screen finds many correlation eigenvalue outliers. That
  is expected in market-like data with latent factors and cross-sectional
  structure; it is not by itself evidence of artificial generation.
- The best simple predictors change by regime: global sample favors
  `feature_04::z`; recent sample favors `feature_16::square_z`. A stable planted
  global formula would more likely replicate the same feature/transform.

## What Looks Potentially Interesting

- Simple univariate transforms beat the repeated target-permutation search null.
  This supports further investigation of low-dimensional formulae.
- `feature_04` has a stable negative linear/rank relation in the global sample.
- `feature_16::square_z` is strongest in the recent sample, suggesting a
  magnitude/volatility-style relation rather than a directional linear one.
- Rank, sign, square, absolute value, and tail indicators all show up in the top
  table; these are exactly the transforms worth testing before heavier models.

## What Did Not Support a Hidden-Key Hypothesis

- Row-index modulo tests are weak. Global best modulo R2 is only `0.000045140`;
  recent best is negative. No useful `row_id % k` pattern appeared.
- Digit tests show extreme non-uniformity for `feature_09`, `feature_10`, and
  `feature_11`, but those columns are low-cardinality/integer-like features
  rather than clear evidence of least-significant-digit watermarking.
- Periodogram peaks have tiny power share. No dominant artificial sinusoidal
  pattern appeared in this sample.

## Methodological Limit

This is still exploratory. The same sampled OOF screen generated and evaluated
the simple-rule family. The permutation null partially controls data snooping by
rerunning the same search on shuffled targets, but a candidate rule still needs:

1. freeze the feature/transform;
2. validate on a different temporal window;
3. test against block/circular-shift nulls;
4. inspect stability by `date_id`, `time_id`, `symbol_id`, and `weight`;
5. only then consider adding it to a predictive model.

## Current Conclusion

The evidence supports “there are weak simple signals hidden in the feature
space” but does not support “the competition host planted a simple synthetic
key.” The structure observed so far is compatible with real anonymized market
data: latent factors, volatility clustering, regime drift, low R2, and weak
feature transforms.

The next scientifically clean step is not to search hundreds more rules on the
same split. It is to freeze the two discovered hypotheses:

- `feature_04::z` from the global run;
- `feature_16::square_z` from the recent run;

and validate them on independent temporal windows plus stronger nulls.
