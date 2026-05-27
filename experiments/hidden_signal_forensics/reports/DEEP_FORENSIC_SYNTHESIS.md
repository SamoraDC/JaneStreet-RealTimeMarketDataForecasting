# Deep Hidden Signal Forensics: Synthesis

## Protocol

The direct model-improvement path was paused. A separate forensic suite was
implemented under `experiments/hidden_signal_forensics/` to test finite
hypothesis families:

- frozen simple rules: `feature_04::z` and `feature_16::square_z`;
- strong nulls: IID permutation, block shuffle, date-block shuffle, circular
  shift, and IAAFT surrogates;
- low-order interactions among top discovered features;
- cross-sectional date/date-time ranks and z-scores;
- latent rank models: Ridge and PLS on rank-normalized features;
- residual mining after the best latent rank model.

All scores below are weighted zero-mean R2 on real sampled Kaggle rows.

## Main Runs

| Run | Rows | Dates | Best univariate | Interaction | Cross-sectional | Latent rank | Residual final | IAAFT max | Interaction null max |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `deep_recent900_stride80_250k` | 250,000 | 900-1461 | `0.001577812` | `0.001228204` | `0.001368708` | `0.003024782` | `0.003337477` | `0.000287141` | `0.000137488` |
| `deep_global_stride200_250k` | 235,637 | 0-1698 | `0.001227595` | `0.001781669` | `0.001270074` | `0.002092472` | `0.003229717` | `0.000076553` | `0.000364837` |

## Findings

### 1. There Is Real Simple Signal

Simple rules beat IID, block, date, circular, and IAAFT nulls. This rejects the
strongest version of “everything is random noise” for these sampled windows.

However, this does not imply a synthetic planted key. Weak simple signal is also
expected in real market-derived data after anonymization.

### 2. `feature_04::z` Is More Stable Than `feature_16::square_z`

`feature_04::z` survives walk-forward validation in both recent and global
runs. Its R2 is small but consistently positive.

`feature_16::square_z` is much less stable. It appeared in an earlier recent
screen but fails or nearly vanishes in several walk-forward windows. It should
not be treated as a robust standalone hypothesis.

### 3. Interactions Help, But They Do Not Look Like A Hidden XOR Key

Best interactions:

- recent: `feature_36:feature_58::z_diff`, R2 `0.001228204`;
- global: `feature_04:feature_47::z_diff`, R2 `0.001781669`.

They beat interaction-search nulls, but the winning forms are mostly linear
spreads/differences, not parity, modulo, or bit-like rules. This points toward
ordinary latent factor geometry rather than a planted cryptographic-style rule.

### 4. Cross-Sectional Normalization Is Useful But Not Dominant

Best cross-sectional variants are close to the best univariate rules but below
rank latent models:

- recent: `feature_36::cs_z::date`, R2 `0.001368708`;
- global: `feature_04::cs_rank::date`, R2 `0.001270074`.

This suggests relative/rank information matters, but not as a standalone
solution.

### 5. Rank Latent Models Are The Strongest Clean Family

The best clean family is rank-normalized latent modeling:

- recent: `pls_rank_k8`, R2 `0.003024782`;
- global: `ridge_rank_alpha10000`, R2 `0.002092472`.

This is the strongest evidence from the suite. The signal appears to live in a
low-dimensional ordinal/latent factor space, not in a single obvious formula.

### 6. Residual Mining Adds Signal But Is Adaptive

Residual correction after the best latent model improves:

- recent: `0.003024782 -> 0.003337477`, delta `+0.000312695`;
- global: `0.002092472 -> 0.003229717`, delta `+0.001137245`.

This is promising, but it is selected adaptively on the same validation family.
It must be frozen and revalidated before being treated as a candidate modeling
component.

## Synthetic-Key Assessment

Evidence **against** a simple planted key:

- modulo/index tests were weak or negative;
- digit anomalies are explained by low-cardinality integer-like features;
- periodogram peaks have tiny power share;
- winning interactions are smooth spreads, not parity/mod/bit rules;
- best feature families drift by window.

Evidence **for** exploitable hidden structure:

- simple transforms beat strong nulls;
- rank latent models materially improve over univariate rules;
- residual mining finds remaining structured error;
- Marchenko-Pastur outliers confirm non-random feature covariance, although
  this is normal in market-like data.

Conclusion: the current evidence supports **weak market-like latent structure**,
not a clear synthetic planted formula.

## Decision

Do not continue blind rule search in the same validation windows. The next valid
step is to freeze a small candidate set:

1. `feature_04::z`;
2. recent latent: `pls_rank_k8`;
3. global latent: `ridge_rank_alpha10000`;
4. residual correction candidates:
   - recent: `pls_rank_k8 + feature_59::z`;
   - global: `ridge_rank_alpha10000 + feature_47::z`.

Then validate these candidates in a new temporal protocol without changing
features, transforms, model family, or hyperparameters.

## Technical Audit

- All implemented screens fit parameters on train and report validation scores.
- Cross-sectional transforms use current feature groups only; they are
  diagnostic and target-free.
- Null tests preserve the same search family where applicable.
- IAAFT surrogates preserve the target marginal distribution approximately by
  construction and preserve Fourier amplitudes iteratively, while breaking the
  direct feature-target alignment.
- Residual mining was audited after an implementation inconsistency was found:
  train and validation residuals now use the same selected latent baseline.
- Full test suite after implementation: `155 passed`.

## Ethical Audit

This study does not prove the competition data is synthetic. It also does not
prove absence of hidden planted signal. It does show that the next rational
research direction is not heavier black-box models, but frozen rank-latent and
residual-correction hypotheses with strict temporal validation.
