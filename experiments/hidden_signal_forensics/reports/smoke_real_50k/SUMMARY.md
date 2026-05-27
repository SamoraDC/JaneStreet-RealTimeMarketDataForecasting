# Hidden Signal Forensic Screen

## Headline

- Rows: `50000` (`33107` train, `16893` validation).
- Dates: `0` to `602`; split date `452`.
- Best univariate rule: `feature_06::rank` with validation R2 `0.0022106031520279235`.
- Best row-index modulo rule: `mod 32` with validation R2 `0.0001832617541254189`.
- Permutation null p95/max: `0.000325451560019685` / `0.0003348075896568714`.
- Marchenko-Pastur upper outliers: `18`.

## Interpretation Boundary

This screen is exploratory. A positive simple rule is only interesting if it is
materially above the permutation-search null and remains stable in a frozen
temporal validation. A low score is not evidence that no market signal exists;
it only weakens the specific hidden-pattern families tested here.
