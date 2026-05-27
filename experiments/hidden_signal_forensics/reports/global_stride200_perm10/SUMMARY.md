# Hidden Signal Forensic Screen

## Headline

- Rows: `235637` (`157428` train, `78209` validation).
- Dates: `0` to `1698`; split date `1274`.
- Best univariate rule: `feature_04::z` with validation R2 `0.0012275946745993194`.
- Best row-index modulo rule: `mod 14` with validation R2 `4.5139568735863556e-05`.
- Permutation null p95/max: `0.00022340786360929276` / `0.000269216121299265`.
- Marchenko-Pastur upper outliers: `19`.

## Interpretation Boundary

This screen is exploratory. A positive simple rule is only interesting if it is
materially above the permutation-search null and remains stable in a frozen
temporal validation. A low score is not evidence that no market signal exists;
it only weakens the specific hidden-pattern families tested here.
