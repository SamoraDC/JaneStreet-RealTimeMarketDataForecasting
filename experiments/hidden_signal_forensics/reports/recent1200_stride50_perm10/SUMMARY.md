# Hidden Signal Forensic Screen

## Headline

- Rows: `300000` (`224131` train, `75869` validation).
- Dates: `1200` to `1608`; split date `1506`.
- Best univariate rule: `feature_16::square_z` with validation R2 `0.0010295024142236153`.
- Best row-index modulo rule: `mod 27` with validation R2 `-0.00021944471821155886`.
- Permutation null p95/max: `9.92093262958127e-05` / `9.963510310462276e-05`.
- Marchenko-Pastur upper outliers: `19`.

## Interpretation Boundary

This screen is exploratory. A positive simple rule is only interesting if it is
materially above the permutation-search null and remains stable in a frozen
temporal validation. A low score is not evidence that no market signal exists;
it only weakens the specific hidden-pattern families tested here.
