# Deep Hidden Signal Forensic Suite

## Headline

- Rows: `50000` (`37266` train, `12734` validation).
- Dates: `900` to `1461`; split date `1321`.
- Best univariate: `feature_36::z` with R2 `0.00225982361965793`.
- Best interaction: `feature_36:feature_37::rank_diff` with R2 `0.0016413818284308768`.
- Best cross-sectional transform: `feature_36::cs_z::date` with R2 `0.0018506103744460045`.
- Best latent model: `ridge_rank_alpha10000` with R2 `0.001633463481775732`.
- Best residual correction final R2: `0.002458254830002393`; delta `0.0008247913482266611`.
- IAAFT best null R2: `0.00034579509431553745`.
- Interaction-search null max R2: `0.0004344141647942834`.

## Audit Boundary

This suite is exhaustive for the finite hypothesis families implemented here:
frozen simple rules, strong nulls, low-order interactions, cross-sectional
normalization, rank latent models, residual mining and IAAFT surrogates. It is
not proof that no other hidden signal exists.

Any candidate found here is still adaptive unless it is explicitly frozen and
validated on a later temporal window.
