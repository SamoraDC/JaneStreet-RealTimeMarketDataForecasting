# Deep Hidden Signal Forensic Suite

## Headline

- Rows: `250000` (`186328` train, `63672` validation).
- Dates: `900` to `1461`; split date `1321`.
- Best univariate: `feature_36::z` with R2 `0.0015778120610383528`.
- Best interaction: `feature_36:feature_58::z_diff` with R2 `0.0012282036124739992`.
- Best cross-sectional transform: `feature_36::cs_z::date` with R2 `0.0013687083951414714`.
- Best latent model: `pls_rank_k8` with R2 `0.0030247820411032356`.
- Best residual correction final R2: `0.003337476701052444`; delta `0.00031269465994920864`.
- IAAFT best null R2: `0.00028714147771990994`.
- Interaction-search null max R2: `0.0001374881134207362`.

## Audit Boundary

This suite is exhaustive for the finite hypothesis families implemented here:
frozen simple rules, strong nulls, low-order interactions, cross-sectional
normalization, rank latent models, residual mining and IAAFT surrogates. It is
not proof that no other hidden signal exists.

Any candidate found here is still adaptive unless it is explicitly frozen and
validated on a later temporal window.
