# Deep Hidden Signal Forensic Suite

## Headline

- Rows: `235637` (`157428` train, `78209` validation).
- Dates: `0` to `1698`; split date `1274`.
- Best univariate: `feature_04::z` with R2 `0.0012275946745993194`.
- Best interaction: `feature_04:feature_47::z_diff` with R2 `0.001781669320267354`.
- Best cross-sectional transform: `feature_04::cs_rank::date` with R2 `0.0012700738896652686`.
- Best latent model: `ridge_rank_alpha10000` with R2 `0.0020924724141028195`.
- Best residual correction final R2: `0.0032297171013584425`; delta `0.001137244687255623`.
- IAAFT best null R2: `7.655269001094478e-05`.
- Interaction-search null max R2: `0.00036483683811960876`.

## Audit Boundary

This suite is exhaustive for the finite hypothesis families implemented here:
frozen simple rules, strong nulls, low-order interactions, cross-sectional
normalization, rank latent models, residual mining and IAAFT surrogates. It is
not proof that no other hidden signal exists.

Any candidate found here is still adaptive unless it is explicitly frozen and
validated on a later temporal window.
