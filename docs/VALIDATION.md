# Release validation

Executed on 4 October 2026 using the environment in `validated_environment.json`.

- 57/57 source-data and quadratic-reference checks passed.
- 13 automated tests passed, including a short CPU NUTS fit, percentage-support checks, predictive quantiles, and identical predictions after posterior save/load.
- `run.py demo` completed a 30-warm-up / 40-sample, one-chain fit and wrote posterior, predictions, and diagnostics.
- Reloading that posterior through `run.py predict` completed successfully.
- The bundled 50,000-step PFN checkpoint loaded in weights-only mode and predicted all three compounds at the three example conditions.
- All 13 OLS LOCO folds completed. Recomputed mean RMSE was 10.431, matching the archived value to three decimals.
- Archived-result reporting reproduced the four headline mean-RMSE values.
- The six companion English plots regenerated successfully from saved numerical inputs.

Full CM-BARS cross-validation, repeated MCMC runs, and 50,000-step PFN pretraining were not rerun for this code packaging task. Their existing research results are preserved under `reference_results/`; short-chain release checks do not establish posterior convergence or replace scientific validation.
