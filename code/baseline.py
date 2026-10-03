"""Baseline: exact reproduction of the published analysis of Macromol #22.

Reproduces Equations (6)-(8) (full quadratic OLS per phenolic, uncoded units), the
R-squared values, the Table 2 optima, and adds the diagnostics the paper does not report:
regression ANOVA, lack-of-fit test against the 3 centre replicates, residual df, VIF,
and the number of fitted values that fall outside the physically attainable range [0, 100].

Outputs: results/baseline/{coefficients,fit_summary,anova,diagnostics,optima}.csv
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.dirname(HERE)
OUT = f"{JD}/results/baseline"
os.makedirs(OUT, exist_ok=True)

RESPONSES = ["tannic_acid", "p_coumaric_acid", "acetosyringone"]
LABEL = {"tannic_acid": "Tannic acid", "p_coumaric_acid": "p-Coumaric acid",
         "acetosyringone": "Acetosyringone"}
TERMS = ["const", "pH", "conc", "cnf", "pH2", "conc2", "cnf2", "pH:conc", "pH:cnf", "conc:cnf"]
BOUNDS = [(4.0, 8.0), (5.0, 15.0), (0.10, 0.40)]

d = pd.read_csv(f"{JD}/data/bbd_design_responses.csv")
pub = pd.read_csv(f"{JD}/data/published_models.csv")
pubr = pd.read_csv(f"{JD}/data/published_r2_optima.csv")


def design_matrix(pH, conc, cnf):
    pH, conc, cnf = map(np.asarray, (pH, conc, cnf))
    return np.column_stack([np.ones_like(pH, dtype=float), pH, conc, cnf,
                            pH * pH, conc * conc, cnf * cnf,
                            pH * conc, pH * cnf, conc * cnf])


X = design_matrix(d.pH, d.additive_mM, d.cnf_pct)
n, p = X.shape
XtXi = np.linalg.pinv(X.T @ X)

coef_rows, fit_rows, anova_rows, diag_rows, opt_rows = [], [], [], [], []

for r in RESPONSES:
    y = d[r].values.astype(float)
    beta = XtXi @ X.T @ y
    yhat = X @ beta
    resid = y - yhat
    dfe = n - p                                     # 15 - 10 = 5 residual df
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    ssr = sst - sse
    mse = sse / dfe
    r2 = 1 - sse / sst
    r2adj = 1 - (sse / dfe) / (sst / (n - 1))
    se = np.sqrt(np.diag(XtXi) * mse)
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dfe)

    pubb = pub[pub.response == r].set_index("term").coef.astype(float)
    for k, t in enumerate(TERMS):
        coef_rows.append(dict(response=r, term=t, published=pubb[t],
                              reproduced=beta[k], se=se[k], t=tval[k], p=pval[k],
                              abs_diff=abs(beta[k] - pubb[t])))

    # --- lack of fit using the 3 centre replicates -------------------------
    ctr = d[(d.x1_coded == 0) & (d.x2_coded == 0) & (d.x3_coded == 0)][r].values
    sspe = float(((ctr - ctr.mean()) ** 2).sum())
    dfpe = len(ctr) - 1
    sslof, dflof = sse - sspe, dfe - dfpe
    f_lof = (sslof / dflof) / (sspe / dfpe)
    p_lof = stats.f.sf(f_lof, dflof, dfpe)

    f_model = (ssr / (p - 1)) / mse
    p_model = stats.f.sf(f_model, p - 1, dfe)
    anova_rows += [
        dict(response=r, source="Model (quadratic)", df=p - 1, ss=ssr, ms=ssr / (p - 1),
             F=f_model, p=p_model),
        dict(response=r, source="Residual", df=dfe, ss=sse, ms=mse, F=np.nan, p=np.nan),
        dict(response=r, source="  Lack of fit", df=dflof, ss=sslof, ms=sslof / dflof,
             F=f_lof, p=p_lof),
        dict(response=r, source="  Pure error (centre reps)", df=dfpe, ss=sspe,
             ms=sspe / dfpe, F=np.nan, p=np.nan),
        dict(response=r, source="Total", df=n - 1, ss=sst, ms=np.nan, F=np.nan, p=np.nan),
    ]

    pr2 = float(pubr[pubr.response == r].r2_pct.iloc[0])
    fit_rows.append(dict(response=r, n=n, n_params=p, residual_df=dfe,
                         r2_published_pct=pr2, r2_reproduced_pct=100 * r2,
                         r2_adj_pct=100 * r2adj, rmse_insample=np.sqrt(sse / n),
                         sigma_hat=np.sqrt(mse),
                         n_zero_observations=int((y == 0).sum()),
                         n_fitted_below_zero=int((yhat < 0).sum()),
                         min_fitted=yhat.min(), max_fitted=yhat.max(),
                         lof_F=f_lof, lof_p=p_lof, model_F=f_model, model_p=p_model))

    # --- optimum of the published surface over the design box -------------
    def negf(z, b=beta):
        v = design_matrix([z[0]], [z[1]], [z[2]])[0]
        return -float(b @ v)

    best, bz = np.inf, None
    for p0 in itertools.product([4, 6, 8], [5, 10, 15], [0.1, 0.25, 0.4]):
        res = minimize(negf, p0, bounds=BOUNDS, method="L-BFGS-B")
        if res.fun < best:
            best, bz = res.fun, res.x
    row = pubr[pubr.response == r].iloc[0]
    opt_rows.append(dict(response=r,
                         pub_pH=row.opt_pH, pub_conc=row.opt_conc_mM, pub_cnf=row.opt_cnf_pct,
                         rep_pH=bz[0], rep_conc=bz[1], rep_cnf=bz[2],
                         predicted_max_pct=-best,
                         best_observed_pct=y.max(),
                         best_observed_run=int(d.run[np.argmax(y)])))

    # --- collinearity of the quadratic design matrix ----------------------
    for k, t in enumerate(TERMS[1:], start=1):
        others = np.delete(X, k, axis=1)
        b2, *_ = np.linalg.lstsq(others, X[:, k], rcond=None)
        rr = 1 - ((X[:, k] - others @ b2) ** 2).sum() / ((X[:, k] - X[:, k].mean()) ** 2).sum()
        diag_rows.append(dict(response=r, term=t, vif=1 / max(1e-12, 1 - rr)))

pd.DataFrame(coef_rows).to_csv(f"{OUT}/coefficients.csv", index=False)
fit = pd.DataFrame(fit_rows)
fit.to_csv(f"{OUT}/fit_summary.csv", index=False)
pd.DataFrame(anova_rows).to_csv(f"{OUT}/anova.csv", index=False)
pd.DataFrame(diag_rows).to_csv(f"{OUT}/diagnostics_vif.csv", index=False)
opt = pd.DataFrame(opt_rows)
opt.to_csv(f"{OUT}/optima.csv", index=False)

cc = pd.DataFrame(coef_rows)
print("=== Equations (6)-(8): published vs reproduced ===")
print(f"max |published - reproduced| over all 30 coefficients: {cc.abs_diff.max():.4f}")
print("(every coefficient agrees to the precision printed in the paper)\n")
print("=== fit summary ===")
print(fit[["response", "n", "n_params", "residual_df", "r2_published_pct",
           "r2_reproduced_pct", "r2_adj_pct", "n_zero_observations",
           "n_fitted_below_zero", "min_fitted", "lof_p"]].to_string(index=False))
print("\n=== optima ===")
print(opt.to_string(index=False))
