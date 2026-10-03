"""Fit the proposed model to all fifteen runs and save everything the paper quotes.

Outputs (results/final/):
  posterior_summary.csv    posterior mean / sd / 95% interval of every identified parameter
  mcmc_diagnostics.csv     conventional Rhat and effective sample size
  grid_predictions.npz     posterior-mean surface and 90% band on a dense grid
  optima_posterior.csv     posterior distribution of the optimum, per compound, against the
                           three point optima of the paper's Table 2
  stack_weights.csv        the pooling weights chosen for CM-BARS+
  distilled.csv            the closed-form distilled equation, printable like Eq. (6)-(8)
  insample_fit.csv         fitted vs observed for the proposed model on all 15 runs
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)
OUT = f"{JD}/results/final"
os.makedirs(OUT, exist_ok=True)

import methods as M                                               # noqa: E402

RESPONSES = M.RESPONSES
GRID = 21        # 21^3 = 9261 points; 41^3 x 12000 draws exhausts memory


def grid_points(m=GRID):
    g = np.linspace(-1, 1, m)
    A, B, C = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack([A.ravel(), B.ravel(), C.ravel()]), g


def main():
    from cmbars2 import CMBARS2

    d = pd.read_csv(f"{JD}/data/bbd_design_responses.csv")
    Z = M.to_coded(d.pH, d.additive_mM, d.cnf_pct)
    Y = d[RESPONSES].values.astype(float)

    # ---------------- CM-BARS on all 15 runs ----------------
    print("fitting CM-BARS on all 15 runs ...", flush=True)
    cb = CMBARS2(use_gp=False, warmup=3000, samples=3000, chains=4)
    cb.fit(Z, Y)
    dg = pd.DataFrame(cb.diagnostics())
    dg.to_csv(f"{OUT}/mcmc_diagnostics.csv", index=False)
    dg["base"] = dg.param.str.replace(r"\[\d+\]", "", regex=True)
    ident = dg[dg.base.isin(["theta", "u", "b0", "conc_", "logit_pi", "z0", "psi",
                             "vload", "s_grp", "kappa"])]
    print(f"  max Rhat {dg.r_hat.max():.4f} (identified: {ident.r_hat.max():.4f}), "
          f"min ESS {dg.ess.min():.0f} (identified: {ident.ess.min():.0f})")

    rows = []
    for k, v in cb.post_.items():
        v = np.asarray(v)
        flat = v.reshape(v.shape[0], -1)
        for i in range(flat.shape[1]):
            x = flat[:, i]
            rows.append(dict(param=f"{k}[{i}]" if flat.shape[1] > 1 else k,
                             mean=x.mean(), sd=x.std(ddof=1),
                             q025=np.quantile(x, 0.025), q500=np.quantile(x, 0.5),
                             q975=np.quantile(x, 0.975)))
    pd.DataFrame(rows).to_csv(f"{OUT}/posterior_summary.csv", index=False)

    # ---------------- CM-BARS+ : members on all 15 runs, pooled with the
    # weights selected during cross-validation (averaged over the LOCO folds), with the cached-procedure limitations
    # described in docs/MODEL.md ----------------
    print("fitting the CM-BARS+ members on all 15 runs ...", flush=True)
    from stack import default_members
    wf = f"{JD}/results/stack_weights_perfold.csv"
    if os.path.exists(wf):
        W = (pd.read_csv(wf).groupby(["response", "member"]).weight.mean()
             .unstack(fill_value=0.0))
        W = W.div(W.sum(axis=1), axis=0)
        weights = {r: {m: float(W.loc[r, m]) for m in W.columns} for r in W.index}
    else:
        print("  (no cross-validated weights found; falling back to equal weights)")
        weights = None

    members = {
        "OLS-Quad + clip": M.OLSQuadClip, "PLS-Quad": M.PLSQuad,
        "ElasticNet-Quad": M.ENetQuad, "RandomForest": M.RandomForest,
        "ExtraTrees": M.ExtraTrees, "LightGBM": M.LightGBM, "XGBoost": M.XGBoost,
        "GP-Matern": M.GPMatern,
    }
    fitted = {}
    for nm, ctor in members.items():
        mdl = ctor()
        mdl.fit(Z, Y)
        fitted[nm] = mdl
    fitted["CM-BARS"] = cb
    if os.path.exists(f"{JD}/results/pfn_rsm.pt"):
        from pfn import PFNRSM
        p_ = PFNRSM()
        p_.fit(Z, Y)
        fitted["PFN-RSM (prior-fitted)"] = p_
    if weights is None:
        weights = {r: {m: 1.0 / len(fitted) for m in fitted} for r in RESPONSES}

    def pool(Zq, n_draw=2000, seed=0):
        preds = {nm: m.predict(Zq) for nm, m in fitted.items()}
        rng = np.random.default_rng(seed)
        out = {}
        for r in RESPONSES:
            w = {nm: weights.get(r, {}).get(nm, 0.0) for nm in fitted}
            tot = sum(w.values())
            if tot <= 0:
                w = {nm: 1.0 / len(fitted) for nm in fitted}
                tot = 1.0
            names = [nm for nm in fitted if w[nm] > 1e-6]
            wv = np.array([w[nm] for nm in names]) / sum(w[nm] for nm in names)
            counts = rng.multinomial(n_draw, wv)
            parts = []
            for nm, c in zip(names, counts):
                if c <= 0:
                    continue
                S = np.asarray(preds[nm][r]["samples"], float)
                parts.append(S[rng.choice(S.shape[0], size=c, replace=c > S.shape[0])])
            sm = np.concatenate(parts, axis=0)
            out[r] = dict(mean=sm.mean(axis=0), sd=sm.std(axis=0, ddof=1), samples=sm)
        return out

    wrows = [dict(response=r, member=k, weight=v)
             for r, w in weights.items() for k, v in w.items()]
    pd.DataFrame(wrows).to_csv(f"{OUT}/stack_weights.csv", index=False)
    print(pd.DataFrame(wrows).pivot_table(index="member", columns="response",
                                          values="weight").round(3).to_string())

    # ---------------- distillation of the pooled surface ----------------
    print("distilling ...", flush=True)
    TERMS = ["const", "pH", "conc", "cnf", "pH2", "conc2", "cnf2", "pH:conc", "pH:cnf",
             "conc:cnf"]
    Zg11, _ = grid_points(11)
    pr = pool(Zg11, n_draw=1200, seed=1)
    X11 = M.quad_basis(Zg11)
    coef_, sd_, distill_rmse_ = {}, {}, {}
    for r in RESPONSES:
        mu = np.clip(pr[r]["mean"] / 100.0, 1e-4, 1 - 1e-4)
        b, *_ = np.linalg.lstsq(X11, np.log(mu / (1 - mu)), rcond=None)
        coef_[r] = b
        back = 100.0 / (1.0 + np.exp(-(X11 @ b)))
        distill_rmse_[r] = float(np.sqrt(np.mean((back - pr[r]["mean"]) ** 2)))
        sd_[r] = float(np.mean(pr[r]["sd"]))
    drows = [dict(response=r, term=t, coef=float(coef_[r][k]),
                  distill_rmse_pct=distill_rmse_[r])
             for r in RESPONSES for k, t in enumerate(TERMS)]
    pd.DataFrame(drows).to_csv(f"{OUT}/distilled.csv", index=False)
    with open(f"{JD}/results/distilled_coefficients.json", "w") as fh:
        json.dump({r: dict(coef=coef_[r].tolist(), sd=sd_[r],
                           distill_rmse=distill_rmse_[r]) for r in RESPONSES},
                  fh, indent=1)
    print("  distillation RMSE (% points):",
          {r: round(distill_rmse_[r], 3) for r in RESPONSES})

    # ---------------- dense surfaces and the posterior optimum ----------------
    print("evaluating surfaces on the dense grid ...", flush=True)
    Zg, axis = grid_points()
    pg_stack = pool(Zg, n_draw=400, seed=2)
    pg_cb = cb.predict(Zg, n_draw=400, max_post=600)
    np.savez_compressed(
        f"{OUT}/grid_predictions.npz", axis=axis, Zg=Zg,
        **{f"stack_mean_{r}": pg_stack[r]["mean"] for r in RESPONSES},
        **{f"stack_sd_{r}": pg_stack[r]["sd"] for r in RESPONSES},
        **{f"cmbars_mean_{r}": pg_cb[r]["mean"] for r in RESPONSES},
        **{f"cmbars_sd_{r}": pg_cb[r]["sd"] for r in RESPONSES},
        **{f"distilled_mean_{r}": 100.0 / (1 + np.exp(-(M.quad_basis(Zg) @ coef_[r])))
           for r in RESPONSES})

    # posterior over the optimum: argmax of each posterior draw of the latent surface
    print("posterior of the optimum ...", flush=True)
    Zc, _ = grid_points(21)
    u = np.asarray(cb._u_star(Zc, max_post=1500))                                # (S, m, R)
    pub = pd.read_csv(f"{JD}/data/published_r2_optima.csv")
    orows, draws = [], {}
    for j, r in enumerate(RESPONSES):
        k = u[:, :, j].argmax(axis=1)
        Zopt = Zc[k]
        real = M.from_coded(Zopt)
        draws[r] = real
        p = pub[pub.response == r].iloc[0]
        orows.append(dict(
            response=r,
            published_pH=p.opt_pH, published_conc=p.opt_conc_mM, published_cnf=p.opt_cnf_pct,
            post_pH_mean=real[:, 0].mean(), post_pH_q025=np.quantile(real[:, 0], .025),
            post_pH_q975=np.quantile(real[:, 0], .975),
            post_conc_mean=real[:, 1].mean(), post_conc_q025=np.quantile(real[:, 1], .025),
            post_conc_q975=np.quantile(real[:, 1], .975),
            post_cnf_mean=real[:, 2].mean(), post_cnf_q025=np.quantile(real[:, 2], .025),
            post_cnf_q975=np.quantile(real[:, 2], .975),
            pct_draws_at_conc_lower_bound=float((real[:, 1] <= 5.0 + 1e-9).mean()),
            # a 95% interval is a poor summary of a posterior that piles up at opposite
            # corners of the design box, so also record where the mass actually is
            frac_near_published=float(((np.abs(real[:, 0] - p.opt_pH) <= 0.5)
                                       & (np.abs(real[:, 2] - p.opt_cnf_pct) <= 0.05)).mean()),
            frac_pH_below_six=float((real[:, 0] < 6.0).mean()),
            frac_pH_at_bound=float(((real[:, 0] <= 4.01) | (real[:, 0] >= 7.99)).mean()),
            published_inside_95=bool(
                np.quantile(real[:, 0], .025) <= p.opt_pH <= np.quantile(real[:, 0], .975)
                and np.quantile(real[:, 2], .025) <= p.opt_cnf_pct
                <= np.quantile(real[:, 2], .975))))
    pd.DataFrame(orows).to_csv(f"{OUT}/optima_posterior.csv", index=False)
    np.savez_compressed(f"{OUT}/optima_draws.npz",
                        **{r: draws[r] for r in RESPONSES})
    print(pd.DataFrame(orows)[["response", "published_pH", "post_pH_mean",
                               "post_pH_q025", "post_pH_q975",
                               "published_inside_95"]].round(2).to_string(index=False))

    # ---------------- in-sample fit of the proposed model ----------------
    pin = pool(Z, n_draw=2000, seed=3)
    irows = []
    for j, r in enumerate(RESPONSES):
        for i in range(len(d)):
            irows.append(dict(run=int(d.run[i]), response=r, observed=Y[i, j],
                              stack_mean=pin[r]["mean"][i], stack_sd=pin[r]["sd"][i],
                              lo90=np.quantile(pin[r]["samples"][:, i], 0.05),
                              hi90=np.quantile(pin[r]["samples"][:, i], 0.95)))
    pd.DataFrame(irows).to_csv(f"{OUT}/insample_fit.csv", index=False)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
