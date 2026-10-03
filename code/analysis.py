"""Turn the raw cross-validation and simulation output into the paper's result tables.

Produces, in results/:
  summary_loco.csv        pooled leave-one-condition-out metrics, every method x response
  summary_loo.csv         the same under leave-one-run-out (replicate leakage)
  headline_vs_published.csv   the proposed method against the published one, metric by
                              metric, with the win/loss tally that the abstract quotes
  rkf_tests.csv           Nadeau-Bengio paired tests over the repeated 5-fold folds
  ablation.csv            each CM-BARS component removed, change in every metric
  sim_summary.csv         simulation means by regime, method and metric
  sim_tests.csv           paired tests of the proposed method against the published one
                          across simulated datasets
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)
RES = f"{JD}/results"

from metrics import HIGHER_BETTER                                 # noqa: E402
from stats_tests import holm, nadeau_bengio                       # noqa: E402

PUBLISHED = "OLS-Quad (published)"
PROPOSED = os.environ.get("PROPOSED", "CM-BARS")
REPORT_METRICS = ["RMSE", "MAE", "MedAE", "MaxAbsErr", "R2", "Spearman",
                  "CRPS", "NLL", "Coverage90_err", "ZeroBalAcc", "FracNegative"]
SIM_HIGHER = {"TestRMSE": -1, "TestCRPS": -1, "TestNLL": -1, "SurfaceRMSE": -1,
              "OptDistance": -1, "OptRegret": -1, "FracNegative": -1,
              "TestCover90_err": -1}


def _pick_proposed(df):
    """Use the stacked model if it was run, otherwise plain CM-BARS."""
    if PROPOSED in set(df.method):
        return PROPOSED
    for cand in ["CM-BARS+ (stacked)", "CM-BARS", "PFN-RSM (prior-fitted)"]:
        if cand in set(df.method):
            return cand
    raise SystemExit("no proposed method found in results")


# ------------------------------------------------------------------ real data
def headline(pooled, proposed, protocol="LOCO"):
    """Metric-by-metric comparison of `proposed` with the published model."""
    p = pooled[pooled.protocol == protocol]
    rows = []
    for r in sorted(p.response.unique()):
        a = p[(p.response == r) & (p.method == PUBLISHED)]
        b = p[(p.response == r) & (p.method == proposed)]
        if not len(a) or not len(b):
            continue
        a, b = a.iloc[0], b.iloc[0]
        for m in REPORT_METRICS:
            if m not in p.columns or pd.isna(a[m]) or pd.isna(b[m]):
                continue
            s = HIGHER_BETTER[m]
            delta = s * (b[m] - a[m])
            rel = (b[m] - a[m]) / abs(a[m]) * 100 if abs(a[m]) > 1e-12 else np.nan
            rows.append(dict(protocol=protocol, response=r, metric=m,
                             published=a[m], proposed=b[m],
                             improvement=delta, rel_change_pct=rel,
                             better=bool(delta > 1e-12),
                             tie=bool(abs(delta) <= 1e-12)))
    return pd.DataFrame(rows)


def rkf_tests(perfold, proposed):
    """Paired Nadeau-Bengio tests per response and metric, Holm-corrected."""
    rows = []
    for r in sorted(perfold.response.unique()):
        g = perfold[perfold.response == r]
        a = g[g.method == PUBLISHED].set_index(["rep", "fold"])
        b = g[g.method == proposed].set_index(["rep", "fold"])
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        a, b = a.loc[common], b.loc[common]
        n_test = a.n_test.mean()
        n_train = a.n_train.mean()
        for m in REPORT_METRICS:
            if m not in a.columns:
                continue
            d = HIGHER_BETTER[m] * (b[m].values - a[m].values)
            d = d[np.isfinite(d)]
            if len(d) < 5:
                continue
            t, pnb = nadeau_bengio(d, n_train, n_test)
            try:
                pw = stats.wilcoxon(d, zero_method="wilcox").pvalue
            except Exception:
                pw = np.nan
            rows.append(dict(response=r, metric=m, n_folds=len(d),
                             published_mean=a[m].mean(), published_sd=a[m].std(ddof=1),
                             proposed_mean=b[m].mean(), proposed_sd=b[m].std(ddof=1),
                             delta=d.mean(), delta_sd=d.std(ddof=1),
                             wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                             t_nb=t, p_nb=pnb, p_wilcoxon=pw))
    R = pd.DataFrame(rows)
    if len(R):
        for r, g in R.groupby("response"):
            R.loc[g.index, "p_nb_holm"] = holm(g.p_nb.values)
        R["signif_win"] = (R.delta > 0) & (R.p_nb_holm < 0.05)
        R["signif_loss"] = (R.delta < 0) & (R.p_nb_holm < 0.05)
    return R


def ablation(pooled, proposed, protocol="LOCO", base_name="CM-BARS"):
    """Change in each metric when one CM-BARS component is removed.

    The reference is CM-BARS itself, not the stacked pool: these are ablations *of*
    CM-BARS, so the question is what each component contributes to it.
    """
    p = pooled[pooled.protocol == protocol]
    base = p[p.method == base_name]
    rows = []
    for m in sorted(set(p.method)):
        if not (m.startswith("CM-BARS") or m in ("Tobit-MT-Bayes",)):
            continue
        if m == base_name:
            continue
        for r in sorted(p.response.unique()):
            b = base[base.response == r]
            v = p[(p.method == m) & (p.response == r)]
            if not len(b) or not len(v):
                continue
            b, v = b.iloc[0], v.iloc[0]
            rows.append(dict(variant=m, response=r,
                             RMSE=v.RMSE, dRMSE=v.RMSE - b.RMSE,
                             CRPS=v.CRPS, dCRPS=v.CRPS - b.CRPS,
                             R2=v.R2, dR2=v.R2 - b.R2,
                             FracNegative=v.FracNegative))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ simulation
def sim_summary(sim):
    sim = sim.copy()
    if "TestCover90" in sim:
        sim["TestCover90_err"] = (sim.TestCover90 - 0.90).abs()
    mets = [m for m in SIM_HIGHER if m in sim.columns]
    g = sim.groupby(["regime", "method", "response"])[mets]
    out = g.agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def sim_tests(sim, proposed):
    sim = sim.copy()
    if "TestCover90" in sim:
        sim["TestCover90_err"] = (sim.TestCover90 - 0.90).abs()
    rows = []
    for (rg, r), g in sim.groupby(["regime", "response"]):
        a = g[g.method == PUBLISHED].set_index("dataset")
        b = g[g.method == proposed].set_index("dataset")
        common = a.index.intersection(b.index)
        if len(common) < 10:
            continue
        a, b = a.loc[common], b.loc[common]
        for m in SIM_HIGHER:
            if m not in a.columns:
                continue
            d = SIM_HIGHER[m] * (b[m].values - a[m].values)
            d = d[np.isfinite(d)]
            if len(d) < 10 or np.allclose(d, 0):
                continue
            t, pv = stats.ttest_rel(np.zeros_like(d), -d)
            try:
                pw = stats.wilcoxon(d).pvalue
            except Exception:
                pw = np.nan
            rows.append(dict(regime=rg, response=r, metric=m, n=len(d),
                             published_mean=a[m].mean(), proposed_mean=b[m].mean(),
                             delta=d.mean(), wins=int((d > 0).sum()),
                             losses=int((d < 0).sum()), p_t=pv, p_wilcoxon=pw))
    R = pd.DataFrame(rows)
    if len(R):
        for k, g in R.groupby(["regime", "response"]):
            R.loc[g.index, "p_holm"] = holm(g.p_wilcoxon.values)
        R["signif_win"] = (R.delta > 0) & (R.p_holm < 0.05)
        R["signif_loss"] = (R.delta < 0) & (R.p_holm < 0.05)
    return R


def main():
    out = {}
    if os.path.exists(f"{RES}/cv_pooled.csv"):
        pooled = pd.read_csv(f"{RES}/cv_pooled.csv")
        proposed = _pick_proposed(pooled)
        print(f"proposed method = {proposed}")
        for proto, fn in [("LOCO", "summary_loco.csv"), ("LOO", "summary_loo.csv")]:
            s = pooled[pooled.protocol == proto]
            if len(s):
                s.to_csv(f"{RES}/{fn}", index=False)
        hl = pd.concat([headline(pooled, proposed, p)
                        for p in pooled.protocol.unique()], ignore_index=True)
        hl.to_csv(f"{RES}/headline_vs_published.csv", index=False)
        out["headline"] = hl
        ab = ablation(pooled, proposed)
        ab.to_csv(f"{RES}/ablation.csv", index=False)

        loco = hl[hl.protocol == "LOCO"]
        w = int(loco.better.sum())
        n = len(loco)
        print(f"\nLOCO: {proposed} beats the published model on {w}/{n} "
              f"metric-response pairs ({100*w/n:.1f}%)")
        print(loco.pivot_table(index="metric", columns="response",
                               values="improvement").round(3).to_string())

    if os.path.exists(f"{RES}/cv_perfold.csv"):
        pf = pd.read_csv(f"{RES}/cv_perfold.csv")
        if len(pf):
            proposed = _pick_proposed(pf)
            t = rkf_tests(pf, proposed)
            t.to_csv(f"{RES}/rkf_tests.csv", index=False)
            if len(t):
                print(f"\nRKF: {int(t.signif_win.sum())} significant wins, "
                      f"{int(t.signif_loss.sum())} significant losses "
                      f"(Holm, {len(t)} tests)")

    if os.path.exists(f"{RES}/simulation.csv"):
        sim = pd.read_csv(f"{RES}/simulation.csv")
        if len(sim):
            proposed = _pick_proposed(sim)
            sim_summary(sim).to_csv(f"{RES}/sim_summary.csv", index=False)
            st = sim_tests(sim, proposed)
            st.to_csv(f"{RES}/sim_tests.csv", index=False)
            if len(st):
                print(f"\nSimulation: {int(st.signif_win.sum())} significant wins, "
                      f"{int(st.signif_loss.sum())} significant losses of {len(st)} tests")
    return out


if __name__ == "__main__":
    main()
