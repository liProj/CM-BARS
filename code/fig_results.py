"""Figures 7-14: cross-validation results, significance, calibration, ablation, optima."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)
RES = f"{JD}/results"

import methods as M                                               # noqa: E402
import plotstyle as PS                                            # noqa: E402
from metrics import HIGHER_BETTER                                 # noqa: E402

PS.setup()
R3 = M.RESPONSES
PUB = "OLS-Quad (published)"


def _proposed(df):
    for c in ["CM-BARS+ (stacked)", "CM-BARS"]:
        if c in set(df.method):
            return c
    return sorted(set(df.method))[0]


def _order(p):
    """Methods ordered by mean RMSE across responses, best first."""
    return (p.groupby("method").RMSE.mean().sort_values().index.tolist())


# ---------------------------------------------------------------- fig 7
def fig07():
    pooled = pd.read_csv(f"{RES}/cv_pooled.csv")
    p = pooled[pooled.protocol == "LOCO"]
    order = _order(p)[::-1]
    prop = _proposed(p)
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 5.4), sharey=True)
    for k, (metric, lab) in enumerate([("RMSE", "RMSE (% points)"),
                                       ("CRPS", "CRPS (% points)"),
                                       ("R2", "out-of-sample $R^2$"),
                                       ("NLL", "negative log score")]):
        ax = axes[k]
        y = np.arange(len(order))
        for i, r in enumerate(R3):
            v = [p[(p.method == m) & (p.response == r)][metric]
                 for m in order]
            v = [float(x.iloc[0]) if len(x) else np.nan for x in v]
            ax.barh(y + (i - 1) * 0.27, v, height=0.26, color=PS.RESP_COL[r],
                    label=PS.RESP_LAB[r] if k == 0 else None)
        ax.set_yticks(y)
        if k == 0:
            lbl = [("$\\bf{" + m.replace(' ', '\\ ').replace('+', '{+}') + "}$")
                   if m in (prop, PUB) else m for m in order]
            ax.set_yticklabels(order, fontsize=7)
            for t, m in zip(ax.get_yticklabels(), order):
                if m == prop:
                    t.set_color("#0B6E4F"); t.set_fontweight("bold")
                if m == PUB:
                    t.set_color("#B03A2E"); t.set_fontweight("bold")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_xlabel(lab)
        ax.set_title(f"({'abcd'[k]}) {metric}", fontsize=9)
        if metric == "R2":
            ax.axvline(0, color="0.4", lw=0.8)
            ax.set_xlim(-2.2, 1.0)        # a few deliberately handicapped variants
            ax.annotate("axis clipped at $-2$", (-2.15, 0.2), fontsize=6, color="0.35")
        if metric == "NLL":
            ax.set_xlim(0, min(8.0, ax.get_xlim()[1]))
    fig.suptitle("Leave-one-condition-out performance on the published 15-run design "
                 "(13 folds, pooled out-of-fold predictions)", fontsize=10, y=1.0)
    PS.save(fig, "fig07_loco_methods")


# ---------------------------------------------------------------- fig 8
def fig08():
    pooled = pd.read_csv(f"{RES}/cv_pooled.csv")
    if set(pooled.protocol) < {"LOCO", "LOO"}:
        print("  (skipping fig08: both protocols needed)")
        return
    a = pooled[pooled.protocol == "LOCO"].set_index(["method", "response"]).RMSE
    b = pooled[pooled.protocol == "LOO"].set_index(["method", "response"]).RMSE
    common = a.index.intersection(b.index)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    ax = axes[0]
    for r in R3:
        idx = [i for i in common if i[1] == r]
        ax.plot([b[i] for i in idx], [a[i] for i in idx], "o", ms=6,
                color=PS.RESP_COL[r], label=PS.RESP_LAB[r])
    lim = [0, max(a.max(), b.max()) * 1.08]
    ax.plot(lim, lim, "-", color="0.6", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("RMSE, leave-one-RUN-out (replicates leak)")
    ax.set_ylabel("RMSE, leave-one-CONDITION-out")
    ax.legend(fontsize=7)
    ax.set_title("(a) Leaving one centre replicate out\nflatters every method", fontsize=9)

    ax = axes[1]
    diff = (pd.DataFrame({"LOCO": a[common], "LOO": b[common]})
            .assign(infl=lambda x: (x.LOCO - x.LOO) / x.LOO * 100)
            .reset_index())
    order = diff.groupby("method").infl.mean().sort_values().index.tolist()
    ax.barh(np.arange(len(order)),
            [diff[diff.method == m].infl.mean() for m in order],
            color=["#B03A2E" if m == PUB else "#0B6E4F" if "CM-BARS+" in m else "#2E86C1"
                   for m in order])
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(order, fontsize=6.5)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("RMSE understated by LOO (%)")
    ax.set_title("(b) Size of the replicate-leakage\noptimism, per method", fontsize=9)
    PS.save(fig, "fig08_protocol_leakage")


# ---------------------------------------------------------------- fig 9
def fig09():
    hl = pd.read_csv(f"{RES}/headline_vs_published.csv")
    hl = hl[hl.protocol == "LOCO"]
    if not len(hl):
        return
    mets = [m for m in ["RMSE", "MAE", "MedAE", "MaxAbsErr", "R2", "Spearman", "CRPS",
                        "NLL", "Coverage90_err", "ZeroBalAcc", "FracNegative"]
            if m in set(hl.metric)]
    M_ = np.full((len(mets), 3), np.nan)
    T_ = np.empty((len(mets), 3), dtype=object)
    for i, m in enumerate(mets):
        for j, r in enumerate(R3):
            s = hl[(hl.metric == m) & (hl.response == r)]
            if not len(s):
                T_[i, j] = ""
                continue
            s = s.iloc[0]
            sgn = HIGHER_BETTER[m]
            delta = sgn * (s.proposed - s.published)
            denom = abs(s.published) if abs(s.published) > 1e-9 else np.nan
            if np.isfinite(denom):
                M_[i, j] = np.clip(delta / denom * 100, -120, 120)
            elif abs(delta) <= 1e-12:
                M_[i, j] = 0.0                  # a genuine tie (both exactly zero)
            else:
                M_[i, j] = 100.0 if delta > 0 else -100.0
            T_[i, j] = f"{s.published:.3g}\n$\\rightarrow${s.proposed:.3g}"
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    im = ax.imshow(M_, cmap=PS.DIV, vmin=-100, vmax=100, aspect="auto")
    for i in range(len(mets)):
        for j in range(3):
            if T_[i, j]:
                ax.text(j, i, T_[i, j], ha="center", va="center", fontsize=6.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([PS.RESP_LAB[r].replace(" ", "\n") for r in R3], fontsize=8)
    ax.set_yticks(range(len(mets))); ax.set_yticklabels(mets, fontsize=8)
    ax.grid(False)
    nwin = int(hl.better.sum()); ntot = len(hl)
    ax.set_title(f"Proposed method vs the published quadratic, LOCO\n"
                 f"{nwin}/{ntot} metric-response pairs improved "
                 f"({100*nwin/ntot:.0f}%)", fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("relative improvement (%), capped at 120", fontsize=7)
    PS.save(fig, "fig09_headline_heatmap")


# ---------------------------------------------------------------- fig 10
def fig10():
    f = f"{RES}/cv_perfold.csv"
    if not os.path.exists(f):
        print("  (skipping fig10: no repeated k-fold yet)")
        return
    pf = pd.read_csv(f)
    prop = _proposed(pf)
    tests = pd.read_csv(f"{RES}/rkf_tests.csv") if os.path.exists(f"{RES}/rkf_tests.csv") \
        else pd.DataFrame()
    mets = ["RMSE", "CRPS", "NLL", "R2", "Coverage90_err"]
    fig, axes = plt.subplots(1, len(mets), figsize=(12.2, 3.6))
    for k, m in enumerate(mets):
        ax = axes[k]
        for j, r in enumerate(R3):
            a = pf[(pf.method == PUB) & (pf.response == r)].set_index(["rep", "fold"])
            b = pf[(pf.method == prop) & (pf.response == r)].set_index(["rep", "fold"])
            idx = a.index.intersection(b.index)
            if not len(idx) or m not in a:
                continue
            d = HIGHER_BETTER[m] * (b.loc[idx, m].values - a.loc[idx, m].values)
            d = d[np.isfinite(d)]
            parts = ax.violinplot([d], positions=[j], widths=0.75, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(PS.RESP_COL[r]); pc.set_alpha(0.45)
            ax.plot([j], [d.mean()], "o", color=PS.RESP_COL[r], ms=5)
            if len(tests):
                t = tests[(tests.metric == m) & (tests.response == r)]
                if len(t):
                    t = t.iloc[0]
                    star = "***" if t.p_nb_holm < 0.001 else "**" if t.p_nb_holm < 0.01 \
                        else "*" if t.p_nb_holm < 0.05 else "n.s."
                    ax.annotate(f"{star}\n{t.wins}/{t.wins+t.losses}",
                                (j, np.quantile(d, 0.97)), ha="center", fontsize=6.5)
        ax.axhline(0, color="0.4", lw=0.9)
        ax.set_xticks(range(3))
        ax.set_xticklabels([PS.RESP_LAB[r].split()[0] for r in R3], fontsize=7)
        ax.set_title(m, fontsize=9)
        if k == 0:
            ax.set_ylabel("improvement over the published model\n(per fold, positive = better)")
    fig.suptitle(f"{prop} minus the published quadratic, repeated grouped 5-fold; "
                 "stars are Holm-corrected Nadeau-Bengio tests", fontsize=9.5, y=1.03)
    PS.save(fig, "fig10_perfold_differences")


# ---------------------------------------------------------------- fig 11
def fig11():
    pooled = pd.read_csv(f"{RES}/cv_pooled.csv")
    p = pooled[pooled.protocol == "LOCO"]
    show = [m for m in [PUB, "OLS-Quad + clip", "Tobit-Quad", "ExtraTrees", "GP-Matern",
                        "CM-BARS", "PFN-RSM (prior-fitted)", "CM-BARS+ (stacked)"]
            if m in set(p.method)]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4))
    ax = axes[0]
    for m in show:
        v = [float(p[(p.method == m) & (p.response == r)].Coverage90.iloc[0])
             for r in R3 if len(p[(p.method == m) & (p.response == r)])]
        ax.plot(range(len(v)), v, "o-", ms=5, color=PS.col(m), label=m, lw=1.4)
    ax.axhline(0.90, color="#B03A2E", ls="--", lw=1.1)
    ax.annotate("nominal 90%", (2.0, 0.915), fontsize=7, color="#B03A2E", ha="right")
    ax.set_xticks(range(3))
    ax.set_xticklabels([PS.RESP_LAB[r].split()[0] for r in R3], fontsize=7)
    ax.set_ylabel("empirical coverage of the 90% interval")
    ax.set_title("(a) Interval calibration", fontsize=9)
    ax.legend(fontsize=6, ncol=1)

    ax = axes[1]
    for m in show:
        v = [float(p[(p.method == m) & (p.response == r)].IntervalWidth90.iloc[0])
             for r in R3 if len(p[(p.method == m) & (p.response == r)])]
        ax.plot(range(len(v)), v, "o-", ms=5, color=PS.col(m), lw=1.4)
    ax.set_xticks(range(3))
    ax.set_xticklabels([PS.RESP_LAB[r].split()[0] for r in R3], fontsize=7)
    ax.set_ylabel("mean 90% interval width (% points)")
    ax.set_title("(b) Sharpness", fontsize=9)

    ax = axes[2]
    y = np.arange(len(show))
    fn = [p[p.method == m].FracNegative.mean() * 100 for m in show]
    ax.barh(y, fn, color=[PS.col(m) for m in show])
    ax.set_yticks(y); ax.set_yticklabels(show, fontsize=6.5)
    ax.set_xlabel("out-of-fold predictions below 0% (%)")
    ax.set_title("(c) Physically impossible\npredictions", fontsize=9)
    fig.suptitle("Calibration, sharpness and physical validity (leave-one-condition-out)",
                 fontsize=9.5, y=1.03)
    PS.save(fig, "fig11_calibration")


# ---------------------------------------------------------------- fig 12
def fig12():
    f = f"{RES}/final/insample_fit.csv"
    if not os.path.exists(f):
        print("  (skipping fig12: run final_fit.py first)")
        return
    ins = pd.read_csv(f)
    d = pd.read_csv(f"{JD}/data/bbd_design_responses.csv")
    coefs = pd.read_csv(f"{RES}/baseline/coefficients.csv")
    TERMS = ["const", "pH", "conc", "cnf", "pH2", "conc2", "cnf2", "pH:conc", "pH:cnf",
             "conc:cnf"]
    pp, cc, ff = d.pH.values, d.additive_mM.values, d.cnf_pct.values
    X = np.column_stack([np.ones_like(pp, float), pp, cc, ff, pp * pp, cc * cc, ff * ff,
                         pp * cc, pp * ff, cc * ff])
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6))
    for j, r in enumerate(R3):
        ax = axes[j]
        s = ins[ins.response == r].sort_values("run")
        b = coefs[coefs.response == r].set_index("term").reproduced[TERMS].values
        pubfit = X @ b
        x = np.arange(15)
        ax.axhline(0, color="0.5", lw=0.9)
        ax.fill_between(x, s.lo90, s.hi90, color="#0B6E4F", alpha=0.20,
                        label="proposed 90% predictive band")
        ax.plot(x, s.stack_mean, "-", color="#0B6E4F", lw=1.6, label="proposed mean")
        ax.plot(x, pubfit, "--", color="#B03A2E", lw=1.4, label="published quadratic")
        obs = d[r].values
        at0 = obs == 0
        ax.plot(x[~at0], obs[~at0], "o", ms=5, color="black", label="observed")
        ax.plot(x[at0], obs[at0], "v", ms=7, color="#B03A2E", label="observed zero")
        ax.set_xticks(x[::2]); ax.set_xticklabels((x + 1)[::2], fontsize=7)
        ax.set_xlabel("Run")
        if j == 0:
            ax.set_ylabel("Adsorbed phenols (%)")
            ax.legend(fontsize=6)
        ax.set_title(PS.RESP_LAB[r], fontsize=9)
    fig.suptitle("The proposed model never leaves [0, 100] and places explicit probability "
                 "mass on 'no adsorption'", fontsize=9.5, y=1.02)
    PS.save(fig, "fig12_predictive_bands")


# ---------------------------------------------------------------- fig 13
def fig13():
    f = f"{RES}/ablation.csv"
    if not os.path.exists(f):
        return
    ab = pd.read_csv(f)
    if not len(ab):
        return
    ab = ab[ab.variant != "CM-BARS+ (stacked)"]
    order = ab.groupby("variant").dRMSE.mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for k, c in enumerate(["dRMSE", "dCRPS"]):
        ax = axes[k]
        y = np.arange(len(order))
        for i, r in enumerate(R3):
            v = [float(ab[(ab.variant == m) & (ab.response == r)][c].iloc[0])
                 if len(ab[(ab.variant == m) & (ab.response == r)]) else np.nan
                 for m in order]
            ax.barh(y + (i - 1) * 0.27, v, height=0.25, color=PS.RESP_COL[r],
                    label=PS.RESP_LAB[r] if k == 0 else None)
        mean = [float(ab[ab.variant == m][c].mean()) for m in order]
        ax.plot(mean, y, "D", ms=5, color="black", label="mean" if k == 0 else None)
        ax.axvline(0, color="0.3", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(order if k == 0 else [], fontsize=7.5)
        ax.set_xlabel(f"change in {c[1:]} when the component is removed\n"
                      "(positive = the component helps CM-BARS)")
        ax.set_title(f"({'ab'[k]}) {c[1:]}", fontsize=9)
        if k == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Ablation of CM-BARS, relative to the full model, by compound "
                 "(leave-one-condition-out)", fontsize=9.5, y=1.02)
    PS.save(fig, "fig13_ablation")


# ---------------------------------------------------------------- fig 14
def fig14():
    f = f"{RES}/final/optima_draws.npz"
    if not os.path.exists(f):
        print("  (skipping fig14: run final_fit.py first)")
        return
    dr = np.load(f)
    pub = pd.read_csv(f"{JD}/data/published_r2_optima.csv")
    post = pd.read_csv(f"{RES}/final/optima_posterior.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.6))
    for j, r in enumerate(R3):
        ax = axes[j]
        a = dr[r]
        ax.scatter(a[:, 0] + np.random.default_rng(j).normal(0, .04, len(a)),
                   a[:, 2] + np.random.default_rng(j + 9).normal(0, .003, len(a)),
                   s=5, alpha=0.10, color=PS.RESP_COL[r], edgecolors="none")
        p = pub[pub.response == r].iloc[0]
        ax.plot(p.opt_pH, p.opt_cnf_pct, marker="*", ms=17, mfc="#B03A2E", mec="white",
                mew=1.0, zorder=5, label="published optimum (Table 2)")
        q = post[post.response == r].iloc[0]
        ax.plot(q.post_pH_mean, q.post_cnf_mean, marker="D", ms=8, mfc="#0B6E4F",
                mec="white", zorder=5, label="posterior mean optimum")
        ax.plot([q.post_pH_q025, q.post_pH_q975], [q.post_cnf_mean] * 2, "-",
                color="#0B6E4F", lw=2, zorder=4)
        ax.plot([q.post_pH_mean] * 2, [q.post_cnf_q025, q.post_cnf_q975], "-",
                color="#0B6E4F", lw=2, zorder=4)
        ax.set_xlim(3.8, 8.2); ax.set_ylim(0.085, 0.415)
        ax.set_xlabel("pH")
        if j == 0:
            ax.set_ylabel("Final CNF conc. (%)")
            ax.legend(fontsize=6.5, loc="upper left")
        ax.set_title(PS.RESP_LAB[r], fontsize=9)
    fig.suptitle("Posterior distribution of the optimum (2000 draws) against the "
                 "single point optimum the paper reports", fontsize=9.5, y=1.02)
    PS.save(fig, "fig14_optimum_posterior")


if __name__ == "__main__":
    for f in [fig07, fig08, fig09, fig10, fig11, fig12, fig13, fig14]:
        try:
            f()
        except Exception as e:
            print(f"  {f.__name__} failed: {type(e).__name__}: {e}")
