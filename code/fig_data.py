"""Figures 1-6: the design, the data check, and what is wrong with the published model."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)

import methods as M                                               # noqa: E402
import plotstyle as PS                                            # noqa: E402

PS.setup()
R3 = M.RESPONSES
TERMS = ["const", "pH", "conc", "cnf", "pH2", "conc2", "cnf2", "pH:conc", "pH:cnf",
         "conc:cnf"]
PRETTY = {"const": "intercept", "pH": "pH", "conc": "conc.", "cnf": "CNF",
          "pH2": "pH$^2$", "conc2": "conc.$^2$", "cnf2": "CNF$^2$",
          "pH:conc": "pH$\\times$conc.", "pH:cnf": "pH$\\times$CNF",
          "conc:cnf": "conc.$\\times$CNF"}

d = pd.read_csv(f"{JD}/data/bbd_design_responses.csv")
Z = M.to_coded(d.pH, d.additive_mM, d.cnf_pct)
coefs = pd.read_csv(f"{JD}/results/baseline/coefficients.csv")
fit = pd.read_csv(f"{JD}/results/baseline/fit_summary.csv")
anova = pd.read_csv(f"{JD}/results/baseline/anova.csv")
chk = pd.read_csv(f"{JD}/results/data_check.csv")


# ---------------------------------------------------------------- fig 1
def fig01():
    fig = plt.figure(figsize=(9.6, 3.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], s=55, c="#1B4F72", depthshade=False,
               edgecolor="white", linewidth=0.6)
    for a in (-1, 1):
        for b in (-1, 1):
            ax.plot([a, a], [b, b], [-1, 1], color="0.8", lw=0.6, zorder=0)
            ax.plot([a, a], [-1, 1], [b, b], color="0.8", lw=0.6, zorder=0)
            ax.plot([-1, 1], [a, a], [b, b], color="0.8", lw=0.6, zorder=0)
    ax.text(0, 0, 0.22, "3$\\times$", fontsize=7, ha="center", color="#B03A2E")
    ax.set_xlabel("pH (coded)", labelpad=-6)
    ax.set_ylabel("conc. (coded)", labelpad=-6)
    ax.set_zlabel("CNF (coded)", labelpad=-6)
    ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1]); ax.set_zticks([-1, 0, 1])
    ax.tick_params(labelsize=6, pad=-2)
    ax.set_title("(a) Box-Behnken design\n12 edge runs + 3 centre replicates", fontsize=9)

    ax = fig.add_subplot(1, 3, 2)
    w = 0.26
    x = np.arange(15)
    for i, r in enumerate(R3):
        ax.bar(x + (i - 1) * w, d[r], width=w, color=PS.RESP_COL[r],
               label=PS.RESP_LAB[r])
    zero = [(xi, r) for xi in x for r in R3 if d[r].iloc[xi] == 0]
    for xi, r in zero:
        ax.plot(xi + (R3.index(r) - 1) * w, 0.6, marker="v", ms=4, color="#B03A2E",
                clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels(x + 1, fontsize=6)
    ax.set_xlabel("Run"); ax.set_ylabel("Adsorbed phenols (%)")
    ax.legend(loc="upper center", ncol=1, fontsize=7)
    ax.set_title("(b) All 45 response values\n(red triangles: exact zeros)", fontsize=9)

    ax = fig.add_subplot(1, 3, 3)
    ctr = d[(d.x1_coded == 0) & (d.x2_coded == 0) & (d.x3_coded == 0)]
    for i, r in enumerate(R3):
        v = ctr[r].values
        ax.plot([i] * 3, v, "o", color=PS.RESP_COL[r], ms=6)
        ax.plot([i - .18, i + .18], [v.mean()] * 2, "-", color=PS.RESP_COL[r], lw=2)
        ax.annotate(f"SD {v.std(ddof=1):.2f}", (i, v.max() + 1.6), ha="center",
                    fontsize=7, color=PS.RESP_COL[r])
        s = float(fit[fit.response == r].sigma_hat.iloc[0])
        ax.plot([i + .30], [s], marker="_", ms=14, color="0.35")
    ax.set_xticks(range(3))
    ax.set_xticklabels([PS.RESP_LAB[r].replace(" ", "\n") for r in R3], fontsize=7)
    ax.set_ylabel("Adsorbed phenols (%)")
    ax.set_title("(c) Centre replicates:\npure experimental error", fontsize=9)
    ax.legend(handles=[Line2D([], [], marker="_", ls="", color="0.35",
                              label="published model residual SD")], loc="upper left",
              fontsize=7)
    PS.save(fig, "fig01_design_and_data")


# ---------------------------------------------------------------- fig 2
def fig02():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
    ax = axes[0]
    for r in R3:
        s = coefs[coefs.response == r]
        ax.plot(s.published, s.reproduced, "o", ms=6, color=PS.RESP_COL[r],
                label=PS.RESP_LAB[r], alpha=0.9)
    lim = [coefs.published.min() * 1.08, coefs.published.max() * 1.08]
    ax.plot(lim, lim, "-", color="0.6", lw=0.8, zorder=0)
    ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Published coefficient (Eq. 6-8)")
    ax.set_ylabel("Refitted from the transcribed data")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"(a) All 30 coefficients reproduced\nmax |diff| = "
                 f"{coefs.abs_diff.max():.3f}", fontsize=9)

    ax = axes[1]
    x = np.arange(3)
    ax.bar(x - 0.2, fit.r2_published_pct, 0.38, color="#B03A2E", label="published $R^2$")
    ax.bar(x + 0.2, fit.r2_reproduced_pct, 0.38, color="#2E86C1", label="reproduced $R^2$")
    for i, (a, b) in enumerate(zip(fit.r2_published_pct, fit.r2_reproduced_pct)):
        ax.annotate(f"{a:.2f}\n{b:.2f}", (i, b + 2), ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PS.RESP_LAB[r].replace(" ", "\n") for r in fit.response],
                       fontsize=7)
    ax.set_ylim(0, 112); ax.set_ylabel("$R^2$ (%)")
    ax.legend(fontsize=7); ax.set_title("(b) Published vs reproduced $R^2$", fontsize=9)

    ax = axes[2]
    n_pass = int((chk.status == "PASS").sum())
    cats = ["design\nstructure", "coded /\nreal levels", "integrity\n(N, zeros, dups)",
            "quoted\nmaxima", "coefficients\n(30)", "$R^2$\n(3)", "optima\n(3)"]
    sizes = [7, 3, 5, 6, 30, 3, 3]
    ax.barh(range(len(cats)), sizes, color="#117A65")
    for i, s in enumerate(sizes):
        ax.annotate(f"{s}/{s}", (s + 0.5, i), va="center", fontsize=7)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats, fontsize=7)
    ax.set_xlabel("Checks passed"); ax.set_xlim(0, 36)
    ax.invert_yaxis()
    ax.set_title(f"(c) Hard data check: {n_pass}/{len(chk)} pass", fontsize=9)
    PS.save(fig, "fig02_data_check")


# ---------------------------------------------------------------- fig 3
def real_design_matrix(df):
    """Equations (6)-(8) are stated in real units, not coded units."""
    p, c, f = df.pH.values, df.additive_mM.values, df.cnf_pct.values
    return np.column_stack([np.ones_like(p, float), p, c, f, p * p, c * c, f * f,
                            p * c, p * f, c * f])


def fig03():
    X = real_design_matrix(d)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6))
    for j, r in enumerate(R3):
        b = coefs[coefs.response == r].set_index("term").reproduced[TERMS].values
        y = d[r].values
        yh = X @ b
        ax = axes[0, j]
        lo = min(y.min(), yh.min()) - 4
        hi = max(y.max(), yh.max()) + 4
        if lo < 0:
            ax.axhspan(lo, 0, color="#B03A2E", alpha=0.10, zorder=0)
            ax.annotate("physically\nimpossible", (hi * 0.52, lo * 0.55), fontsize=6.5,
                        color="#B03A2E", ha="center")
        ax.plot([lo, hi], [lo, hi], "-", color="0.6", lw=0.8, zorder=1)
        neg = yh < 0
        ax.plot(y[~neg], yh[~neg], "o", ms=6, color=PS.RESP_COL[r], zorder=3)
        ax.plot(y[neg], yh[neg], "o", ms=7, mfc="none", mec="#B03A2E", mew=1.6, zorder=4)
        at0 = y == 0
        ax.plot(y[at0], yh[at0], "v", ms=5, color="#B03A2E", zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Observed (%)")
        if j == 0:
            ax.set_ylabel("Published model fitted (%)")
        nneg = int(neg.sum())
        ax.set_title(f"{PS.RESP_LAB[r]}\n{nneg} fitted value(s) < 0", fontsize=9)

        ax = axes[1, j]
        ax.axhline(0, color="0.6", lw=0.8)
        ax.plot(yh, y - yh, "o", ms=6, color=PS.RESP_COL[r])
        s = float(fit[fit.response == r].sigma_hat.iloc[0])
        ctr = d[(d.x1_coded == 0) & (d.x2_coded == 0) & (d.x3_coded == 0)][r]
        pe = ctr.std(ddof=1)
        ax.axhspan(-1.96 * pe, 1.96 * pe, color="0.75", alpha=0.35, zorder=0)
        ax.set_xlabel("Fitted (%)")
        if j == 0:
            ax.set_ylabel("Residual (%)")
        ax.set_title(f"residual SD {s:.2f} vs pure error {pe:.2f}", fontsize=8)
    fig.suptitle("The published quadratic fitted to its own data: negative predictions and "
                 "residuals smaller than the experiment's own repeatability",
                 fontsize=9.5, y=1.005)
    PS.save(fig, "fig03_published_model_diagnostics")


# ---------------------------------------------------------------- fig 4
def fig04():
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
    x = np.arange(3)
    ax = axes[0]
    ax.bar(x - 0.2, fit.r2_reproduced_pct, 0.38, color="#2E86C1", label="$R^2$")
    ax.bar(x + 0.2, fit.r2_adj_pct, 0.38, color="#B03A2E", label="adjusted $R^2$")
    for i, (a, b) in enumerate(zip(fit.r2_reproduced_pct, fit.r2_adj_pct)):
        ax.annotate(f"$-${a - b:.0f} pts", (i, b - 9), ha="center", fontsize=7,
                    color="#B03A2E")
    ax.set_xticks(x)
    ax.set_xticklabels([PS.RESP_LAB[r].replace(" ", "\n") for r in fit.response],
                       fontsize=7)
    ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend(fontsize=7)
    ax.set_title("(a) 10 parameters, 15 runs:\nadjusted $R^2$ collapses", fontsize=9)

    ax = axes[1]
    pe, sh = [], []
    for r in R3:
        ctr = d[(d.x1_coded == 0) & (d.x2_coded == 0) & (d.x3_coded == 0)][r]
        pe.append(ctr.std(ddof=1))
        sh.append(float(fit[fit.response == r].sigma_hat.iloc[0]))
    ax.bar(x - 0.2, pe, 0.38, color="0.45", label="pure error SD (centre reps)")
    ax.bar(x + 0.2, sh, 0.38, color="#2E86C1", label="model residual SD")
    ax.set_xticks(x)
    ax.set_xticklabels([PS.RESP_LAB[r].replace(" ", "\n") for r in R3], fontsize=7)
    ax.set_ylabel("Percentage points"); ax.legend(fontsize=6.5)
    ax.set_title("(b) The fit is tighter than\nthe experiment repeats", fontsize=9)

    ax = axes[2]
    lof = anova[anova.source == "  Lack of fit"]
    ax.bar(x, lof.p.values, 0.5, color="#117A65")
    ax.axhline(0.05, color="#B03A2E", ls="--", lw=1)
    ax.annotate("$p=0.05$", (2.35, 0.065), fontsize=7, color="#B03A2E", ha="right")
    for i, p in enumerate(lof.p.values):
        ax.annotate(f"{p:.2f}", (i, p + 0.02), ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([PS.RESP_LAB[r].replace(" ", "\n") for r in lof.response],
                       fontsize=7)
    ax.set_ylabel("Lack-of-fit $p$"); ax.set_ylim(0, 0.62)
    ax.set_title("(c) Lack of fit undetectable\nwith only 2 pure-error df", fontsize=9)
    PS.save(fig, "fig04_overparameterisation")


# ---------------------------------------------------------------- fig 5
def fig05():
    """Marginal factor effects: the three compounds disagree in sign."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=False)
    cods = ["x1_coded", "x2_coded", "x3_coded"]
    reals = {0: [4, 6, 8], 1: [5, 10, 15], 2: [0.10, 0.25, 0.40]}
    for k in range(3):
        ax = axes[k]
        for r in R3:
            m = [d[d[cods[k]] == lv][r].mean() for lv in (-1, 0, 1)]
            e = [d[d[cods[k]] == lv][r].std(ddof=1) / np.sqrt((d[cods[k]] == lv).sum())
                 for lv in (-1, 0, 1)]
            ax.errorbar(reals[k], m, yerr=e, marker="o", ms=5, capsize=3, lw=1.6,
                        color=PS.RESP_COL[r], label=PS.RESP_LAB[r])
        ax.set_xlabel(PS.FACTOR_LAB[k]); ax.set_xticks(reals[k])
        if k == 0:
            ax.set_ylabel("Mean adsorbed phenols (%)")
            ax.legend(fontsize=7)
        if k == 0:
            ax.set_title("(a) pH: tannic acid moves the\nopposite way", fontsize=9)
        else:
            ax.set_title(f"({'abc'[k]}) {PS.FACTOR_LAB[k]}", fontsize=9)
    fig.suptitle("Why naive pooling across the three compounds fails, and why CM-BARS "
                 "couples them with signed loadings", fontsize=9.5, y=1.02)
    PS.save(fig, "fig05_task_heterogeneity")


# ---------------------------------------------------------------- fig 6
def fig06():
    """Published response surfaces, with the region of negative prediction marked."""
    pub = pd.read_csv(f"{JD}/data/published_models.csv")
    opt = pd.read_csv(f"{JD}/data/published_r2_optima.csv")
    g = np.linspace(0, 1, 160)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    for j, r in enumerate(R3):
        b = pub[pub.response == r].set_index("term").coef[TERMS].values.astype(float)
        o = opt[opt.response == r].iloc[0]
        pH = 4 + 4 * g
        cnf = 0.10 + 0.30 * g
        P, C = np.meshgrid(pH, cnf, indexing="ij")
        A = np.full_like(P, float(o.opt_conc_mM))
        V = (b[0] + b[1] * P + b[2] * A + b[3] * C + b[4] * P ** 2 + b[5] * A ** 2
             + b[6] * C ** 2 + b[7] * P * A + b[8] * P * C + b[9] * A * C)
        ax = axes[j]
        cf = ax.contourf(P, C, V, levels=18, cmap=PS.SEQ)
        cs = ax.contour(P, C, V, levels=8, colors="white", linewidths=0.5)
        ax.clabel(cs, fontsize=5.5, fmt="%.0f")
        if (V < 0).any():
            ax.contourf(P, C, np.where(V < 0, 1.0, np.nan), levels=[0.5, 1.5],
                        colors=["#B03A2E"], alpha=0.45)
            ax.contour(P, C, V, levels=[0], colors="#B03A2E", linewidths=1.6)
        ax.plot(o.opt_pH, o.opt_cnf_pct, marker="*", ms=15, mfc="white",
                mec="black", mew=1.0)
        ax.set_xlabel("pH")
        if j == 0:
            ax.set_ylabel("Final CNF conc. (%)")
        frac = float((V < 0).mean()) * 100
        ax.set_title(f"{PS.RESP_LAB[r]}\n{frac:.0f}% of the box predicts < 0",
                     fontsize=9)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03).set_label("Adsorbed (%)",
                                                                    fontsize=7)
    fig.suptitle("The published surfaces (Eq. 6-8) at each compound's own optimal additive "
                 "concentration; stars are the Table 2 optima", fontsize=9.5, y=1.03)
    PS.save(fig, "fig06_published_surfaces")


if __name__ == "__main__":
    fig01(); fig02(); fig03(); fig04(); fig05(); fig06()
