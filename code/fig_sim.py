"""Figures 15-20: simulation study, the prior-fitted network, stacking and distillation."""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)
RES = f"{JD}/results"

import methods as M                                               # noqa: E402
import plotstyle as PS                                            # noqa: E402

PS.setup()
R3 = M.RESPONSES
PUB = "OLS-Quad (published)"
REG_LAB = {"cmbars": "censored multi-task\n(CM-BARS' own prior)",
           "ols": "plain quadratic, uncensored\n(the PUBLISHED model's prior)",
           "gp": "GP-dominated\n(neither model is right)",
           "sparse": "sparse truth\n(strongly shrinkable)"}


def _proposed(df):
    for c in ["CM-BARS+ (stacked)", "CM-BARS"]:
        if c in set(df.method):
            return c
    return sorted(set(df.method))[0]


# ---------------------------------------------------------------- fig 15
def fig15():
    sim = pd.read_csv(f"{RES}/simulation.csv")
    regs = [r for r in ["cmbars", "ols", "gp", "sparse"] if r in set(sim.regime)]
    order = sim.groupby("method").SurfaceRMSE.mean().sort_values().index.tolist()[::-1]
    fig, axes = plt.subplots(1, len(regs), figsize=(3.1 * len(regs), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for k, rg in enumerate(regs):
        ax = axes[k]
        s = sim[sim.regime == rg]
        v = [s[s.method == m].SurfaceRMSE.mean() for m in order]
        e = [s[s.method == m].SurfaceRMSE.std(ddof=1)
             / np.sqrt(max(1, s[s.method == m].SurfaceRMSE.count())) for m in order]
        cols = ["#B03A2E" if m == PUB else "#0B6E4F" if m.startswith("CM-BARS+")
                else "#117A65" if m.startswith("CM-BARS") else "#7F8C8D" for m in order]
        ax.barh(np.arange(len(order)), v, xerr=e, color=cols, error_kw=dict(lw=0.8))
        ax.set_yticks(np.arange(len(order)))
        if k == 0:
            ax.set_yticklabels(order, fontsize=7)
        ax.set_xlabel("Surface RMSE (% points)")
        ax.set_title(REG_LAB.get(rg, rg), fontsize=8.5)
    fig.suptitle("Recovery of the true mean response surface, 15-run design, "
                 f"{int(sim.groupby(['regime','method']).dataset.nunique().max())} "
                 "simulated experiments per regime", fontsize=9.5, y=1.02)
    PS.save(fig, "fig15_simulation_surface")


# ---------------------------------------------------------------- fig 16
def fig16():
    sim = pd.read_csv(f"{RES}/simulation.csv")
    prop = _proposed(sim)
    regs = [r for r in ["cmbars", "ols", "gp", "sparse"] if r in set(sim.regime)]
    show = [m for m in [PUB, "OLS-Quad + clip", "Ridge-Quad", "ExtraTrees", "GP-Matern",
                        "Tobit-Quad", "CM-BARS", prop] if m in set(sim.method)]
    show = list(dict.fromkeys(show))
    fig, axes = plt.subplots(1, len(regs), figsize=(3.1 * len(regs), 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for k, rg in enumerate(regs):
        ax = axes[k]
        s = sim[sim.regime == rg]
        data = [s[s.method == m].OptRegret.dropna().values for m in show]
        bp = ax.boxplot(data, vert=True, widths=0.6, showfliers=False,
                        patch_artist=True, medianprops=dict(color="black", lw=1.2))
        for patch, m in zip(bp["boxes"], show):
            patch.set_facecolor(PS.col(m))
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(show) + 1))
        ax.set_xticklabels(show, rotation=55, ha="right", fontsize=6.5)
        if k == 0:
            ax.set_ylabel("Optimum regret (% points of adsorption lost)")
        ax.set_title(REG_LAB.get(rg, rg), fontsize=8.5)
    fig.suptitle("The decision-relevant loss: adsorption given up by acting on each "
                 "method's recommended factor settings", fontsize=9.5, y=1.04)
    PS.save(fig, "fig16_simulation_regret")


# ---------------------------------------------------------------- fig 17
def fig17():
    sim = pd.read_csv(f"{RES}/simulation.csv")
    regs = [r for r in ["cmbars", "ols", "gp", "sparse"] if r in set(sim.regime)]
    order = sim.groupby("method").SurfaceRMSE.mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    ax = axes[0]
    for rg in regs:
        s = sim[sim.regime == rg]
        v = [s[s.method == m].TestCover90.mean() for m in order]
        ax.plot(range(len(order)), v, "o-", ms=4, lw=1.2, label=rg)
    ax.axhline(0.90, color="#B03A2E", ls="--", lw=1.1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=65, ha="right", fontsize=6.2)
    ax.set_ylabel("coverage of the 90% interval")
    ax.legend(fontsize=6.5, title="regime", title_fontsize=6.5)
    ax.set_title("(a) Interval calibration", fontsize=9)

    ax = axes[1]
    for rg in regs:
        s = sim[sim.regime == rg]
        v = [s[s.method == m].FracNegative.mean() * 100 for m in order]
        ax.plot(range(len(order)), v, "o-", ms=4, lw=1.2, label=rg)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=65, ha="right", fontsize=6.2)
    ax.set_ylabel("grid predictions below 0% (%)")
    ax.set_title("(b) Physically impossible predictions", fontsize=9)

    ax = axes[2]
    for rg in regs:
        s = sim[sim.regime == rg]
        v = [s[s.method == m].OptDistance.mean() for m in order]
        ax.plot(range(len(order)), v, "o-", ms=4, lw=1.2, label=rg)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=65, ha="right", fontsize=6.2)
    ax.set_ylabel("distance to the true optimum (coded units)")
    ax.set_title("(c) Optimum localisation", fontsize=9)
    fig.suptitle("Simulation study: calibration, physical validity and optimum recovery",
                 fontsize=9.5, y=1.04)
    PS.save(fig, "fig17_simulation_calibration")


# ---------------------------------------------------------------- fig 18
def fig18():
    h = f"{RES}/pfn_train_history.csv"
    if not os.path.exists(h):
        print("  (skipping fig18)")
        return
    hist = pd.read_csv(h)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    ax = axes[0]
    ax.plot(hist.step, hist.nll, "-", color="#0B6E4F", lw=1.6)
    ax.set_xlabel("training step (each step = 256 synthetic experiments)")
    ax.set_ylabel("negative log predictive score")
    ax.set_title(f"(a) Prior-fitted network: {int(hist.step.max()):,} steps,\n"
                 f"{hist.step.max()*256/1e6:.1f}M synthetic designs, "
                 f"{hist.seconds.max()/3600:.1f} h on one GB10 GPU", fontsize=8.5)

    ax = axes[1]
    pooled_f = f"{RES}/cv_pooled.csv"
    if os.path.exists(pooled_f):
        p = pd.read_csv(pooled_f)
        p = p[p.protocol == "LOCO"]
        show = [m for m in ["PFN-RSM (prior-fitted)", "CM-BARS", PUB] if m in set(p.method)]
        x = np.arange(3)
        for i, m in enumerate(show):
            v = [float(p[(p.method == m) & (p.response == r)].CRPS.iloc[0])
                 if len(p[(p.method == m) & (p.response == r)]) else np.nan for r in R3]
            ax.bar(x + (i - 1) * 0.26, v, 0.25, color=PS.col(m), label=m)
        ax.set_xticks(x)
        ax.set_xticklabels([PS.RESP_LAB[r].replace(" ", "\n") for r in R3], fontsize=7)
        ax.set_ylabel("CRPS (% points)")
        ax.legend(fontsize=6.5)
    ax.set_title("(b) One forward pass vs full MCMC\n(leave-one-condition-out)",
                 fontsize=8.5)
    PS.save(fig, "fig18_pfn_training")


# ---------------------------------------------------------------- fig 19
def fig19():
    sw = f"{RES}/final/stack_weights.csv"
    if not os.path.exists(sw):
        print("  (skipping fig19)")
        return
    w = pd.read_csv(sw)
    p = w.pivot_table(index="member", columns="response", values="weight").fillna(0)
    p = p.reindex(sorted(p.index, key=lambda m: -p.loc[m].mean()))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    ax = axes[0]
    bottom = np.zeros(3)
    xs = np.arange(3)
    for m in p.index:
        v = np.array([p.loc[m, r] if r in p.columns else 0 for r in R3])
        ax.bar(xs, v, 0.6, bottom=bottom, label=m, color=PS.col(m))
        bottom += v
    ax.set_xticks(xs)
    ax.set_xticklabels([PS.RESP_LAB[r].replace(" ", "\n") for r in R3], fontsize=7)
    ax.set_ylabel("pooling weight")
    ax.legend(fontsize=6.3, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title("(a) CM-BARS+ pooling weights", fontsize=9)

    ax = axes[1]
    dj = f"{RES}/distilled_coefficients.json"
    if os.path.exists(dj):
        dd = json.load(open(dj))
        v = [dd[r]["distill_rmse"] for r in R3]
        ax.bar(xs, v, 0.55, color=[PS.RESP_COL[r] for r in R3])
        for i, x in enumerate(v):
            ax.annotate(f"{x:.2f}", (i, x), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([PS.RESP_LAB[r].replace(" ", "\n") for r in R3], fontsize=7)
        ax.set_ylabel("distillation RMSE (% points)")
    ax.set_title("(b) Cost of compressing the stack\ninto one printable equation",
                 fontsize=9)
    PS.save(fig, "fig19_stack_and_distillation")


# ---------------------------------------------------------------- fig 20
def fig20():
    f = f"{RES}/final/grid_predictions.npz"
    if not os.path.exists(f):
        print("  (skipping fig20)")
        return
    z = np.load(f)
    axis = z["axis"]
    m = len(axis)
    d = pd.read_csv(f"{JD}/data/bbd_design_responses.csv")
    opt = pd.read_csv(f"{JD}/data/published_r2_optima.csv")
    post = pd.read_csv(f"{RES}/final/optima_posterior.csv") \
        if os.path.exists(f"{RES}/final/optima_posterior.csv") else None
    pH = 4 + 4 * (axis + 1) / 2
    cnf = 0.10 + 0.30 * (axis + 1) / 2
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4))
    for j, r in enumerate(R3):
        S = z[f"stack_mean_{r}"].reshape(m, m, m)
        SD = z[f"stack_sd_{r}"].reshape(m, m, m)
        # slice at the posterior-mean additive concentration
        kc = int(np.argmin(np.abs(axis - (-1.0))))     # 5 mM = coded -1 for all three
        V = S[:, kc, :]
        W = SD[:, kc, :]
        P, C = np.meshgrid(pH, cnf, indexing="ij")
        ax = axes[0, j]
        cf = ax.contourf(P, C, V, levels=18, cmap=PS.SEQ)
        cs = ax.contour(P, C, V, levels=8, colors="white", linewidths=0.5)
        ax.clabel(cs, fontsize=5.5, fmt="%.0f")
        o = opt[opt.response == r].iloc[0]
        ax.plot(o.opt_pH, o.opt_cnf_pct, marker="*", ms=14, mfc="#B03A2E", mec="white")
        if post is not None:
            q = post[post.response == r].iloc[0]
            ax.plot(q.post_pH_mean, q.post_cnf_mean, marker="D", ms=7, mfc="#0B6E4F",
                    mec="white")
        ax.set_xlabel("pH")
        if j == 0:
            ax.set_ylabel("Final CNF conc. (%)")
        ax.set_title(f"{PS.RESP_LAB[r]}: posterior mean", fontsize=9)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03).set_label("%", fontsize=7)

        ax = axes[1, j]
        cf = ax.contourf(P, C, W, levels=18, cmap="magma")
        ax.set_xlabel("pH")
        if j == 0:
            ax.set_ylabel("Final CNF conc. (%)")
        ax.set_title("predictive SD", fontsize=9)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03).set_label("% points", fontsize=7)
    fig.suptitle("Proposed surfaces at 5 mM additive, with the uncertainty the published "
                 "point estimates do not carry\n(stars: published Table 2 optima; "
                 "diamonds: posterior-mean optima)", fontsize=9.5, y=1.02)
    PS.save(fig, "fig20_proposed_surfaces")


if __name__ == "__main__":
    for f in [fig15, fig16, fig17, fig18, fig19, fig20]:
        try:
            f()
        except Exception as e:
            print(f"  {f.__name__} failed: {type(e).__name__}: {e}")
