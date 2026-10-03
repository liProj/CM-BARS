"""Simulation study: does the new method recover the truth better than the published one?

Fifteen runs cannot settle a methodological comparison on their own, so the real-data
cross-validation is backed by a simulation in which the truth is known.  Surfaces are drawn
from `prior.py` under four regimes and every method is handed exactly the design the paper
used (the 15-run Box-Behnken), so the comparison is at the paper's own sample size.

  cmbars  hierarchical multi-task quadratic + Matern deviation, censored at zero
          (the regime CM-BARS assumes)
  ols     independent plain quadratics, homoscedastic Gaussian noise, NOT censored
          (the regime the *published* method assumes -- the honest adversarial case)
  gp      GP-dominated surfaces with only a weak quadratic trend (neither model is right)
  sparse  only linear terms and interactions active (a strongly shrinkable truth)

Six quantities are scored per simulated dataset and response:

  TestRMSE / TestCRPS / TestNLL / TestCover90   on a fresh independent replicate measured
                                                at the same 15 design points
  SurfaceRMSE                                   against the true mean response E[y|z] on a
                                                dense grid over the design box
  OptDistance                                   coded-unit distance from the recommended
                                                optimum to the true optimum
  OptRegret                                     true mean response lost by acting on the
                                                recommended optimum instead of the true one
  FracNegative                                  share of grid predictions below 0%

`OptRegret` is the quantity a formulation scientist actually pays: how much adsorption is
given up by trusting the model's recommended pH / concentration / nanocellulose loading.
"""
import argparse
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)
CACHE = f"{JD}/results/sim_cache"
os.makedirs(CACHE, exist_ok=True)

# Each worker runs its own JAX process, so every library must be pinned to a single
# thread; otherwise N workers x N threads oversubscribes the machine and XLA compilation
# slows down by orders of magnitude.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 "
    "--xla_force_host_platform_device_count=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import methods as M                                               # noqa: E402
from metrics import crps_censored_samples, nll_from_samples        # noqa: E402

RESPONSES = M.RESPONSES
REGIMES = ["cmbars", "ols", "gp", "sparse"]
GRID_M = 9


def grid_points(m=GRID_M):
    g = np.linspace(-1, 1, m)
    A, B, C = np.meshgrid(g, g, g, indexing="ij")
    return np.column_stack([A.ravel(), B.ravel(), C.ravel()])


def censored_mean(eta, sigma, lower=0.0, censor=True):
    """E[max(lower, eta + N(0, sigma))] -- the true mean response of the DGP."""
    if not censor:
        return eta
    s = np.maximum(sigma, 1e-9)
    z = (eta - lower) / s
    return lower + (eta - lower) * norm.cdf(z) + s * norm.pdf(z)


def sim_methods(fast=False):
    from cmbars2 import CMBARS2
    w, s = (300, 300) if fast else (800, 800)
    reg = {
        "OLS-Quad (published)": M.OLSQuad,
        "OLS-Quad + clip": M.OLSQuadClip,
        "Ridge-Quad": M.RidgeQuad,
        "ExtraTrees": M.ExtraTrees,
        "LightGBM": M.LightGBM,
        "GP-Matern": M.GPMatern,
        "Tobit-Quad": M.TobitQuad,
        "CM-BARS": lambda: CMBARS2(use_gp=False, warmup=w, samples=s, chains=2),
        "CM-BARS -coupling": lambda: CMBARS2(use_gp=False, coupling="none", warmup=w,
                                             samples=s, chains=2,
                                             name="CM-BARS -coupling"),
        "CM-BARS -zero-part": lambda: CMBARS2(use_gp=False, zero_part=False, warmup=w,
                                              samples=s, chains=2,
                                              name="CM-BARS -zero-part"),
    }
    if os.path.exists(f"{JD}/results/pfn_rsm.pt"):
        from pfn import PFNRSM
        reg["PFN-RSM (prior-fitted)"] = PFNRSM
    return reg


def make_dataset(regime, idx, device="cpu"):
    """One simulated experiment: training data, a fresh replicate, and the true surface."""
    import torch

    from prior import BBD, basis, sample_surfaces
    gen = torch.Generator(device=device).manual_seed(100000 + idx)
    Zg = grid_points()
    Zall = np.vstack([BBD, Zg])
    s = sample_surfaces(1, Zall, gen, device, n_task=3, regime=regime, censor=True)
    nb = BBD.shape[0]
    eta = s["eta"][0].cpu().numpy()                               # (n_all, R)
    sigma = s["sigma"][0].cpu().numpy()                           # (R,)
    censor = regime != "ols"
    rng = np.random.default_rng(900000 + idx)
    noise = rng.standard_normal((Zall.shape[0], 3)) * sigma[None, :]
    y_all = eta + noise
    if censor:
        y_all = np.clip(y_all, 0.0, 100.0)
    y_tr = y_all[:nb]
    noise2 = rng.standard_normal((nb, 3)) * sigma[None, :]
    y_te = eta[:nb] + noise2
    if censor:
        y_te = np.clip(y_te, 0.0, 100.0)
    true_mean_grid = censored_mean(eta[nb:], sigma[None, :], censor=censor)
    return dict(Ztr=BBD, y_tr=y_tr, y_te=y_te, Zg=Zg,
                true_mean_grid=true_mean_grid, sigma=sigma, censor=censor)


def run_one(args):
    regime, idx, name, fast = args
    path = f"{CACHE}/{regime}_{idx}_{abs(hash(name)) % 10**10}.pkl"
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            os.remove(path)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.time()
    rows = []
    try:
        D = make_dataset(regime, idx)
        mdl = sim_methods(fast=fast)[name]()
        mdl.fit(D["Ztr"], D["y_tr"])
        pte = mdl.predict(D["Ztr"])
        pg = mdl.predict(D["Zg"], n_draw=400) if hasattr(mdl, "predict") else None
        for j, r in enumerate(RESPONSES):
            yt = D["y_te"][:, j]
            pr = pte[r]
            sm = pr.get("samples")
            lo, hi = (np.quantile(sm, 0.05, axis=0), np.quantile(sm, 0.95, axis=0)) \
                if sm is not None else (None, None)
            tm = D["true_mean_grid"][:, j]
            gm = pg[r]["mean"]
            k_hat = int(np.argmax(gm))
            k_true = int(np.argmax(tm))
            rows.append(dict(
                regime=regime, dataset=idx, method=name, response=r,
                TestRMSE=float(np.sqrt(np.mean((pr["mean"] - yt) ** 2))),
                TestCRPS=float(crps_censored_samples(yt, sm)) if sm is not None else np.nan,
                TestNLL=float(nll_from_samples(yt, sm)) if sm is not None else np.nan,
                TestCover90=float(np.mean((yt >= lo) & (yt <= hi))) if lo is not None else np.nan,
                SurfaceRMSE=float(np.sqrt(np.mean((gm - tm) ** 2))),
                OptDistance=float(np.linalg.norm(D["Zg"][k_hat] - D["Zg"][k_true])),
                OptRegret=float(tm[k_true] - tm[k_hat]),
                FracNegative=float(np.mean(gm < -1e-9)),
                seconds=time.time() - t0, error=None))
    except Exception:
        rows = [dict(regime=regime, dataset=idx, method=name, response=r,
                     seconds=time.time() - t0, error=traceback.format_exc()[-800:])
                for r in RESPONSES]
    with open(path, "wb") as fh:
        pickle.dump(rows, fh)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="datasets per regime")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--regimes", default=",".join(REGIMES))
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    reg = sim_methods(fast=a.fast)
    names = [n.strip() for n in a.only.split(",")] if a.only else list(reg)
    regimes = [r.strip() for r in a.regimes.split(",")]
    tasks = [(rg, i, nm, a.fast) for rg in regimes for i in range(a.n) for nm in names]
    print(f"{len(tasks)} fits ({len(names)} methods x {len(regimes)} regimes x {a.n})",
          flush=True)

    allrows, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fu in as_completed(futs):
            allrows += fu.result()
            done += 1
            if done % 200 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  {el:.0f}s, {el/done*(len(tasks)-done):.0f}s left",
                      flush=True)

    df = pd.DataFrame(allrows)
    bad = df[df.error.notna()] if "error" in df else pd.DataFrame()
    if len(bad):
        print(f"{len(bad)} failed rows; first error:\n{bad.error.iloc[0]}")
        bad.to_csv(f"{JD}/results/sim_errors.csv", index=False)
    df = df[df.error.isna()] if "error" in df else df
    df.to_csv(f"{JD}/results/simulation.csv", index=False)
    print(f"\n{len(df)} rows -> results/simulation.csv")
    for rg in regimes:
        sub = df[df.regime == rg]
        if not len(sub):
            continue
        piv = sub.groupby("method")[["TestRMSE", "SurfaceRMSE", "OptRegret",
                                     "TestCover90", "FracNegative"]].mean()
        print(f"\n=== regime: {rg} ===")
        print(piv.sort_values("SurfaceRMSE").round(3).to_string())


if __name__ == "__main__":
    main()
