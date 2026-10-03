"""Condition-grouped experiments for the CNF--phenolic dataset.

LOCO holds out all responses and replicates at one of 13 conditions.
LOO holds out one run; other center-point replicates can remain in training.
RKF repeats grouped five-fold splits. LOCO/LOO metrics use the complete
out-of-fold vector; RKF metrics are calculated per fold.
Use run.py cv for configuration- and data-specific cache namespaces.
"""
import argparse
import hashlib
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)
CACHE = f"{JD}/results/cv_cache"
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

import methods as M                                            # noqa: E402
from metrics import HIGHER_BETTER, all_metrics                  # noqa: E402

RESPONSES = M.RESPONSES


# ---------------------------------------------------------------- method registry
def build_registry(fast=False):
    """name -> zero-argument constructor.  `fast` shortens the MCMC for smoke runs."""
    from importlib import import_module
    def CMBARS1(**kw): return import_module("cmbars").CMBARS(**kw)
    def CMBARS2(**kw): return import_module("cmbars2").CMBARS2(**kw)
    w, s = (300, 300) if fast else (1000, 1000)
    wg, sg = (400, 400) if fast else (2000, 2000)

    def cb(**kw):
        return lambda: CMBARS2(warmup=w, samples=s, chains=2, **kw)

    reg = {
        # ---- the published method and conventional alternatives -------------
        "OLS-Quad (published)": M.OLSQuad,
        "OLS-Quad + clip": M.OLSQuadClip,
        "Ridge-Quad": M.RidgeQuad,
        "Lasso-Quad": M.LassoQuad,
        "ElasticNet-Quad": M.ENetQuad,
        "PLS-Quad": M.PLSQuad,
        # ---- machine-learning baselines -------------------------------------
        "RandomForest": M.RandomForest,
        "ExtraTrees": M.ExtraTrees,
        "LightGBM": M.LightGBM,
        "XGBoost": M.XGBoost,
        "GP-Matern": M.GPMatern,
        "Tobit-Quad": M.TobitQuad,
        # ---- the proposed method --------------------------------------------
        "CM-BARS": cb(use_gp=False),
        # ---- ablations: one component removed at a time ---------------------
        "CM-BARS -zero-part": cb(use_gp=False, zero_part=False,
                                 name="CM-BARS -zero-part"),
        "CM-BARS -bounded link": cb(use_gp=False, bounded=False,
                                    name="CM-BARS -bounded link"),
        "CM-BARS -shrinkage": cb(use_gp=False, shrink=False,
                                 name="CM-BARS -shrinkage"),
        "CM-BARS -coupling": cb(use_gp=False, coupling="none",
                                name="CM-BARS -coupling"),
        "CM-BARS mean-pooled": cb(use_gp=False, coupling="mean",
                                  name="CM-BARS mean-pooled"),
        "CM-BARS +GP": lambda: CMBARS2(use_gp=True, warmup=w, samples=s, chains=2,
                                       target_accept=0.92, max_tree_depth=10,
                                       name="CM-BARS +GP"),
        "CM-BARS -all": cb(use_gp=False, zero_part=False, bounded=False, shrink=False,
                           coupling="none", name="CM-BARS -all"),
        # ---- the Tobit-on-the-raw-scale predecessor (first design iteration) -
        "Tobit-MT-Bayes": lambda: CMBARS1(warmup=w, samples=s, chains=2,
                                          use_gp=False, name="Tobit-MT-Bayes"),
    }
    if any(os.path.exists(q) for q in [os.environ.get("PFN_CHECKPOINT", ""), f"{JD}/results/pfn_rsm.pt", f"{JD}/checkpoints/pfn_rsm.pt"]):
        reg["PFN-RSM (prior-fitted)"] = lambda: import_module("pfn").PFNRSM()
    # CM-BARS+ is assembled by stack_oof.py from the members' cached out-of-fold
    # predictions and is served straight from the cache; the constructor below is only a
    # fallback and is never invoked when stack_oof.py has already run.
    from stack import CMBARSPlus
    reg["CM-BARS+ (stacked)"] = CMBARSPlus
    return reg


# ---------------------------------------------------------------- folds
def condition_groups(d):
    """Group index per run: the three centre runs share a group."""
    key = list(zip(d.x1_coded, d.x2_coded, d.x3_coded))
    uniq = {}
    g = []
    for k in key:
        uniq.setdefault(k, len(uniq))
        g.append(uniq[k])
    return np.asarray(g)


def make_folds(d, n_rep=20, k=5, seed=0):
    g = condition_groups(d)
    n_cond = g.max() + 1
    folds = []
    for c in range(n_cond):                                     # LOCO
        te = np.where(g == c)[0]
        folds.append(("LOCO", 0, c, np.setdiff1d(np.arange(len(g)), te), te))
    for i in range(len(g)):                                     # LOO
        folds.append(("LOO", 0, i, np.setdiff1d(np.arange(len(g)), [i]), np.array([i])))
    rng = np.random.default_rng(seed)                           # repeated grouped 5-fold
    for rep in range(n_rep):
        order = rng.permutation(n_cond)
        chunks = np.array_split(order, k)
        for f, ch in enumerate(chunks):
            te = np.where(np.isin(g, ch))[0]
            folds.append(("RKF", rep, f, np.setdiff1d(np.arange(len(g)), te), te))
    return folds


# ---------------------------------------------------------------- worker
def _key(protocol, rep, fold, name):
    h = hashlib.md5(f"{protocol}|{rep}|{fold}|{name}".encode()).hexdigest()[:16]
    namespace = os.environ.get("CMBARS_CACHE_NAMESPACE", "legacy")
    folder = os.path.join(CACHE, namespace)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{h}.pkl")


def run_one(args):
    protocol, rep, fold, tr, te, name, Z, Y, fast = args
    path = _key(protocol, rep, fold, name)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                cached = pickle.load(fh)
                if not cached.get("error"):
                    return cached
        except Exception:
            os.remove(path)
    t0 = time.time()
    try:
        mdl = build_registry(fast=fast)[name]()
        mdl.fit(Z[tr], Y[tr])
        pred = mdl.predict(Z[te])
        rec = dict(protocol=protocol, rep=rep, fold=fold, method=name,
                   test=te.tolist(), n_train=len(tr), n_test=len(te),
                   seconds=time.time() - t0, error=None,
                   pred={r: dict(mean=np.asarray(pred[r]["mean"], np.float32),
                                 sd=np.asarray(pred[r].get("sd", np.full(len(te), np.nan)),
                                               np.float32),
                                 samples=np.asarray(pred[r]["samples"], np.float32)
                                 if "samples" in pred[r] else None)
                         for r in RESPONSES})
    except Exception:
        rec = dict(protocol=protocol, rep=rep, fold=fold, method=name, test=te.tolist(),
                   n_train=len(tr), n_test=len(te), seconds=time.time() - t0,
                   error=traceback.format_exc()[-1500:], pred=None)
    with open(path, "wb") as fh:
        pickle.dump(rec, fh)
    return rec


# ---------------------------------------------------------------- scoring
def pooled_scores(records, Y, protocol):
    """Metrics on the pooled out-of-fold predictions (used for LOCO and LOO)."""
    rows = []
    bym = {}
    for rec in records:
        if rec["protocol"] != protocol or rec["pred"] is None:
            continue
        bym.setdefault(rec["method"], []).append(rec)
    n = len(Y)
    for name, recs in bym.items():
        for j, r in enumerate(RESPONSES):
            mean = np.full(n, np.nan)
            sd = np.full(n, np.nan)
            smp = None
            for rec in recs:
                te = np.asarray(rec["test"])
                mean[te] = rec["pred"][r]["mean"]
                sd[te] = rec["pred"][r]["sd"]
                s = rec["pred"][r]["samples"]
                if s is not None:
                    if smp is None:
                        smp = np.full((s.shape[0], n), np.nan, np.float32)
                    smp[:, te] = s
            ok = ~np.isnan(mean)
            if ok.sum() < n:
                continue
            m = all_metrics(Y[:, j], mean,
                            samples=smp if smp is not None and not np.isnan(smp).any() else None,
                            sd=sd if not np.isnan(sd).any() else None)
            rows.append(dict(protocol=protocol, method=name, response=r,
                             n=int(ok.sum()), **m))
    return pd.DataFrame(rows)


def perfold_scores(records, Y, protocol="RKF"):
    rows = []
    for rec in records:
        if rec["protocol"] != protocol or rec["pred"] is None:
            continue
        te = np.asarray(rec["test"])
        for j, r in enumerate(RESPONSES):
            p = rec["pred"][r]
            m = all_metrics(Y[te, j], p["mean"],
                            samples=p["samples"], sd=p["sd"])
            rows.append(dict(protocol=protocol, method=rec["method"], response=r,
                             rep=rec["rep"], fold=rec["fold"],
                             n_test=rec["n_test"], n_train=rec["n_train"],
                             seconds=rec["seconds"], **m))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated subset of method names")
    ap.add_argument("--protocols", default="LOCO,LOO,RKF")
    a = ap.parse_args()

    d = pd.read_csv(f"{JD}/data/bbd_design_responses.csv")
    Z = M.to_coded(d.pH, d.additive_mM, d.cnf_pct)
    Y = d[RESPONSES].values.astype(float)

    reg = build_registry(fast=a.fast)
    names = [n.strip() for n in a.only.split(",")] if a.only else list(reg)
    missing = [n for n in names if n not in reg]
    if missing:
        raise SystemExit(f"unknown methods: {missing}")
    want = set(p.strip() for p in a.protocols.split(","))
    folds = [f for f in make_folds(d, n_rep=a.n_rep) if f[0] in want]

    tasks = [(p, rep, f, tr, te, nm, Z, Y, a.fast)
             for (p, rep, f, tr, te) in folds for nm in names]
    print(f"{len(tasks)} fits ({len(names)} methods x {len(folds)} folds), "
          f"{a.workers} workers", flush=True)

    records, done, t0 = [], 0, time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for fu in as_completed(futs):
            rec = fu.result()
            records.append(rec)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  {el:.0f}s elapsed, "
                      f"{el / done * (len(tasks) - done):.0f}s left", flush=True)

    errs = [r for r in records if r["error"]]
    if errs:
        print(f"\n{len(errs)} failed fits, e.g.:\n{errs[0]['method']}\n{errs[0]['error']}")
        pd.DataFrame([{k: r[k] for k in ("protocol", "rep", "fold", "method", "error")}
                      for r in errs]).to_csv(f"{JD}/results/cv_errors.csv", index=False)

    out = []
    for p in ["LOCO", "LOO"]:
        if p in want:
            out.append(pooled_scores(records, Y, p))
    pooled = pd.concat([o for o in out if len(o)], ignore_index=True) if any(len(o) for o in out) else pd.DataFrame()
    if len(pooled):
        pooled.to_csv(f"{JD}/results/cv_pooled.csv", index=False)
    if "RKF" in want:
        pf = perfold_scores(records, Y)
        pf.to_csv(f"{JD}/results/cv_perfold.csv", index=False)
        print(f"\nRKF per-fold rows: {len(pf)}")

    if len(pooled):
        piv = (pooled[pooled.protocol == "LOCO"]
               .pivot_table(index="method", columns="response", values="RMSE"))
        piv["mean_RMSE"] = piv.mean(axis=1)
        print("\n=== LOCO pooled RMSE (lower is better) ===")
        print(piv.sort_values("mean_RMSE").round(3).to_string())

    if errs:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
