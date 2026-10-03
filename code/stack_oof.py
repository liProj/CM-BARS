"""Historical CM-BARS+ pooling from cached out-of-fold predictions.

Weights for an outer test fold use predictions from other folds whose member
training sets may include that test condition. This introduces indirect
information leakage. These records are not unbiased outer-fold generalization
estimates and are separate from the main CM-BARS evaluation. Strictly nested
pooling requires inner predictions generated entirely inside each outer
training set. This script preserves the historical supplementary workflow.
"""
import argparse
import hashlib
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)
CACHE = f"{JD}/results/cv_cache"

import methods as M                                               # noqa: E402
from experiments import _key, make_folds                          # noqa: E402
from stack import _simplex_weights                                # noqa: E402

RESPONSES = M.RESPONSES
# The member set is fixed a priori by *model family*, not by measured performance:
# one or two representatives of each of the families a practitioner might reach for --
# the published quadratic, penalised/latent linear models, tree ensembles, a kernel
# method, the proposed Bayesian model, and the amortised network.
MEMBERS = [
    "OLS-Quad + clip",          # the published model, physically clipped
    "PLS-Quad", "ElasticNet-Quad",   # latent / penalised linear
    "RandomForest", "ExtraTrees", "LightGBM", "XGBoost",   # tree ensembles
    "GP-Matern",                # kernel
    "CM-BARS",                  # the proposed Bayesian model
    "PFN-RSM (prior-fitted)",   # the amortised network
]
NAME = "CM-BARS+ (stacked)"


def load(protocol, rep, fold, name):
    p = _key(protocol, rep, fold, name)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def build(protocol, n_rep=20, members=None, verbose=True):
    members = members or MEMBERS
    d = pd.read_csv(f"{JD}/data/bbd_design_responses.csv")
    Y = d[RESPONSES].values.astype(float)
    folds = [f for f in make_folds(d, n_rep=n_rep) if f[0] == protocol]

    # group folds into "repeats" so weights are fitted within a repeat for RKF
    by_rep = {}
    for (p, rep, f, tr, te) in folds:
        by_rep.setdefault(rep, []).append((f, te))

    have = []
    for nm in members:
        ok = all(load(protocol, rep, f, nm) is not None
                 for rep, lst in by_rep.items() for f, _ in lst)
        if ok:
            have.append(nm)
        elif verbose:
            print(f"  skipping member (incomplete cache): {nm}")
    if len(have) < 2:
        print(f"  {protocol}: fewer than two complete members, nothing to stack")
        return 0, {}

    made, wlog = 0, {}
    for rep, lst in by_rep.items():
        # member samples per fold
        S = {nm: {f: load(protocol, rep, f, nm) for f, _ in lst} for nm in have}
        for f, te in lst:
            # --- weights from every OTHER fold of this repeat ---
            w_r = {}
            for j, r in enumerate(RESPONSES):
                sets, ys = [], None
                cols = []
                for nm in have:
                    parts, yy = [], []
                    for f2, te2 in lst:
                        if f2 == f:
                            continue
                        rec = S[nm][f2]
                        if rec is None or rec["pred"] is None:
                            continue
                        parts.append(np.asarray(rec["pred"][r]["samples"], float))
                        yy.append(Y[np.asarray(rec["test"]), j])
                    if not parts:
                        continue
                    n_s = min(a.shape[0] for a in parts)
                    sets.append(np.concatenate([a[:n_s] for a in parts], axis=1))
                    ys = np.concatenate(yy)
                    cols.append(nm)
                if len(sets) < 2 or ys is None:
                    w_r[r] = {nm: 1.0 / len(have) for nm in have}
                    continue
                n_s = min(a.shape[0] for a in sets)
                w = _simplex_weights([a[:n_s] for a in sets], ys)
                w_r[r] = {nm: float(wi) for nm, wi in zip(cols, w)}
            wlog[(rep, f)] = w_r

            # --- apply the weights to this fold ---
            rng = np.random.default_rng(1000 + rep * 97 + f)
            pred = {}
            ok = True
            for j, r in enumerate(RESPONSES):
                names = [nm for nm in have if w_r[r].get(nm, 0) > 1e-6]
                if not names:
                    names = have
                wv = np.array([w_r[r].get(nm, 0.0) for nm in names], float)
                wv = wv / wv.sum() if wv.sum() > 0 else np.ones(len(names)) / len(names)
                n_draw = 2000
                counts = rng.multinomial(n_draw, wv)
                parts = []
                for nm, c in zip(names, counts):
                    if c <= 0:
                        continue
                    rec = S[nm][f]
                    if rec is None or rec["pred"] is None:
                        ok = False
                        break
                    sm = np.asarray(rec["pred"][r]["samples"], float)
                    idx = rng.choice(sm.shape[0], size=c, replace=c > sm.shape[0])
                    parts.append(sm[idx])
                if not ok or not parts:
                    ok = False
                    break
                sm = np.concatenate(parts, axis=0)
                pred[r] = dict(mean=sm.mean(axis=0).astype(np.float32),
                               sd=sm.std(axis=0, ddof=1).astype(np.float32),
                               samples=sm.astype(np.float32))
            if not ok:
                continue
            ref = S[have[0]][f]
            rec = dict(protocol=protocol, rep=rep, fold=f, method=NAME,
                       test=list(np.asarray(te)), n_train=ref["n_train"],
                       n_test=ref["n_test"], seconds=0.0, error=None, pred=pred)
            with open(_key(protocol, rep, f, NAME), "wb") as fh:
                pickle.dump(rec, fh)
            made += 1
    if verbose:
        print(f"  {protocol}: wrote {made} pooled folds from members {have}")
    return made, wlog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocols", default="LOCO,LOO,RKF")
    ap.add_argument("--n-rep", type=int, default=20)
    a = ap.parse_args()
    allw = {}
    for p in [x.strip() for x in a.protocols.split(",")]:
        _, w = build(p, n_rep=a.n_rep)
        if p == "LOCO" and w:
            allw = w
    if allw:
        rows = []
        for (rep, f), wr in allw.items():
            for r, dd in wr.items():
                for nm, v in dd.items():
                    rows.append(dict(fold=f, response=r, member=nm, weight=v))
        df = pd.DataFrame(rows)
        df.to_csv(f"{JD}/results/stack_weights_perfold.csv", index=False)
        piv = df.pivot_table(index="member", columns="response", values="weight")
        print("\nmean pooling weight across LOCO folds:")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
