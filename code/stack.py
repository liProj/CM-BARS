"""Predictive distribution pooling and mean-surface distillation.

CMBARSPlus.fit implements grouped inner-fold weight fitting and refits members
on its supplied training set. This class is distinct from stack_oof.py, which
produced the historical cached outer-fold results in the paper supplement.
Weights minimize a regularized CRPS objective per compound. Distillation
approximates the teacher mean with a bounded logit-quadratic expression.
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)

import methods as M                                              # noqa: E402
from metrics import crps_censored_samples                         # noqa: E402

RESPONSES = M.RESPONSES
SPEC = f"{JD}/results/stack_spec.json"
DISTILL = f"{JD}/results/distilled_coefficients.json"


def default_members():
    """Member constructors.  Kept deliberately small and diverse."""
    from cmbars2 import CMBARS2
    mem = {
        "CM-BARS": lambda: CMBARS2(use_gp=False, warmup=600, samples=600, chains=2),
        "ExtraTrees": M.ExtraTrees,
        "LightGBM": M.LightGBM,
        "GP-Matern": M.GPMatern,
        "OLS-Quad": M.OLSQuadClip,
    }
    if os.path.exists(f"{JD}/results/pfn_rsm.pt"):
        from pfn import PFNRSM
        mem["PFN-RSM"] = PFNRSM
    return mem


def _inner_folds(Z, k=4, seed=0):
    """Grouped k-fold over the unique conditions of the training set."""
    key = [tuple(np.round(r, 9)) for r in np.asarray(Z, float)]
    uniq = {}
    g = []
    for t in key:
        uniq.setdefault(t, len(uniq))
        g.append(uniq[t])
    g = np.asarray(g)
    nc = g.max() + 1
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(nc), min(k, nc))
    out = []
    for ch in chunks:
        te = np.where(np.isin(g, ch))[0]
        tr = np.setdiff1d(np.arange(len(g)), te)
        if len(te) and len(tr) >= 6:
            out.append((tr, te))
    return out


def _simplex_weights(sample_sets, y, l2=0.05, n_sub=400, seed=0):
    """Weights on the simplex minimising the CRPS of the pooled predictive distribution.

    For a linear pool p_w = sum_k w_k p_k the CRPS has a closed form that is quadratic in
    the weights,

        CRPS(w) = w . A  -  0.5 * w' B w,
        A_k = E|X_k - y|,   B_kl = E|X_k - X_l'| ,   X_k ~ p_k independent,

    and the energy-distance property of the second term makes it convex on the simplex, so
    a single constrained solve finds the global optimum.

    The weights are additionally shrunk towards the uniform vector. With a dozen conditions
    the CRPS-optimal weights are noisy, and because every outer fold gets its own weight
    vector that noise would otherwise make predictions from different folds mutually
    inconsistent, which damages statistics computed on the pooled out-of-fold vector --
    rank correlation in particular.
    """
    K = len(sample_sets)
    if K == 1:
        return np.ones(1)
    rng = np.random.default_rng(seed)
    S = []
    for a in sample_sets:
        a = np.asarray(a, float)
        idx = rng.choice(a.shape[0], size=min(n_sub, a.shape[0]), replace=False)
        S.append(a[idx])
    y = np.asarray(y, float)
    A = np.array([np.abs(s - y[None, :]).mean() for s in S])
    B = np.zeros((K, K))
    for k in range(K):
        for l in range(k, K):
            B[k, l] = B[l, k] = np.abs(S[k][:, None, :] - S[l][None, :, :]).mean()

    u = np.ones(K) / K
    lam = l2 * max(float(np.abs(A).mean()), 1e-9) * K

    def obj(w):
        return float(w @ A - 0.5 * w @ B @ w + lam * ((w - u) @ (w - u)))

    def grad(w):
        return A - B @ w + 2 * lam * (w - u)

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones_like(w)},)
    bounds = [(0.0, 1.0)] * K
    best, bw = np.inf, u.copy()
    starts = [u] + [np.eye(K)[k] * 0.7 + 0.3 / K for k in range(K)]
    for w0 in starts:
        try:
            r = minimize(obj, w0 / w0.sum(), jac=grad, bounds=bounds, constraints=cons,
                         method="SLSQP", options=dict(maxiter=200, ftol=1e-9))
        except Exception:
            continue
        if r.success and r.fun < best:
            best = r.fun
            w = np.clip(r.x, 0.0, None)
            bw = w / max(w.sum(), 1e-12)
    return bw


class CMBARSPlus(M.Method):
    """Linear pool of member predictive distributions, weighted per response."""
    name = "CM-BARS+ (stacked)"
    multitask = True
    probabilistic = True

    def __init__(self, members=None, inner_k=4, seed=0, name=None, save_spec=False):
        self.members = members or default_members()
        self.inner_k, self.seed, self.save_spec = inner_k, seed, save_spec
        if name:
            self.name = name

    def fit(self, Z, Y):
        Z = np.asarray(Z, float)
        Y = np.asarray(Y, float)
        n = len(Z)
        names = list(self.members)
        # ---- inner out-of-fold predictive samples per member ----
        oof = {nm: {r: None for r in RESPONSES} for nm in names}
        covered = np.zeros(n, bool)
        for tr, te in _inner_folds(Z, self.inner_k, self.seed):
            for nm in names:
                mdl = self.members[nm]()
                mdl.fit(Z[tr], Y[tr])
                pr = mdl.predict(Z[te])
                for r in RESPONSES:
                    s = pr[r]["samples"]
                    if oof[nm][r] is None:
                        oof[nm][r] = np.full((s.shape[0], n), np.nan, np.float32)
                    ns = min(s.shape[0], oof[nm][r].shape[0])
                    oof[nm][r][:ns, te] = s[:ns]
            covered[te] = True
        self.weights_ = {}
        for j, r in enumerate(RESPONSES):
            sets, keep = [], []
            for nm in names:
                a = oof[nm][r]
                if a is None:
                    continue
                ok = covered & ~np.isnan(a).any(axis=0)
                if ok.sum() < 6:
                    continue
                sets.append(a[:, ok])
                keep.append(nm)
            if not sets:
                self.weights_[r] = {nm: 1.0 / len(names) for nm in names}
                continue
            ok = covered & ~np.isnan(oof[keep[0]][r]).any(axis=0)
            w = _simplex_weights(sets, Y[ok, j])
            self.weights_[r] = {nm: float(wi) for nm, wi in zip(keep, w)}
        # ---- refit members on the whole training set ----
        self.fitted_ = {}
        for nm in names:
            mdl = self.members[nm]()
            mdl.fit(Z, Y)
            self.fitted_[nm] = mdl
        if self.save_spec:
            with open(SPEC, "w") as fh:
                json.dump(self.weights_, fh, indent=1)
        return self

    def predict(self, Zte, n_draw=2000):
        Zte = np.atleast_2d(np.asarray(Zte, float))
        preds = {nm: m.predict(Zte) for nm, m in self.fitted_.items()}
        rng = np.random.default_rng(self.seed + 17)
        out = {}
        for r in RESPONSES:
            w = self.weights_.get(r, {})
            names = [nm for nm in preds if w.get(nm, 0) > 1e-6]
            if not names:
                names = list(preds)
                wv = np.ones(len(names)) / len(names)
            else:
                wv = np.array([w[nm] for nm in names], float)
                wv = wv / wv.sum()
            counts = rng.multinomial(n_draw, wv)
            parts = []
            for nm, c in zip(names, counts):
                if c <= 0:
                    continue
                S = preds[nm][r]["samples"]
                idx = rng.choice(S.shape[0], size=c, replace=c > S.shape[0])
                parts.append(np.asarray(S)[idx])
            sm = np.concatenate(parts, axis=0)
            out[r] = self._pack(sm.mean(axis=0), sd=sm.std(axis=0, ddof=1), samples=sm)
        return out


class Distilled(M.Method):
    """Closed-form logit-quadratic fitted to the stacked posterior-mean surface.

    The teacher (CM-BARS+) is evaluated on a dense grid over the design box and a quadratic
    in the three coded factors is fitted to logit(mean/100) by least squares.  The student
    is one printable equation per phenolic compound that is bounded in [0, 100] by
    construction.
    """
    name = "CM-BARS+ distilled"
    multitask = True
    probabilistic = True

    def __init__(self, teacher=None, grid=11, seed=0, name=None, save=False):
        self.teacher = teacher
        self.grid, self.seed, self.save = grid, seed, save
        if name:
            self.name = name

    @staticmethod
    def _grid(m):
        g = np.linspace(-1, 1, m)
        A, B, C = np.meshgrid(g, g, g, indexing="ij")
        return np.column_stack([A.ravel(), B.ravel(), C.ravel()])

    def fit(self, Z, Y):
        t = self.teacher or CMBARSPlus(seed=self.seed)
        t.fit(Z, Y)
        self.teacher_ = t
        Zg = self._grid(self.grid)
        pr = t.predict(Zg, n_draw=800)
        X = M.quad_basis(Zg)
        self.coef_, self.sd_ = {}, {}
        self.distill_rmse_ = {}
        for r in RESPONSES:
            mu = np.clip(pr[r]["mean"] / 100.0, 1e-4, 1 - 1e-4)
            u = np.log(mu / (1 - mu))
            b, *_ = np.linalg.lstsq(X, u, rcond=None)
            self.coef_[r] = b
            back = 100.0 / (1.0 + np.exp(-(X @ b)))
            self.distill_rmse_[r] = float(np.sqrt(np.mean((back - pr[r]["mean"]) ** 2)))
            self.sd_[r] = float(np.mean(pr[r]["sd"]))
        if self.save:
            with open(DISTILL, "w") as fh:
                json.dump({r: dict(coef=self.coef_[r].tolist(),
                                   sd=self.sd_[r],
                                   distill_rmse=self.distill_rmse_[r])
                           for r in RESPONSES}, fh, indent=1)
        return self

    def predict(self, Zte, n_draw=2000):
        X = M.quad_basis(np.atleast_2d(np.asarray(Zte, float)))
        rng = np.random.default_rng(self.seed + 23)
        out = {}
        for r in RESPONSES:
            mu = 100.0 / (1.0 + np.exp(-(X @ self.coef_[r])))
            s = max(self.sd_[r], 1e-3)
            sm = np.clip(mu[None, :] + s * rng.standard_normal((n_draw, len(mu))), 0, 100)
            out[r] = self._pack(mu, sd=np.full(len(mu), s), samples=sm)
        return out
