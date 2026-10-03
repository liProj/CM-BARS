"""Common estimator interface and conventional regression comparators.

Inputs are coded factors (n, 3) and percentage responses (n, 3), ordered as
tannic acid, p-coumaric acid, and acetosyringone. Outputs contain predictive
means, standard deviations, and/or samples for each response. Recorded zeros
are observations; the main model treats them with a hurdle-Beta likelihood.
"""
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
RESPONSES = ["tannic_acid", "p_coumaric_acid", "acetosyringone"]
LOWER, UPPER = 0.0, 100.0

# factor ranges, used to map real units <-> coded units in [-1, 1]
FACTOR_RANGE = {"pH": (4.0, 8.0), "additive_mM": (5.0, 15.0), "cnf_pct": (0.10, 0.40)}


def to_coded(pH, conc, cnf):
    """Map real factor values to coded units in [-1, 1]."""
    def c(v, lo, hi):
        return 2 * (np.asarray(v, float) - lo) / (hi - lo) - 1
    Z = np.column_stack([c(pH, *FACTOR_RANGE["pH"]),
                         c(conc, *FACTOR_RANGE["additive_mM"]),
                         c(cnf, *FACTOR_RANGE["cnf_pct"])])
    return np.round(Z, 12)   # the design levels are exactly -1, 0, +1


def from_coded(Z):
    Z = np.atleast_2d(np.asarray(Z, float))
    out = []
    for k, key in enumerate(["pH", "additive_mM", "cnf_pct"]):
        lo, hi = FACTOR_RANGE[key]
        out.append(lo + (Z[:, k] + 1) * (hi - lo) / 2)
    return np.column_stack(out)


def quad_basis(Z, include_const=True):
    """Full second-order basis: [1,] z1 z2 z3 z1^2 z2^2 z3^2 z1z2 z1z3 z2z3."""
    Z = np.atleast_2d(np.asarray(Z, float))
    z1, z2, z3 = Z[:, 0], Z[:, 1], Z[:, 2]
    cols = [z1, z2, z3, z1 * z1, z2 * z2, z3 * z3, z1 * z2, z1 * z3, z2 * z3]
    if include_const:
        cols = [np.ones_like(z1)] + cols
    return np.column_stack(cols)


BASIS_NAMES = ["pH", "conc", "cnf", "pH^2", "conc^2", "cnf^2",
               "pH:conc", "pH:cnf", "conc:cnf"]
GROUP = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])          # linear / quadratic / interaction


# ---------------------------------------------------------------- base class
class Method:
    name = "base"
    multitask = False
    probabilistic = False

    def fit(self, Z, Y):
        raise NotImplementedError

    def predict(self, Z):
        raise NotImplementedError

    @staticmethod
    def _pack(mean, sd=None, samples=None):
        out = {"mean": np.asarray(mean, float)}
        if sd is not None:
            out["sd"] = np.asarray(sd, float)
        if samples is not None:
            out["samples"] = np.asarray(samples, float)
        return out


class _PerResponse(Method):
    """Wraps a single-response regressor so it behaves like a multi-response method."""

    def fit(self, Z, Y):
        self.models_ = {}
        for j, r in enumerate(RESPONSES):
            self.models_[r] = self._fit_one(np.asarray(Z, float), np.asarray(Y, float)[:, j], r)
        return self

    def predict(self, Z):
        return {r: self._predict_one(self.models_[r], np.asarray(Z, float), r)
                for r in RESPONSES}


# ---------------------------------------------------------------- the published method
class OLSQuad(_PerResponse):
    """The reference paper's method: full quadratic fitted by ordinary least squares.

    Predictive spread is taken as the residual standard error, which is the conventional
    way to attach an interval to this model and is needed so that the probabilistic
    metrics can be computed for it at all.
    """
    name = "OLS-Quad (published)"
    probabilistic = True
    clip = False

    def _fit_one(self, Z, y, r):
        X = quad_basis(Z)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        dfe = max(1, len(y) - X.shape[1])
        sse = float(((y - X @ beta) ** 2).sum())
        return beta, np.sqrt(sse / dfe) if dfe > 0 else y.std(ddof=1)

    def _predict_one(self, mdl, Z, r):
        beta, s = mdl
        mu = quad_basis(Z) @ beta
        if self.clip:
            mu = np.clip(mu, LOWER, UPPER)
        rng = np.random.default_rng(0)
        samples = mu[None, :] + s * rng.standard_normal((2000, len(mu)))
        if self.clip:
            samples = np.clip(samples, LOWER, UPPER)
        return self._pack(mu, sd=np.full(len(mu), max(s, 1e-6)), samples=samples)


class OLSQuadClip(OLSQuad):
    """Same model, predictions truncated to the physically attainable range [0, 100]."""
    name = "OLS-Quad + clip"
    clip = True


# ---------------------------------------------------------------- penalised linear models
class _SkLinear(_PerResponse):
    probabilistic = True

    def _make(self):
        raise NotImplementedError

    def _fit_one(self, Z, y, r):
        from sklearn.model_selection import KFold
        X = quad_basis(Z, include_const=False)
        mdl = self._make()
        mdl.fit(X, y)
        # residual spread by k-fold, so the interval is not the in-sample one
        k = min(5, len(y))
        resid = []
        for tr, te in KFold(k, shuffle=True, random_state=0).split(X):
            m2 = self._make()
            m2.fit(X[tr], y[tr])
            resid.append(y[te] - m2.predict(X[te]))
        return mdl, max(float(np.concatenate(resid).std(ddof=1)), 1e-3)

    def _predict_one(self, mdl, Z, r):
        m, s = mdl
        mu = m.predict(quad_basis(Z, include_const=False))
        rng = np.random.default_rng(0)
        sm = np.clip(mu[None, :] + s * rng.standard_normal((2000, len(mu))), LOWER, UPPER)
        return self._pack(mu, sd=np.full(len(mu), s), samples=sm)


class RidgeQuad(_SkLinear):
    name = "Ridge-Quad"

    def _make(self):
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 3, 25)))


class LassoQuad(_SkLinear):
    name = "Lasso-Quad"

    def _make(self):
        from sklearn.linear_model import LassoCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LassoCV(alphas=40, cv=3, random_state=0, max_iter=20000))


class ENetQuad(_SkLinear):
    name = "ElasticNet-Quad"

    def _make(self):
        from sklearn.linear_model import ElasticNetCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             ElasticNetCV(l1_ratio=[.1, .5, .9, 1.], alphas=30, cv=3,
                                          random_state=0, max_iter=20000))


class PLSQuad(_SkLinear):
    name = "PLS-Quad"

    def _make(self):
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        class _P(PLSRegression):
            def predict(self, X, **kw):
                return np.asarray(super().predict(X, **kw)).ravel()
        return make_pipeline(StandardScaler(), _P(n_components=2))


# ---------------------------------------------------------------- tree ensembles
class _Tree(_PerResponse):
    probabilistic = True

    def _fit_one(self, Z, y, r):
        m = self._make()
        m.fit(Z, y)
        from sklearn.model_selection import KFold
        resid = []
        for tr, te in KFold(min(5, len(y)), shuffle=True, random_state=0).split(Z):
            m2 = self._make()
            m2.fit(Z[tr], y[tr])
            resid.append(y[te] - m2.predict(Z[te]))
        return m, max(float(np.concatenate(resid).std(ddof=1)), 1e-3)

    def _predict_one(self, mdl, Z, r):
        m, s = mdl
        mu = np.asarray(m.predict(Z), float)
        rng = np.random.default_rng(0)
        sm = np.clip(mu[None, :] + s * rng.standard_normal((2000, len(mu))), LOWER, UPPER)
        return self._pack(mu, sd=np.full(len(mu), s), samples=sm)


class RandomForest(_Tree):
    name = "RandomForest"

    def _make(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=500, random_state=0, n_jobs=1)


class ExtraTrees(_Tree):
    name = "ExtraTrees"

    def _make(self):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(n_estimators=500, random_state=0, n_jobs=1)


class LightGBM(_Tree):
    name = "LightGBM"

    def _make(self):
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=4,
                                 min_child_samples=2, verbose=-1, random_state=0, n_jobs=1)


class XGBoost(_Tree):
    name = "XGBoost"

    def _make(self):
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=2,
                                min_child_weight=1, subsample=0.9, colsample_bytree=0.9,
                                reg_lambda=1.0, random_state=0, n_jobs=1, verbosity=0)


# ---------------------------------------------------------------- Gaussian process
class GPMatern(_PerResponse):
    name = "GP-Matern"
    probabilistic = True

    def _fit_one(self, Z, y, r):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        amp = max(y.std(ddof=1), 1e-3)
        k = (ConstantKernel(amp ** 2, (1e-4, 1e6))
             * Matern(length_scale=1.5, length_scale_bounds=(0.2, 20.0), nu=2.5)
             + WhiteKernel(noise_level=0.25 * amp ** 2, noise_level_bounds=(1e-6, 1e6)))
        g = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=8,
                                     random_state=0)
        g.fit(Z, y)
        return g

    def _predict_one(self, g, Z, r):
        mu, sd = g.predict(Z, return_std=True)
        sd = np.maximum(sd, 1e-6)
        rng = np.random.default_rng(0)
        sm = np.clip(mu[None, :] + sd[None, :] * rng.standard_normal((2000, len(mu))),
                     LOWER, UPPER)
        return self._pack(mu, sd=sd, samples=sm)


# ---------------------------------------------------------------- Tobit quadratic (MLE)
class TobitQuad(_PerResponse):
    """Censored-normal (Tobit) quadratic fitted by maximum likelihood, one response at a time.

    Isolates the effect of respecting the censoring at zero, without any of the
    multi-task sharing or shrinkage that CMBARS adds.
    """
    name = "Tobit-Quad"
    probabilistic = True

    def _fit_one(self, Z, y, r):
        from scipy.optimize import minimize
        from scipy.stats import norm
        X = quad_basis(Z)
        at = y <= LOWER + 1e-12
        b0, *_ = np.linalg.lstsq(X, y, rcond=None)
        s0 = max(np.sqrt(((y - X @ b0) ** 2).mean()), 1e-2)

        def nll(th):
            b, ls = th[:-1], th[-1]
            s = np.exp(np.clip(ls, -6, 6))
            mu = X @ b
            ll = np.where(at, norm.logcdf((LOWER - mu) / s), norm.logpdf(y, mu, s))
            return -np.sum(np.clip(ll, -60, None)) + 1e-3 * float(b[1:] @ b[1:])

        res = minimize(nll, np.r_[b0, np.log(s0)], method="L-BFGS-B",
                       options=dict(maxiter=4000))
        b = res.x[:-1]
        s = float(np.exp(np.clip(res.x[-1], -6, 6)))
        return b, s

    def _predict_one(self, mdl, Z, r):
        b, s = mdl
        eta = quad_basis(Z) @ b
        rng = np.random.default_rng(0)
        sm = np.clip(eta[None, :] + s * rng.standard_normal((2000, len(eta))), LOWER, UPPER)
        return self._pack(sm.mean(axis=0), sd=np.full(len(eta), s), samples=sm)
