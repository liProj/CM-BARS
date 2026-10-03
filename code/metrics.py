"""Evaluation metrics for the censored percentage responses of Macromol #22.

Point metrics mirror what the reference paper reports (R-squared) and add the out-of-sample
and probabilistic metrics it omits.  `HIGHER_BETTER` gives the sign convention used by the
paired tests: +1 means larger is better, -1 means smaller is better.
"""
import numpy as np
from scipy import stats
from scipy.stats import norm, spearmanr

HIGHER_BETTER = {
    "RMSE": -1, "MAE": -1, "R2": +1, "Spearman": +1, "MedAE": -1,
    "CRPS": -1, "NLL": -1, "Coverage90_err": -1, "IntervalWidth90": -1,
    "ZeroBalAcc": +1, "FracNegative": -1, "MaxAbsErr": -1,
}
PRIMARY = ["RMSE", "MAE", "R2", "Spearman", "CRPS", "NLL", "Coverage90_err", "ZeroBalAcc"]


def crps_gaussian(y, mu, sd):
    """Closed-form CRPS of a Gaussian forecast (Gneiting & Raftery 2007)."""
    sd = np.maximum(np.asarray(sd, float), 1e-9)
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sd
    return float(np.mean(sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def crps_censored_samples(y, samples):
    """CRPS from posterior-predictive samples, via the empirical-CDF energy form.

    CRPS = E|X - y| - 0.5 E|X - X'|, estimated with the sorted-sample identity so that
    point masses at zero (from censoring) are handled exactly.
    """
    y = np.asarray(y, float)
    out = []
    for i in range(len(y)):
        x = np.sort(np.asarray(samples[:, i], float))
        n = len(x)
        term1 = np.abs(x - y[i]).mean()
        # E|X - X'| for an empirical distribution = (2/n^2) * sum_i (2i - n + 1) x_(i)
        w = 2 * np.arange(1, n + 1) - n - 1
        term2 = 2.0 * (w * x).sum() / (n * n)
        out.append(term1 - 0.5 * term2)
    return float(np.mean(out))


def nll_censored_normal(y, mu, sd, lower=0.0):
    """Negative log-likelihood under a Tobit (type I) normal censored at `lower`.

    Observations exactly at the censoring bound contribute the probability mass
    Phi((lower - mu)/sd); interior observations contribute the normal density.
    """
    y = np.asarray(y, float); mu = np.asarray(mu, float)
    sd = np.maximum(np.asarray(sd, float), 1e-9)
    at = y <= lower + 1e-12
    ll = np.empty_like(y)
    ll[at] = norm.logcdf((lower - mu[at]) / sd[at])
    ll[~at] = norm.logpdf(y[~at], mu[~at], sd[~at])
    return float(-np.mean(np.clip(ll, -50, None)))


def nll_from_samples(y, samples, lower=0.0, bw=None):
    """NLL of a predictive sample set: point mass at the bound + kernel density above it."""
    y = np.asarray(y, float)
    out = []
    for i in range(len(y)):
        x = np.asarray(samples[:, i], float)
        p0 = float((x <= lower + 1e-12).mean())
        if y[i] <= lower + 1e-12:
            out.append(np.log(max(p0, 1e-6)))
        else:
            pos = x[x > lower + 1e-12]
            if len(pos) < 5:
                out.append(np.log(1e-6))
                continue
            h = bw or max(1.06 * pos.std(ddof=1) * len(pos) ** (-0.2), 1e-3)
            dens = np.exp(-0.5 * ((y[i] - pos) / h) ** 2).sum() / (len(pos) * h * np.sqrt(2 * np.pi))
            out.append(np.log(max((1 - p0) * dens, 1e-6)))
    return float(-np.mean(np.clip(out, -50, None)))


def point_metrics(y, pred):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    err = pred - y
    r2_den = ((y - y.mean()) ** 2).sum()
    sp = spearmanr(y, pred).statistic if len(y) > 2 and np.ptp(pred) > 0 else np.nan
    return dict(RMSE=float(np.sqrt((err ** 2).mean())),
                MAE=float(np.abs(err).mean()),
                MedAE=float(np.median(np.abs(err))),
                MaxAbsErr=float(np.abs(err).max()),
                R2=float(1 - (err ** 2).sum() / r2_den) if r2_den > 0 else np.nan,
                Spearman=float(sp) if sp is not None else np.nan,
                FracNegative=float((pred < -1e-9).mean()))


def zero_balanced_accuracy(y, pred, thr=0.5, lower=0.0, p_zero=None):
    """Balanced accuracy of detecting a censored observation (y == 0).

    The event is scored from the model's own predictive probability of it,
    P(y <= lower), which is the standard way to score a binary event from a predictive
    distribution and is well defined for every method compared here: for a sampled
    predictive it is the mass at or below the floor, and for the published model's
    Gaussian predictive it is Phi(-mu/sigma). Thresholding a *clipped point prediction*
    instead would reward a model for predicting physically impossible negative adsorption,
    which is precisely the defect under study.

    Returns NaN when the response has no censored observations at all (tannic acid), so
    the metric is only averaged where it is defined.
    """
    yz = np.asarray(y, float) <= lower + 1e-12
    if yz.all() or (~yz).all():
        return np.nan
    if p_zero is None:
        pz = np.asarray(pred, float) <= thr
    else:
        pz = np.asarray(p_zero, float) >= 0.5
    tpr = pz[yz].mean()
    tnr = (~pz[~yz]).mean()
    return float(0.5 * (tpr + tnr))


def interval_metrics(y, lo, hi, nominal=0.90):
    y = np.asarray(y, float)
    cov = float(((y >= np.asarray(lo)) & (y <= np.asarray(hi))).mean())
    return dict(Coverage90=cov,
                Coverage90_err=float(abs(cov - nominal)),
                IntervalWidth90=float(np.mean(np.asarray(hi) - np.asarray(lo))))


def all_metrics(y, pred, samples=None, sd=None, lo=None, hi=None, lower=0.0):
    """Assemble every metric that the supplied prediction object can support."""
    m = point_metrics(y, pred)
    p_zero = None
    if samples is not None:
        p_zero = (np.asarray(samples, float) <= lower + 1e-9).mean(axis=0)
    elif sd is not None:
        p_zero = norm.cdf((lower - np.asarray(pred, float))
                          / np.maximum(np.asarray(sd, float), 1e-9))
    m["ZeroBalAcc"] = zero_balanced_accuracy(y, pred, thr=0.5, lower=lower, p_zero=p_zero)
    if samples is not None:
        m["CRPS"] = crps_censored_samples(y, samples)
        m["NLL"] = nll_from_samples(y, samples, lower=lower)
        if lo is None:
            lo = np.quantile(samples, 0.05, axis=0)
            hi = np.quantile(samples, 0.95, axis=0)
    elif sd is not None:
        m["CRPS"] = crps_gaussian(y, pred, sd)
        m["NLL"] = nll_censored_normal(y, pred, sd, lower=lower)
        if lo is None:
            lo = pred - 1.645 * np.asarray(sd)
            hi = pred + 1.645 * np.asarray(sd)
    if lo is not None and hi is not None:
        m.update(interval_metrics(y, lo, hi))
    return m
