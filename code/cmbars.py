"""Multitask Tobit Bayesian response surface used as a comparator.

This is the earlier raw-percentage censored-normal implementation, reported
as Tobit-MT-Bayes. The main CM-BARS hurdle-Beta model is in cmbars2.py.
"""
import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.stats import norm as jnorm
from numpyro.infer import MCMC, NUTS

from methods import GROUP, LOWER, RESPONSES, UPPER, Method, quad_basis

# NOTE: deliberately NOT calling numpyro.set_host_device_count(): this module is
# imported inside every cross-validation worker, and extra host devices multiply
# XLA's thread pools across processes (a load average of 200 on 20 cores, and an
# XLA compile that took 58 minutes, were traced to exactly this).
JITTER = 1e-6


def matern52(Z1, Z2, ell):
    d = jnp.sqrt(jnp.maximum(
        ((Z1[:, None, :] - Z2[None, :, :]) ** 2).sum(-1), 1e-24)) / ell
    s5 = jnp.sqrt(5.0) * d
    return (1.0 + s5 + 5.0 / 3.0 * d ** 2) * jnp.exp(-s5)


def model(G, Ztr, Y, at0, amp, ymean, censored=True, multitask=True, use_gp=True,
          shrink=True):
    n, nb = G.shape
    R = Y.shape[1]

    a = numpyro.sample("a", dist.Normal(jnp.asarray(ymean), 2.0 * jnp.asarray(amp)))

    if shrink:
        s_grp = numpyro.sample("s_grp", dist.HalfNormal(jnp.ones(3)))
        sj = s_grp[jnp.asarray(GROUP)]
    else:
        sj = jnp.full(nb, 5.0)

    eps = numpyro.sample("eps", dist.Normal(jnp.zeros((R, nb)), 1.0))
    if multitask:
        mu = numpyro.sample("mu", dist.Normal(jnp.zeros(nb), sj))
        tau = numpyro.sample("tau", dist.HalfNormal(sj))
        theta = mu[None, :] + tau[None, :] * eps
    else:
        theta = sj[None, :] * eps
    numpyro.deterministic("theta", theta)

    lin = a[None, :] + jnp.asarray(amp)[None, :] * (G @ theta.T)

    if use_gp:
        ell = numpyro.sample("ell", dist.LogNormal(jnp.log(1.5), 0.5))
        rho = numpyro.sample("rho", dist.HalfNormal(0.25 * jnp.ones(R)))
        K = matern52(Ztr, Ztr, ell) + JITTER * jnp.eye(n)
        L = jnp.linalg.cholesky(K)
        w = numpyro.sample("w", dist.Normal(jnp.zeros((R, n)), 1.0))
        f = (L @ w.T)                                        # (n, R)
        eta = lin + jnp.asarray(amp)[None, :] * rho[None, :] * f
    else:
        eta = lin
    numpyro.deterministic("eta", eta)

    sfrac = numpyro.sample("sigma_frac", dist.HalfNormal(0.6 * jnp.ones(R)))
    sigma = jnp.asarray(amp) * sfrac
    numpyro.deterministic("sigma", sigma)

    sg = jnp.broadcast_to(sigma[None, :], eta.shape)
    if censored:
        # Tobit type I: mass at the bound for censored cells, density otherwise
        ll = jnp.where(at0,
                       jnorm.logcdf((LOWER - eta) / sg),
                       jnorm.logpdf(Y, eta, sg))
    else:
        ll = jnorm.logpdf(Y, eta, sg)
    numpyro.factor("obs", jnp.clip(ll, -60.0, None).sum())


class CMBARS(Method):
    """Posterior-sampled censoring-aware multi-task response surface."""
    multitask = True
    probabilistic = True

    def __init__(self, censored=True, multitask=True, use_gp=True, shrink=True,
                 warmup=1000, samples=1000, chains=2, seed=0, name=None,
                 target_accept=0.9, max_tree_depth=10):
        self.cfg = dict(censored=censored, multitask=multitask, use_gp=use_gp,
                        shrink=shrink)
        self.warmup, self.samples, self.chains, self.seed = warmup, samples, chains, seed
        self.target_accept, self.max_tree_depth = target_accept, max_tree_depth
        tag = "".join(k for k, v in [("c", censored), ("m", multitask),
                                     ("g", use_gp), ("s", shrink)] if v)
        self.name = name or (f"CM-BARS" if tag == "cmgs" else f"CM-BARS[{tag or 'none'}]")

    def fit(self, Z, Y):
        Z = np.asarray(Z, float)
        Y = np.asarray(Y, float)
        self.Ztr_ = Z
        self.amp_ = np.maximum(Y.std(axis=0, ddof=1), 1e-2)
        self.ymean_ = Y.mean(axis=0)
        G = quad_basis(Z, include_const=False)
        at0 = Y <= LOWER + 1e-12
        kernel = NUTS(model, target_accept_prob=self.target_accept,
                      max_tree_depth=self.max_tree_depth)
        self.mcmc_ = MCMC(kernel, num_warmup=self.warmup, num_samples=self.samples,
                          num_chains=self.chains, progress_bar=False,
                          chain_method="sequential")
        self.mcmc_.run(jax.random.PRNGKey(self.seed), jnp.asarray(G), jnp.asarray(Z),
                       jnp.asarray(Y), jnp.asarray(at0), jnp.asarray(self.amp_),
                       jnp.asarray(self.ymean_), **self.cfg)
        self.post_ = {k: np.asarray(v) for k, v in self.mcmc_.get_samples().items()}
        return self

    # ---- posterior predictive -------------------------------------------
    def _eta_star(self, Zte):
        """Posterior draws of the latent surface at new points, (S, n_te, R)."""
        P = self.post_
        Gs = jnp.asarray(quad_basis(np.asarray(Zte, float), include_const=False))
        a, theta = jnp.asarray(P["a"]), jnp.asarray(P["theta"])
        amp = jnp.asarray(self.amp_)
        lin = a[:, None, :] + amp[None, None, :] * jnp.einsum("bj,srj->sbr", Gs, theta)
        if not self.cfg["use_gp"]:
            return lin
        Ztr = jnp.asarray(self.Ztr_)
        Zte_j = jnp.asarray(np.asarray(Zte, float))
        ell, rho, w = (jnp.asarray(P["ell"]), jnp.asarray(P["rho"]), jnp.asarray(P["w"]))
        n = Ztr.shape[0]

        def one(ell_s, rho_s, w_s):
            K = matern52(Ztr, Ztr, ell_s) + JITTER * jnp.eye(n)
            L = jnp.linalg.cholesky(K)
            f = L @ w_s.T                                    # (n, R)
            Ks = matern52(Ztr, Zte_j, ell_s)                 # (n, b)
            A = jax.scipy.linalg.cho_solve((L, True), Ks)    # (n, b)
            mean = A.T @ f                                   # (b, R)
            Kss = matern52(Zte_j, Zte_j, ell_s)
            var = jnp.maximum(jnp.diag(Kss) - (Ks * A).sum(0), 1e-10)
            return mean, var, rho_s

        means, vars_, rhos = jax.vmap(one)(ell, rho, w)
        key = jax.random.PRNGKey(self.seed + 7)
        z = jax.random.normal(key, means.shape)
        fstar = means + jnp.sqrt(vars_)[:, :, None] * z
        return lin + amp[None, None, :] * rhos[:, None, :] * fstar

    def predict(self, Zte, n_draw=2000):
        eta = np.asarray(self._eta_star(Zte))                # (S, b, R)
        sig = self.post_["sigma"]                            # (S, R)
        S, b, R = eta.shape
        rng = np.random.default_rng(self.seed + 1)
        take = rng.choice(S, size=min(n_draw, S) if n_draw <= S else n_draw,
                          replace=n_draw > S)
        e = eta[take]
        s = sig[take]
        draws = e + s[:, None, :] * rng.standard_normal(e.shape)
        if self.cfg["censored"]:
            draws = np.clip(draws, LOWER, UPPER)
        out = {}
        for j, r in enumerate(RESPONSES):
            sm = draws[:, :, j]
            out[r] = self._pack(sm.mean(axis=0), sd=sm.std(axis=0, ddof=1), samples=sm)
        return out

    def diagnostics(self):
        import numpyro.diagnostics as nd
        g = self.mcmc_.get_samples(group_by_chain=True)
        rows = []
        for k, v in g.items():
            v = np.asarray(v)
            flat = v.reshape(v.shape[0], v.shape[1], -1)
            for i in range(flat.shape[2]):
                x = flat[:, :, i]
                rows.append(dict(param=f"{k}[{i}]" if flat.shape[2] > 1 else k,
                                 r_hat=float(nd.gelman_rubin(x)) if x.shape[0] > 1 else np.nan,
                                 ess=float(nd.effective_sample_size(x))))
        return rows
