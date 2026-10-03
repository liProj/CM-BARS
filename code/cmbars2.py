"""CM-BARS zero-aware multitask bounded response surface.

The paper's core configuration is CMBARS2(use_gp=False, warmup=1000,
samples=1000, chains=2). Its hurdle-Beta observation layer models recorded
zeros separately from positive proportions. Signed low-rank coefficient
sharing, task-specific deviations, and blockwise shrinkage regularize the
quadratic process basis. Empirical prior centers use training responses only.
The implementation floors each log likelihood at -60 (a working posterior).
Component benefits are response-dependent; see docs/MODEL.md.
"""
import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import expit, logit
from numpyro.infer import MCMC, NUTS

from methods import GROUP, LOWER, RESPONSES, UPPER, Method, quad_basis

JITTER = 1e-6
SCALE = 100.0        # responses are percentages


def matern52(Z1, Z2, ell):
    d = jnp.sqrt(jnp.maximum(((Z1[:, None, :] - Z2[None, :, :]) ** 2).sum(-1), 1e-24)) / ell
    s5 = jnp.sqrt(5.0) * d
    return (1.0 + s5 + 5.0 / 3.0 * d ** 2) * jnp.exp(-s5)


def model(G, Ztr, Y, at0, p0_init, mu_init, n_factor=2, zero_part=True, coupling="factor",
          use_gp=True, shrink=True, bounded=True):
    n, nb = G.shape
    R = Y.shape[1]
    gi = jnp.asarray(GROUP)

    # ---- hierarchical scales for the three coefficient blocks ----
    if shrink:
        s_grp = numpyro.sample("s_grp", dist.HalfNormal(jnp.ones(3)))
        sj = s_grp[gi]
    else:
        sj = jnp.full(nb, 5.0)

    # ---- multi-task coupling of the surface coefficients ----
    if coupling == "factor":
        Phi = numpyro.sample("Phi", dist.Normal(jnp.zeros((n_factor, nb)), sj[None, :]))
        Lam = numpyro.sample("Lam", dist.Normal(jnp.zeros((R, n_factor)), 1.0))
        kappa = numpyro.sample("kappa", dist.HalfNormal(sj))
        delta = numpyro.sample("delta", dist.Normal(jnp.zeros((R, nb)), kappa[None, :]))
        theta = Lam @ Phi + delta
    elif coupling == "mean":
        mu_j = numpyro.sample("mu_j", dist.Normal(jnp.zeros(nb), sj))
        tau = numpyro.sample("tau", dist.HalfNormal(sj))
        eps = numpyro.sample("eps", dist.Normal(jnp.zeros((R, nb)), 1.0))
        theta = mu_j[None, :] + tau[None, :] * eps
    else:                                                     # independent tasks
        eps = numpyro.sample("eps", dist.Normal(jnp.zeros((R, nb)), 1.0))
        theta = sj[None, :] * eps
    numpyro.deterministic("theta", theta)

    # ---- positive-part mean on a logit link ----
    b0 = numpyro.sample("b0", dist.Normal(jnp.asarray(mu_init), 1.5))
    u = b0[None, :] + (G @ theta.T)                           # (n, R)

    if use_gp:
        ell = numpyro.sample("ell", dist.LogNormal(jnp.log(1.5), 0.5))
        rho = numpyro.sample("rho", dist.HalfNormal(0.3 * jnp.ones(R)))
        K = matern52(Ztr, Ztr, ell) + JITTER * jnp.eye(n)
        L = jnp.linalg.cholesky(K)
        w = numpyro.sample("w", dist.Normal(jnp.zeros((R, n)), 1.0))
        u = u + rho[None, :] * (L @ w.T)
    numpyro.deterministic("u", u)

    conc = numpyro.sample("conc", dist.Gamma(2.0 * jnp.ones(R), 0.08))   # Beta dispersion
    numpyro.deterministic("conc_", conc)

    # ---- zero part: parsimonious, linear in the factors, signed task loading ----
    if zero_part:
        s0 = numpyro.sample("s0", dist.HalfNormal(1.0))
        psi = numpyro.sample("psi", dist.Normal(jnp.zeros(3), s0))
        vload = numpyro.sample("vload", dist.Normal(jnp.zeros(R), 1.0))
        z0 = numpyro.sample("z0", dist.Normal(jnp.asarray(p0_init), 1.5))
        lin0 = z0[None, :] + vload[None, :] * (Ztr @ psi)[:, None]
        logit_pi = jnp.clip(lin0, -12.0, 12.0)
        numpyro.deterministic("logit_pi", logit_pi)
    else:
        logit_pi = None

    # ---- likelihood ----
    if bounded:
        mu = jnp.clip(expit(u), 1e-5, 1 - 1e-5)
        a_b = mu * conc[None, :]
        b_b = (1.0 - mu) * conc[None, :]
        yfrac = jnp.clip(Y / SCALE, 1e-6, 1 - 1e-6)
        ll_pos = dist.Beta(a_b, b_b).log_prob(yfrac)
    else:
        # Gaussian fallback on the raw percentage scale (ablation: no bounded link)
        sig = numpyro.sample("sigma_raw", dist.HalfNormal(10.0 * jnp.ones(R)))
        ll_pos = dist.Normal(u * 10.0, sig[None, :]).log_prob(Y)

    if zero_part:
        lp0 = jax.nn.log_sigmoid(logit_pi)
        lp1 = jax.nn.log_sigmoid(-logit_pi)
        ll = jnp.where(at0, lp0, lp1 + ll_pos)
    else:
        ll = ll_pos
    numpyro.factor("obs", jnp.clip(ll, -60.0, None).sum())


class CMBARS2(Method):
    """Two-part, bounded-link, factor-coupled Bayesian response surface."""
    multitask = True
    probabilistic = True

    def __init__(self, zero_part=True, coupling="factor", use_gp=True, shrink=True,
                 bounded=True, n_factor=2, warmup=600, samples=600, chains=2, seed=0,
                 name=None, target_accept=0.92, max_tree_depth=11):
        self.cfg = dict(zero_part=zero_part, coupling=coupling, use_gp=use_gp,
                        shrink=shrink, bounded=bounded, n_factor=n_factor)
        self.warmup, self.samples, self.chains, self.seed = warmup, samples, chains, seed
        self.target_accept, self.max_tree_depth = target_accept, max_tree_depth
        if name:
            self.name = name
        else:
            off = [k for k, v in [("zero-part", zero_part), ("GP", use_gp),
                                  ("shrinkage", shrink), ("bounded link", bounded)] if not v]
            if coupling != "factor":
                off.append(f"coupling={coupling}")
            self.name = "CM-BARS" if not off else "CM-BARS -" + " -".join(off)

    def fit(self, Z, Y):
        Z = np.asarray(Z, float)
        Y = np.asarray(Y, float)
        self.Ztr_ = Z
        G = quad_basis(Z, include_const=False)
        at0 = Y <= LOWER + 1e-12
        with np.errstate(divide="ignore"):
            frac0 = np.clip(at0.mean(axis=0), 0.02, 0.5)
            p0_init = np.log(frac0 / (1 - frac0))
            pos = np.where(at0, np.nan, Y)
            mpos = np.nanmean(np.where(np.isnan(pos), np.nan, pos), axis=0) / SCALE
            mpos = np.clip(np.nan_to_num(mpos, nan=0.1), 1e-3, 0.9)
            mu_init = np.log(mpos / (1 - mpos))
        self.p0_init_, self.mu_init_ = p0_init, mu_init
        kernel = NUTS(model, target_accept_prob=self.target_accept,
                      max_tree_depth=self.max_tree_depth)
        self.mcmc_ = MCMC(kernel, num_warmup=self.warmup, num_samples=self.samples,
                          num_chains=self.chains, progress_bar=False,
                          chain_method="sequential")
        self.mcmc_.run(jax.random.PRNGKey(self.seed), jnp.asarray(G), jnp.asarray(Z),
                       jnp.asarray(Y), jnp.asarray(at0), jnp.asarray(p0_init),
                       jnp.asarray(mu_init), **self.cfg)
        self.post_ = {k: np.asarray(v) for k, v in self.mcmc_.get_samples().items()}
        return self

    # ---------------- posterior predictive --------------------------------
    def _u_star(self, Zte, max_post=None):
        """Posterior draws of the latent surface at new points, (S, n_te, R).

        `max_post` thins the posterior before the einsum: on a dense grid the full chain
        times the number of grid points does not fit in memory.
        """
        P = self.post_
        if max_post is not None:
            n_s = len(np.asarray(P["b0"]))
            if n_s > max_post:
                idx = np.linspace(0, n_s - 1, max_post).astype(int)
                P = {k: np.asarray(v)[idx] for k, v in P.items()}
        Gs = jnp.asarray(quad_basis(np.asarray(Zte, float), include_const=False))
        b0, theta = jnp.asarray(P["b0"]), jnp.asarray(P["theta"])
        u = b0[:, None, :] + jnp.einsum("bj,srj->sbr", Gs, theta)
        if not self.cfg["use_gp"]:
            return u
        Ztr = jnp.asarray(self.Ztr_)
        Zte_j = jnp.asarray(np.asarray(Zte, float))
        ell, rho, w = jnp.asarray(P["ell"]), jnp.asarray(P["rho"]), jnp.asarray(P["w"])
        n = Ztr.shape[0]

        def one(ell_s, w_s):
            K = matern52(Ztr, Ztr, ell_s) + JITTER * jnp.eye(n)
            L = jnp.linalg.cholesky(K)
            f = L @ w_s.T
            Ks = matern52(Ztr, Zte_j, ell_s)
            A = jax.scipy.linalg.cho_solve((L, True), Ks)
            mean = A.T @ f
            var = jnp.maximum(jnp.diag(matern52(Zte_j, Zte_j, ell_s)) - (Ks * A).sum(0), 1e-10)
            return mean, var

        means, vars_ = jax.vmap(one)(ell, w)
        key = jax.random.PRNGKey(self.seed + 11)
        fstar = means + jnp.sqrt(vars_)[:, :, None] * jax.random.normal(key, means.shape)
        return u + rho[:, None, :] * fstar

    def _pi_star(self, Zte, max_post=None):
        P = self.post_
        if not self.cfg["zero_part"]:
            return None
        if max_post is not None:
            n_s = len(np.asarray(P["z0"]))
            if n_s > max_post:
                idx = np.linspace(0, n_s - 1, max_post).astype(int)
                P = {k: np.asarray(v)[idx] for k, v in P.items()}
        Zte_j = np.asarray(Zte, float)
        z0, psi, v = P["z0"], P["psi"], P["vload"]
        proj = Zte_j @ psi.T                                  # (b, S)
        lin = z0[:, None, :] + v[:, None, :] * proj.T[:, :, None]
        return expit(np.clip(lin, -12, 12))

    def predict(self, Zte, n_draw=2000, max_post=None):
        u = np.asarray(self._u_star(Zte, max_post=max_post))  # (S, b, R)
        S, b, R = u.shape
        rng = np.random.default_rng(self.seed + 3)
        take = rng.choice(S, size=n_draw, replace=n_draw > S)
        u = u[take]
        pi = self._pi_star(Zte, max_post=max_post)
        pi = pi[take] if pi is not None else None
        if self.cfg["bounded"]:
            mu = np.clip(1.0 / (1.0 + np.exp(-u)), 1e-5, 1 - 1e-5)
            conc = self._thin("conc_", max_post)[take]        # (S, R)
            a_b = mu * conc[:, None, :]
            b_b = (1 - mu) * conc[:, None, :]
            draws = rng.beta(np.maximum(a_b, 1e-3), np.maximum(b_b, 1e-3)) * SCALE
        else:
            sig = self._thin("sigma_raw", max_post)[take]
            draws = np.clip(u * 10.0 + sig[:, None, :] * rng.standard_normal(u.shape),
                            LOWER, UPPER)
        if pi is not None:
            draws = np.where(rng.random(draws.shape) < pi, 0.0, draws)
        out = {}
        for j, r in enumerate(RESPONSES):
            sm = draws[:, :, j]
            out[r] = self._pack(sm.mean(axis=0), sd=sm.std(axis=0, ddof=1), samples=sm)
        return out

    def _thin(self, key, max_post):
        v = np.asarray(self.post_[key])
        if max_post is not None and len(v) > max_post:
            v = v[np.linspace(0, len(v) - 1, max_post).astype(int)]
        return v

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
