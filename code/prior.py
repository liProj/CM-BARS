"""Generative prior over censored multi-task response surfaces.

Used twice:
  * as the training distribution of the amortized network in `npe.py` (prior-fitted /
    in-context learning: the network is trained on surfaces drawn from this prior and then
    applied to the real 15-run design without any further fitting);
  * as the data-generating process of the simulation study in `simulate.py`.

The prior mirrors the CM-BARS model so that the amortized network approximates its
posterior predictive, and it has switches so the simulation study can also generate data
under the *published* model's assumptions (`regime="ols"`: a plain uncensored quadratic
with homoscedastic noise), which is the regime where the reference method should win.

Everything is vectorised over the batch dimension and runs on the GPU.
"""
import numpy as np
import torch

# the published 3-factor Box-Behnken design in coded units (Table 1)
BBD = np.array([
    [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
    [-1, 0, -1], [1, 0, -1], [-1, 0, 1], [1, 0, 1],
    [0, -1, -1], [0, 1, -1], [0, -1, 1], [0, 1, 1],
    [0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
GROUP_T = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
N_BASIS = 9
LOWER, UPPER = 0.0, 100.0


def basis(Z):
    """(..., 3) -> (..., 9) second-order basis without the constant."""
    z1, z2, z3 = Z[..., 0], Z[..., 1], Z[..., 2]
    return torch.stack([z1, z2, z3, z1 * z1, z2 * z2, z3 * z3,
                        z1 * z2, z1 * z3, z2 * z3], dim=-1)


def matern52(Z1, Z2, ell):
    """Matern-5/2 correlation. Z1 (B,n,3), Z2 (B,m,3), ell (B,) -> (B,n,m)."""
    d2 = ((Z1[:, :, None, :] - Z2[:, None, :, :]) ** 2).sum(-1).clamp_min(1e-24)
    d = d2.sqrt() / ell[:, None, None]
    s5 = np.sqrt(5.0) * d
    return (1 + s5 + 5.0 / 3.0 * d ** 2) * torch.exp(-s5)


def _half_normal(shape, scale, gen, device, dtype):
    return (torch.randn(shape, generator=gen, device=device, dtype=dtype) * scale).abs()


def sample_surfaces(B, Z, gen, device, n_task=3, regime="cmbars", dtype=torch.float32,
                    censor=True):
    """Draw B multi-task surfaces and evaluate them at the design points Z.

    Z        (n, 3) coded design, shared across the batch.
    regime   "cmbars"  hierarchical multi-task quadratic + Matern GP deviation, censored
             "ols"     independent plain quadratics, homoscedastic, NOT censored
             "gp"      GP-dominated surfaces with a weak quadratic trend
             "sparse"  only linear + one interaction active (a strongly shrinkable truth)
    Returns dict with `y` (B, n, R), `eta` (clean latent, same shape) and the parameters.
    """
    n = Z.shape[0]
    Zt = torch.as_tensor(Z, device=device, dtype=dtype)
    Zb = Zt[None].expand(B, n, 3)
    G = basis(Zb)                                                     # (B, n, 9)
    R = n_task

    # response level and amplitude: percentages, spanning the range seen in the paper
    amp = torch.exp(torch.randn(B, R, generator=gen, device=device, dtype=dtype) * 0.7
                    + np.log(6.0))                                    # ~ 1.5 .. 25 %
    a = (torch.rand(B, R, generator=gen, device=device, dtype=dtype) * 55.0 + 2.0)

    if regime == "ols":
        theta = torch.randn(B, R, N_BASIS, generator=gen, device=device, dtype=dtype) * 0.6
        rho = torch.zeros(B, R, device=device, dtype=dtype)
        ell = torch.ones(B, device=device, dtype=dtype) * 1.5
    else:
        s_grp = _half_normal((B, 3), 1.0, gen, device, dtype).clamp_min(1e-3)
        if regime == "sparse":
            s_grp = s_grp * torch.tensor([1.0, 0.05, 0.15], device=device, dtype=dtype)
        sj = s_grp[:, GROUP_T.to(device)]                              # (B, 9)
        mu = torch.randn(B, N_BASIS, generator=gen, device=device, dtype=dtype) * sj
        tau = _half_normal((B, N_BASIS), 1.0, gen, device, dtype) * sj
        eps = torch.randn(B, R, N_BASIS, generator=gen, device=device, dtype=dtype)
        theta = mu[:, None, :] + tau[:, None, :] * eps
        ell = torch.exp(torch.randn(B, generator=gen, device=device, dtype=dtype) * 0.5
                        + np.log(1.5))
        rscale = 0.25 if regime != "gp" else 1.2
        rho = _half_normal((B, R), rscale, gen, device, dtype)
        if regime == "gp":
            theta = theta * 0.3

    eta = a[:, None, :] + amp[:, None, :] * torch.einsum("bnj,brj->bnr", G, theta)

    if float(rho.abs().max()) > 0:
        K = matern52(Zb, Zb, ell) + 1e-5 * torch.eye(n, device=device, dtype=dtype)[None]
        L = torch.linalg.cholesky(K)
        w = torch.randn(B, n, R, generator=gen, device=device, dtype=dtype)
        f = L @ w                                                      # (B, n, R)
        eta = eta + amp[:, None, :] * rho[:, None, :] * f

    sfrac = _half_normal((B, R), 0.6, gen, device, dtype).clamp(0.02, 2.0)
    sigma = amp * sfrac
    y = eta + sigma[:, None, :] * torch.randn(B, n, R, generator=gen, device=device,
                                              dtype=dtype)
    if censor and regime != "ols":
        y = y.clamp(LOWER, UPPER)
    return dict(y=y, eta=eta, theta=theta, a=a, amp=amp, sigma=sigma, rho=rho, ell=ell)


def true_optimum(a, amp, theta, rho=None, ell=None, Z_grid=None, device="cpu"):
    """Argmax of the clean latent surface over a grid, per batch element and task."""
    G = basis(Z_grid)                                                  # (m, 9)
    eta = a[:, None, :] + amp[:, None, :] * torch.einsum("mj,brj->bmr", G, theta)
    idx = eta.argmax(dim=1)                                            # (B, R)
    return idx, eta
