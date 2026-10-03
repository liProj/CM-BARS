"""PFN-RSM synthetic-prior transformer for small response-surface datasets.

The generator uses Gaussian noise and clipping at zero, whereas CM-BARS uses
a hurdle-Beta likelihood. PFN-RSM is an amortized predictor under its own task
prior, not an approximation to the identical CM-BARS posterior. Context-only
normalization and masked query attention prevent access to query responses.
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JD = os.path.dirname(HERE)
from methods import LOWER, RESPONSES, UPPER, Method, quad_basis   # noqa: E402
from prior import BBD, basis, sample_surfaces                     # noqa: E402

CKPT = os.environ.get("PFN_CHECKPOINT") or (f"{JD}/results/pfn_rsm.pt" if os.path.exists(f"{JD}/results/pfn_rsm.pt") else f"{JD}/checkpoints/pfn_rsm.pt")
N_MIX = 5
N_TASK = 3
FEAT = 3 + 9 + 1 + 1 + 1 + N_TASK + 1      # z, basis, y~, is_cens, bound~, task, is_query


def make_features(Z, Yn, is_cens, bound, is_query):
    """Z (B,n,3); Yn/is_cens (B,n,R); bound (B,R) -> tokens (B, n*R, FEAT)."""
    B, n, _ = Z.shape
    R = Yn.shape[2]
    G = basis(Z)                                                  # (B,n,9)
    zz = torch.cat([Z, G], dim=-1)[:, :, None, :].expand(B, n, R, 12)
    task = torch.eye(R, device=Z.device, dtype=Z.dtype)[None, None].expand(B, n, R, R)
    y = Yn[..., None]
    c = is_cens[..., None].to(Z.dtype)
    bd = bound[:, None, :, None].expand(B, n, R, 1)
    q = torch.full((B, n, R, 1), float(is_query), device=Z.device, dtype=Z.dtype)
    tok = torch.cat([zz, y, c, bd, task, q], dim=-1)
    return tok.reshape(B, n * R, FEAT)


class PFNRSMNet(nn.Module):
    def __init__(self, d_model=128, nhead=8, nlayer=6, dff=256, n_mix=N_MIX):
        super().__init__()
        self.inp = nn.Linear(FEAT, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dff, dropout=0.0,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, dff),
                                  nn.GELU(), nn.Linear(dff, 3 * n_mix))
        self.n_mix = n_mix

    def forward(self, ctx_tok, qry_tok):
        L_c, L_q = ctx_tok.shape[1], qry_tok.shape[1]
        x = torch.cat([ctx_tok, qry_tok], dim=1)
        L = L_c + L_q
        # context is visible to everyone; a query sees the context and itself only
        allowed = torch.zeros(L, L, dtype=torch.bool, device=x.device)
        allowed[:, :L_c] = True
        allowed[torch.arange(L), torch.arange(L)] = True
        mask = torch.zeros(L, L, device=x.device, dtype=x.dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        h = self.enc(self.inp(x), mask=mask)[:, L_c:]
        o = self.head(h)
        logit, mu, logs = o.split(self.n_mix, dim=-1)
        return logit, mu, logs.clamp(-6.0, 3.0)


def tobit_mixture_logprob(y, is_cens, bound, logit, mu, logs):
    """log p(y) under a mixture of normals censored from below at `bound`."""
    logw = F.log_softmax(logit, dim=-1)
    sd = logs.exp()
    z = (y[..., None] - mu) / sd
    lpdf = -0.5 * z ** 2 - logs - 0.5 * math.log(2 * math.pi)
    zb = (bound[..., None] - mu) / sd
    lcdf = torch.special.log_ndtr(zb)
    lp = torch.where(is_cens[..., None], lcdf, lpdf)
    return torch.logsumexp(logw + lp, dim=-1)


# --------------------------------------------------------------------- training
def _batch(B, h, gen, device, regimes=("cmbars", "cmbars", "cmbars", "gp", "sparse", "ols"),
           n_extra=10, dtype=torch.float32):
    """One training batch: context = BBD minus `h` runs, query = those runs + random points."""
    reg = regimes[int(torch.randint(len(regimes), (1,), generator=gen, device=device))]
    n_b = BBD.shape[0]
    extra = (torch.rand(n_extra, 3, generator=gen, device=device, dtype=dtype) * 2 - 1)
    Zall = torch.cat([torch.as_tensor(BBD, device=device, dtype=dtype), extra], dim=0)
    s = sample_surfaces(B, Zall.cpu().numpy(), gen, device, n_task=N_TASK, regime=reg,
                        dtype=dtype, censor=True)
    y = s["y"]                                                     # (B, n_all, R)
    # choose h held-out BBD runs per batch element (shared index set for easy batching)
    perm = torch.randperm(n_b, generator=gen, device=device)
    qidx = perm[:h]
    cidx = perm[h:]
    qidx_all = torch.cat([qidx, torch.arange(n_b, n_b + n_extra, device=device)])

    yc, yq = y[:, cidx, :], y[:, qidx_all, :]
    m = yc.mean(dim=1)
    sd = yc.std(dim=1, unbiased=False).clamp_min(0.25)
    yc_n = (yc - m[:, None, :]) / sd[:, None, :]
    yq_n = (yq - m[:, None, :]) / sd[:, None, :]
    bound = (LOWER - m) / sd
    cc = yc <= LOWER + 1e-9
    cq = yq <= LOWER + 1e-9
    if reg == "ols":                        # the uncensored regime has no floor
        cc = torch.zeros_like(cc)
        cq = torch.zeros_like(cq)
        bound = torch.full_like(bound, -1e3)
    ctx = make_features(Zall[cidx][None].expand(B, len(cidx), 3), yc_n, cc, bound, 0.0)
    qry = make_features(Zall[qidx_all][None].expand(B, len(qidx_all), 3),
                        torch.zeros_like(yq_n), torch.zeros_like(cq), bound, 1.0)
    return ctx, qry, yq_n.reshape(B, -1), cq.reshape(B, -1), bound


def train(steps=50000, B=256, lr=3e-4, device=None, seed=0, log_every=500):
    os.makedirs(f"{JD}/results", exist_ok=True)
    output_ckpt = f"{JD}/results/pfn_rsm.pt"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    net = PFNRSMNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.05)
    hist = []
    t0 = time.time()
    run = []
    for step in range(1, steps + 1):
        h = int(torch.randint(1, 5, (1,), generator=gen, device=device))
        ctx, qry, yq, cq, bound = _batch(B, h, gen, device)
        bd = bound[:, None, :].expand(-1, qry.shape[1] // N_TASK, -1).reshape(B, -1)
        logit, mu, logs = net(ctx, qry)
        lp = tobit_mixture_logprob(yq, cq, bd, logit, mu, logs)
        loss = -lp.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(float(loss))
        if step % log_every == 0 or step == steps:
            m = float(np.mean(run[-log_every:]))
            hist.append(dict(step=step, nll=m, seconds=time.time() - t0))
            print(f"  step {step:6d}  NLL {m:7.4f}  {time.time() - t0:6.0f}s", flush=True)
    torch.save(dict(state=net.state_dict(), hist=hist, steps=steps, B=B, seed=seed), output_ckpt)
    import pandas as pd
    pd.DataFrame(hist).to_csv(f"{JD}/results/pfn_train_history.csv", index=False)
    print("saved", output_ckpt)
    return net


# --------------------------------------------------------------------- inference
class PFNRSM(Method):
    """Pre-trained prior-fitted network applied to the real design in one forward pass."""
    name = "PFN-RSM (prior-fitted)"
    multitask = True
    probabilistic = True

    _net = None
    _dev = None
    _checkpoint_key = None

    def __init__(self, ckpt=CKPT, device=None, seed=0, name=None):
        self.ckpt, self.seed = ckpt, seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if name:
            self.name = name

    @classmethod
    def _load(cls, ckpt, device):
        key = (os.path.abspath(ckpt), os.stat(ckpt).st_mtime_ns)
        if cls._net is None or cls._dev != device or cls._checkpoint_key != key:
            st = torch.load(ckpt, map_location=device, weights_only=True)
            net = PFNRSMNet().to(device)
            net.load_state_dict(st["state"])
            net.eval()
            cls._net, cls._dev = net, device
            cls._checkpoint_key = key
        return cls._net

    def fit(self, Z, Y):
        self.Z_ = np.asarray(Z, float)
        self.Y_ = np.asarray(Y, float)
        self.m_ = self.Y_.mean(axis=0)
        self.s_ = np.maximum(self.Y_.std(axis=0), 0.25)
        self.net_ = self._load(self.ckpt, self.device)
        return self

    @torch.no_grad()
    def predict(self, Zte, n_draw=2000, chunk=512):
        """Predict at `Zte`.

        Query points are processed in chunks: attention is quadratic in the number of
        tokens and each query point contributes one token per task, so a dense design-box
        grid would otherwise build a 30k x 30k attention matrix.
        """
        Zq_all = np.atleast_2d(np.asarray(Zte, float))
        if len(Zq_all) > chunk:
            outs = [self.predict(Zq_all[i:i + chunk], n_draw=n_draw, chunk=chunk)
                    for i in range(0, len(Zq_all), chunk)]
            merged = {}
            for r in RESPONSES:
                sm = np.concatenate([o[r]["samples"] for o in outs], axis=1)
                merged[r] = self._pack(sm.mean(axis=0), sd=sm.std(axis=0, ddof=1),
                                       samples=sm)
            return merged
        dev = self.device
        Zc = torch.as_tensor(self.Z_, device=dev, dtype=torch.float32)[None]
        Zq = torch.as_tensor(Zq_all, device=dev, dtype=torch.float32)[None]
        m = torch.as_tensor(self.m_, device=dev, dtype=torch.float32)[None]
        s = torch.as_tensor(self.s_, device=dev, dtype=torch.float32)[None]
        Yc = torch.as_tensor(self.Y_, device=dev, dtype=torch.float32)[None]
        Yn = (Yc - m[:, None, :]) / s[:, None, :]
        cc = Yc <= LOWER + 1e-9
        bound = (LOWER - m) / s
        ctx = make_features(Zc, Yn, cc, bound, 0.0)
        nq = Zq.shape[1]
        qry = make_features(Zq, torch.zeros(1, nq, N_TASK, device=dev),
                            torch.zeros(1, nq, N_TASK, dtype=torch.bool, device=dev),
                            bound, 1.0)
        logit, mu, logs = self.net_(ctx, qry)
        w = F.softmax(logit, dim=-1)[0].cpu().numpy()              # (nq*R, K)
        mu = mu[0].cpu().numpy()
        sd = logs.exp()[0].cpu().numpy()
        rng = np.random.default_rng(self.seed + 5)
        k = np.array([rng.choice(N_MIX, size=n_draw, p=w[i] / w[i].sum())
                      for i in range(w.shape[0])])                 # (nq*R, n_draw)
        draw_n = mu[np.arange(w.shape[0])[:, None], k] + \
            sd[np.arange(w.shape[0])[:, None], k] * rng.standard_normal(k.shape)
        draw_n = draw_n.reshape(nq, N_TASK, n_draw)
        out = {}
        for j, r in enumerate(RESPONSES):
            sm = (draw_n[:, j, :].T * self.s_[j] + self.m_[j])
            sm = np.clip(sm, LOWER, UPPER)
            out[r] = self._pack(sm.mean(axis=0), sd=sm.std(axis=0, ddof=1), samples=sm)
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "smoke"])
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    if a.cmd == "train":
        train(steps=a.steps, B=a.batch)
    else:
        train(steps=200, B=64, log_every=50)
