"""Validated input and portable posterior I/O for the paper's CM-BARS model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from methods import RESPONSES, FACTOR_RANGE, to_coded

FACTORS = list(FACTOR_RANGE)

def read_data(path, responses=True):
    frame = pd.read_csv(path)
    required = FACTORS + (RESPONSES if responses else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    values = frame[required].to_numpy(dtype=float)
    if not len(frame) or not np.isfinite(values).all():
        raise ValueError("CSV must contain nonempty, finite numeric observations.")
    for key, (lo, hi) in FACTOR_RANGE.items():
        if not frame[key].between(lo, hi).all():
            raise ValueError(f"{key} must remain within [{lo}, {hi}].")
    if responses and ((frame[RESPONSES] < 0).any().any() or (frame[RESPONSES] >= 100).any().any()):
        raise ValueError("This hurdle-Beta implementation supports 0 <= response < 100.")
    if responses and len(frame) < 3:
        raise ValueError("At least three training rows are required.")
    return frame

def factors(frame):
    return to_coded(frame.pH, frame.additive_mM, frame.cnf_pct)

def model(**kwargs):
    from cmbars2 import CMBARS2
    config = dict(use_gp=False, warmup=1000, samples=1000, chains=2, seed=0, name="CM-BARS")
    config.update(kwargs)
    return CMBARS2(**config)

def save_posterior(fitted, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(format_version=1, cfg=fitted.cfg, seed=fitted.seed,
                responses=RESPONSES, factor_ranges=FACTOR_RANGE)
    np.savez_compressed(destination, Ztr=fitted.Ztr_, metadata=np.array(json.dumps(meta)),
                        **{f"posterior_{k}": v for k, v in fitted.post_.items()})

def load_posterior(path):
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata"]))
        if meta["format_version"] != 1 or meta["responses"] != RESPONSES:
            raise ValueError("Unsupported posterior format or response order.")
        fitted = model(seed=meta["seed"], **meta["cfg"])
        fitted.Ztr_ = saved["Ztr"].copy()
        fitted.post_ = {k.removeprefix("posterior_"): saved[k].copy()
                       for k in saved.files if k.startswith("posterior_")}
    return fitted

def predictions(fitted, queries, n_draw=2000):
    z = factors(queries)
    out = fitted.predict(z, n_draw=n_draw)
    rows = []
    for response in RESPONSES:
        samples = out[response]["samples"]
        for i, (_, row) in enumerate(queries.iterrows()):
            rows.append({**{f: float(row[f]) for f in FACTORS}, "query": i,
                         "response": response, "mean": float(out[response]["mean"][i]),
                         "sd": float(out[response]["sd"][i]),
                         "q05": float(np.quantile(samples[:, i], .05)),
                         "q95": float(np.quantile(samples[:, i], .95)),
                         "p_zero": float((samples[:, i] == 0).mean())})
    return pd.DataFrame(rows)
