"""Hard data check for Macromol #22 (10.3390/macromol6020022).

Verifies, number for number, that the dataset transcribed from Table 1 (design) and
supplementary Table S1 (responses) is the dataset the published models were fitted to:
  * Box-Behnken structure (15 runs, 3 factors, 3 levels, 12 edge + 3 centre)
  * coded <-> uncoded level consistency
  * sample size, missing values, duplicate rows, response ranges
  * the three maxima quoted in the abstract/Section 3.1 (60.4 / 28.1 / 12.6)
  * all 30 coefficients of Equations (6)-(8) refitted by OLS in uncoded units
  * the three R-squared values (91.10 / 85.23 / 93.93 %)
  * the three optima of Table 2, re-derived by maximising each published surface

Writes results/data_check.csv and prints a pass/fail summary.
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
JD = os.path.dirname(HERE)
RES = f"{JD}/results"
os.makedirs(RES, exist_ok=True)

RESPONSES = ["tannic_acid", "p_coumaric_acid", "acetosyringone"]
TERMS = ["const", "pH", "conc", "cnf", "pH2", "conc2", "cnf2", "pH:conc", "pH:cnf", "conc:cnf"]

d = pd.read_csv(f"{JD}/data/bbd_design_responses.csv")
pub = pd.read_csv(f"{JD}/data/published_models.csv")
pubr = pd.read_csv(f"{JD}/data/published_r2_optima.csv")

checks = []
def chk(item, published, found, ok, note=""):
    checks.append(dict(item=item, published=published, reproduced=found,
                       status="PASS" if ok else "FAIL", note=note))

# ---------- 1. design structure ----------
chk("number of runs", 15, len(d), len(d) == 15)
chk("number of factors", 3, 3, True)
chk("levels per factor (coded)", "{-1,0,1}",
    str(sorted(set(d.x1_coded) | set(d.x2_coded) | set(d.x3_coded))),
    sorted(set(d.x1_coded) | set(d.x2_coded) | set(d.x3_coded)) == [-1, 0, 1])

# a 3-factor Box-Behnken is the 12 edge midpoints + centre replicates
coded = d[["x1_coded", "x2_coded", "x3_coded"]].values
n_edge = sum(1 for r in coded if sorted(np.abs(r)) == [0, 1, 1])
n_centre = sum(1 for r in coded if not np.any(r))
chk("edge-midpoint runs (|x|=1 in exactly 2 factors)", 12, n_edge, n_edge == 12)
chk("centre-point replicates", 3, n_centre, n_centre == 3)
chk("other runs (none allowed in a BBD)", 0, 15 - n_edge - n_centre,
    15 - n_edge - n_centre == 0)
# rotatability: every pair of factors must see a full 2x2 at the third factor's centre
pairs_ok = True
for i, j in itertools.combinations(range(3), 2):
    k = ({0, 1, 2} - {i, j}).pop()
    blk = coded[coded[:, k] == 0]
    blk = blk[np.abs(blk[:, i]) == 1]
    got = {(int(r[i]), int(r[j])) for r in blk}
    pairs_ok &= got == {(-1, -1), (-1, 1), (1, -1), (1, 1)}
chk("balanced 2x2 block for each factor pair", "complete", "complete" if pairs_ok else "incomplete",
    pairs_ok)

# ---------- 2. coded <-> uncoded consistency ----------
LEV = {"pH": (4, 6, 8), "additive_mM": (5, 10, 15), "cnf_pct": (0.10, 0.25, 0.40)}
for cod, col in [("x1_coded", "pH"), ("x2_coded", "additive_mM"), ("x3_coded", "cnf_pct")]:
    lo, mid, hi = LEV[col]
    want = d[cod].map({-1: lo, 0: mid, 1: hi})
    ok = np.allclose(want.values, d[col].values)
    chk(f"coded level -> experimental value ({col})", f"{lo}/{mid}/{hi}",
        "consistent" if ok else "mismatch", ok)

# ---------- 3. data integrity ----------
chk("missing values in design + responses", 0, int(d.isna().sum().sum()), d.isna().sum().sum() == 0)
chk("duplicate design rows (3 intended centre replicates)", 3,
    int(d.duplicated(subset=["pH", "additive_mM", "cnf_pct"], keep=False).sum()) - 0,
    int(d.duplicated(subset=["pH", "additive_mM", "cnf_pct"], keep=False).sum()) == 3,
    "the only repeats are the declared centre points")
chk("duplicate full rows (design + all 3 responses)", 0, int(d.duplicated().sum()),
    d.duplicated().sum() == 0)
chk("response values outside [0, 100] %", 0,
    int(((d[RESPONSES] < 0) | (d[RESPONSES] > 100)).sum().sum()),
    ((d[RESPONSES] < 0) | (d[RESPONSES] > 100)).sum().sum() == 0)
chk("response cells transcribed (15 runs x 3 phenolics)", 45, d[RESPONSES].size,
    d[RESPONSES].size == 45)

# ---------- 4. quoted maxima and zero counts ----------
for r, quoted in [("tannic_acid", 60.4), ("p_coumaric_acid", 28.1), ("acetosyringone", 12.6)]:
    got = d[r].max()
    chk(f"maximum quoted in Section 3.1 ({r})", quoted, round(got, 2),
        abs(round(got, 1) - quoted) < 0.051)
chk("tannic acid adsorbed in all 15 runs (text claim)", "all 15 > 0", int((d.tannic_acid > 0).sum()),
    (d.tannic_acid > 0).all())
chk("runs with zero adsorption (p-coumaric)", ">0 per text", int((d.p_coumaric_acid == 0).sum()),
    (d.p_coumaric_acid == 0).sum() > 0, "recorded exact zeros; mechanism not inferred")
chk("runs with zero adsorption (acetosyringone)", ">0 per text",
    int((d.acetosyringone == 0).sum()), (d.acetosyringone == 0).sum() > 0)

# ---------- 5. refit Equations (6)-(8) in uncoded units ----------
def design_matrix(df):
    p, c, f = df.pH.values, df.additive_mM.values, df.cnf_pct.values
    return np.column_stack([np.ones_like(p), p, c, f, p * p, c * c, f * f, p * c, p * f, c * f])

X = design_matrix(d)
refit = {}
for r in RESPONSES:
    y = d[r].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    refit[r] = (beta, r2)
    pubb = pub[pub.response == r].set_index("term").coef
    for k, t in enumerate(TERMS):
        pv, fv = float(pubb[t]), beta[k]
        # the paper prints each coefficient rounded; accept agreement to the printed precision
        dec = len(str(pv).split(".")[1]) if "." in str(pv) else 0
        ok = abs(round(fv, dec) - pv) <= 10 ** (-dec) * 1.01
        chk(f"Eq. coefficient {t} [{r}]", pv, round(fv, max(dec, 4)), ok)
    pr2 = float(pubr[pubr.response == r].r2_pct.iloc[0])
    chk(f"R-squared [{r}]", f"{pr2:.2f}%", f"{100 * r2:.2f}%", abs(100 * r2 - pr2) < 0.05)

# ---------- 6. re-derive the Table 2 optima from the published surfaces ----------
BOUNDS = [(4, 8), (5, 15), (0.10, 0.40)]
for r in RESPONSES:
    b = pub[pub.response == r].set_index("term").coef[TERMS].values.astype(float)
    def negf(z, b=b):
        p, c, f = z
        v = np.array([1, p, c, f, p * p, c * c, f * f, p * c, p * f, c * f])
        return -float(b @ v)
    best, bz = np.inf, None
    for p0 in itertools.product([4, 6, 8], [5, 10, 15], [0.1, 0.25, 0.4]):
        res = minimize(negf, p0, bounds=BOUNDS, method="L-BFGS-B")
        if res.fun < best:
            best, bz = res.fun, res.x
    row = pubr[pubr.response == r].iloc[0]
    got = (round(bz[0], 1), round(bz[1], 1), round(bz[2], 2))
    want = (row.opt_pH, row.opt_conc_mM, row.opt_cnf_pct)
    ok = (abs(got[0] - want[0]) <= 0.15 and abs(got[1] - want[1]) <= 0.15
          and abs(got[2] - want[2]) <= 0.015)
    chk(f"Table 2 optimum [{r}]", f"pH {want[0]}, {want[1]} mM, {want[2]}%",
        f"pH {got[0]}, {got[1]} mM, {got[2]}%", ok,
        "maximum of the published quadratic over the design box")

out = pd.DataFrame(checks)
out.to_csv(f"{RES}/data_check.csv", index=False)
n_pass = (out.status == "PASS").sum()
print(out.to_string(index=False, max_colwidth=46))
print(f"\n{n_pass}/{len(out)} checks PASS")
if n_pass < len(out):
    print("\nFAILURES:")
    print(out[out.status == "FAIL"].to_string(index=False, max_colwidth=60))
