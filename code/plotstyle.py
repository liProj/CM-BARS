"""Shared figure style for the Macromol #22 study. Colour-blind-safe and print-ready."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg")

PAL = {
    "CM-BARS+ (stacked)": "#0B6E4F",
    "CM-BARS+ distilled": "#3FA06B",
    "CM-BARS": "#117A65",
    "PFN-RSM (prior-fitted)": "#1F8A70",
    "OLS-Quad (published)": "#B03A2E",
    "OLS-Quad + clip": "#D98880",
    "Tobit-Quad": "#CA6F1E",
    "Tobit-MT-Bayes": "#E67E22",
    "Ridge-Quad": "#8E6C8A",
    "Lasso-Quad": "#A569BD",
    "ElasticNet-Quad": "#7D3C98",
    "PLS-Quad": "#5B2C6F",
    "RandomForest": "#7F8C8D",
    "ExtraTrees": "#2E86C1",
    "LightGBM": "#1B4F72",
    "XGBoost": "#2874A6",
    "GP-Matern": "#17A589",
}
# the three phenolic compounds, in the order used throughout
RESP_COL = {"tannic_acid": "#1B4F72", "p_coumaric_acid": "#CA6F1E",
            "acetosyringone": "#117A65"}
RESP_LAB = {"tannic_acid": "Tannic acid", "p_coumaric_acid": "p-Coumaric acid",
            "acetosyringone": "Acetosyringone"}
FACTOR_LAB = {0: "pH", 1: "Additive conc. (mM)", 2: "Final CNF conc. (%)"}
SEQ = "viridis"
DIV = "RdBu_r"


def setup():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 330, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    })


def col(m):
    return PAL.get(m, "#95A5A6")


def save(fig, name, outdir=None):
    outdir = outdir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(outdir, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/{name}.{ext}")
    plt.close(fig)
    print(f"  figure -> {outdir}/{name}.png")
