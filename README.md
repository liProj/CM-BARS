# CM-BARS

**Zero-aware multitask probabilistic response surfaces for phenolic adsorption on cellulose nanofibers.**

[中文使用说明](README_zh.md) · [Model](docs/MODEL.md) · [Reproduction guide](docs/REPRODUCIBILITY.md)

CM-BARS combines a hurdle-Beta observation distribution, signed low-rank coefficient sharing, compound-specific deviations, and blockwise shrinkage. This repository contains the paper model, conventional baselines, component ablations, PFN-RSM pretraining and inference, public example data, archived numerical results, and figure-generation code.

The main model is `code/cmbars2.py`, configured with `use_gp=False`. `code/cmbars.py` is the **Tobit-MT-Bayes comparator**, not the paper's main hurdle-Beta model.

## Install

Use Python 3.13 and a virtual environment. CM-BARS runs on CPU; a GPU is useful for full PFN pretraining.

```bash
git clone https://github.com/liProj/CM-BARS.git
cd CM-BARS
python -m venv .venv
```

Activate on Linux / DGX:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the main model and ordinary baselines:

```bash
python -m pip install -r requirements.txt
python run.py doctor
```

For PFN-RSM, LightGBM, and XGBoost, install the additional dependencies:

```bash
python -m pip install -r requirements-full.txt
```

The validated environment is listed in `docs/validated_environment.json`. A DGX PyTorch environment can retain its hardware-compatible PyTorch installation. The main CM-BARS sampler is configured for CPU.

## Start with a complete example

```bash
python run.py check
python run.py reference
python run.py demo
python run.py predict --posterior results/demo/posterior.npz --query examples/query.csv
```

`check` validates the 15-run example against the source design and published coefficients. `reference` prints archived paper scores without training. `demo` performs a deliberately short 30-warm-up / 40-sample, one-chain fit and saves a portable posterior, predictions, and diagnostics in `results/demo/`. It is an execution check, not an inferential run or a reproduction of paper scores.

Prediction output contains process factors, compound, mean adsorption, predictive standard deviation, the 5th and 95th percentiles, and an empirical zero probability. Adsorption and interval values are in **percentage points**; `p_zero` is in `[0,1]`.

## Fit and predict

```bash
python run.py fit --data data/bbd_design_responses.csv
python run.py predict --posterior results/fit/posterior.npz --query examples/query.csv --output results/new_conditions.csv
```

The default fit uses 1000 warm-up iterations, 1000 retained iterations, two sequential chains, and seed 0. For the full-data sampling budget described in the paper:

```bash
python run.py fit --warmup 3000 --samples 3000 --chains 4 --output results/full_fit
```

Inspect `diagnostics.csv` before drawing conclusions from a fit. The saved `.npz` posterior can be loaded for later predictions without rerunning MCMC.

Input CSV columns are:

| Column | Meaning | Supported range |
|---|---|---|
| `pH` | Solution pH | 4–8 |
| `additive_mM` | Phenolic concentration | 5–15 mM |
| `cnf_pct` | CNF concentration, percent w/v | 0.10–0.40 |
| `tannic_acid` | TA adsorption percentage | 0 ≤ y < 100 |
| `p_coumaric_acid` | PC adsorption percentage | 0 ≤ y < 100 |
| `acetosyringone` | AS adsorption percentage | 0 ≤ y < 100 |

Queries require only the first three columns. Use `0.25` for 0.25% CNF. Training data require all three responses, finite values, and at least three rows. The model is configured for these three compounds and fixed design ranges; the CLI rejects extrapolation and observations at exactly 100%, which would require an upper-bound mass model.

## Run experiments

```bash
python run.py cv --methods CM-BARS "OLS-Quad (published)" ExtraTrees --protocols LOCO --workers 1
python run.py cv --methods CM-BARS "OLS-Quad (published)" --protocols RKF --repeats 20 --workers 1
python run.py analyze
```

LOCO holds out all replicates and responses at each of 13 conditions. LOO is available with `--protocols LOO`; it retains other center-point replicates when one center run is held out. RKF uses repeated grouped five-fold splits. The default MCMC budget in validation is 1000 + 1000 iterations with two chains; `--fast` reduces it to 300 + 300 for development.

Each completed CLI run saves a configuration and metric snapshot under `results/runs/`. Cache namespaces depend on configuration, code, data, and the selected PFN checkpoint, preventing fast runs from being reused as paper-budget results. Top-level `results/cv_*.csv` files hold the latest run of each metric type. `reference_results/` remains unchanged.

The full method list and exact ablation commands are in [the reproduction guide](docs/REPRODUCIBILITY.md).

## PFN-RSM

A 50,000-step checkpoint is included under `checkpoints/` (approximately 3.4 MB). Use it directly:

```bash
python run.py predict --model pfn --query examples/query.csv --output results/pfn_predictions.csv
python run.py cv --methods "PFN-RSM (prior-fitted)" --protocols LOCO
```

To pretrain a new network:

```bash
python run.py pfn-train --steps 50000 --batch 256 --device cuda
```

New weights are saved to `results/pfn_rsm.pt`, leaving the bundled checkpoint intact. When present, these newly trained weights are used by default; `PFN_CHECKPOINT` can select another checkpoint. PFN-RSM uses its own synthetic Gaussian/censoring prior and is evaluated separately from CM-BARS's hurdle-Beta model.

## Figures and archived results

```bash
python run.py figures
```

This regenerates six companion plots from saved numerical inputs under `paper_figures/figures/`. Of these, the mean-RMSE, interval, and CM-BARS surface plots appear in the final paper. `code/fig_data.py` and `code/fig_results.py` retain the original experimental plotting routines; their expected inputs are documented in the reproduction guide.

Archived LOCO mean RMSE values, in percentage points, are:

| Model | Mean RMSE |
|---|---:|
| OLS quadratic | 10.431 |
| CM-BARS | 6.417 |
| PFN-RSM | 6.332 |
| ExtraTrees | 5.646 |

These are saved paper results, not scores from the short demo. CM-BARS improves on OLS while offering bounded distributions and explicit zero probabilities; ExtraTrees has lower average point error. Historical CM-BARS+ cached-pooling scores are supplementary records with indirect information leakage and are not unbiased outer-fold generalization estimates.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover condition isolation, input validation, predictive support, posterior round trips, scoring, and configuration-sensitive caches. See `docs/VALIDATION.md` for the executed checks.

## Data attribution and citation

The example data were transcribed from Table 1 and Supplementary Table S1 of García-Fuentevilla et al., *Optimization of Bioactive Compounds Incorporation into Nanocellulose-Based Films for Food Packaging Applications*, Macromol 6(2), 22 (2026), [doi:10.3390/macromol6020022](https://doi.org/10.3390/macromol6020022). Published quadratic coefficients and reference optima are retained for validation. Please cite the source study when using these measurements.

This repository accompanies the manuscript *Zero-aware multitask probabilistic response surfaces for phenolic adsorption on cellulose nanofibers*. No publication DOI is assigned to this manuscript in the supplied materials.
