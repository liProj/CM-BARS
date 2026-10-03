# Reproduction guide

## Three distinct workflows

1. `run.py reference` and `run.py figures` inspect or plot archived paper results. They do not fit new models.
2. `run.py demo` checks that the main model fits, saves, loads, and predicts with short chains. Its scores are not paper results.
3. `run.py cv` recomputes validation with the stated sampling budget. Hardware, numerical libraries, and Monte Carlo randomness can change the exact scores. The current validated environment is recorded separately from the original DGX experiment record.

## Data and configurations

Run `python run.py check` first. The example contains 15 runs, 13 distinct conditions, three responses, and zero counts 0/4/3. Factor coding is `(pH−6)/2`, `(additive_mM−10)/5`, and `(cnf_pct−0.25)/0.15`.

Core LOCO results:

```bash
python run.py cv --methods CM-BARS "OLS-Quad (published)" ExtraTrees --protocols LOCO
```

Full conventional baseline comparison (install `requirements-full.txt` first):

```bash
python run.py cv --methods "OLS-Quad (published)" "OLS-Quad + clip" Ridge-Quad Lasso-Quad ElasticNet-Quad PLS-Quad RandomForest ExtraTrees LightGBM XGBoost GP-Matern Tobit-Quad Tobit-MT-Bayes CM-BARS "PFN-RSM (prior-fitted)" --protocols LOCO --workers 1
```

Component ablations:

```bash
python run.py cv --methods CM-BARS "CM-BARS -zero-part" "CM-BARS -bounded link" "CM-BARS -shrinkage" "CM-BARS -coupling" "CM-BARS mean-pooled" "CM-BARS +GP" "CM-BARS -all" --protocols LOCO --workers 1
```

The historical `-bounded link` label denotes the Gaussian observation/scale variant. `mean-pooled` means coefficient mean sharing and is different from CM-BARS+ distribution pooling. Removing coupling still retains shared block scales.

Protocol comparison and repeated grouped evaluation:

```bash
python run.py cv --methods CM-BARS "OLS-Quad (published)" ExtraTrees --protocols LOCO,LOO,RKF --repeats 20 --workers 1
python run.py analyze --method CM-BARS
```

The repeated analysis applies Nadeau–Bengio variance correction and Holm adjustment across available metrics per compound. Run snapshots reside in `results/runs/<configuration-hash>/`; analyze a coherent set of outputs rather than mixing independently edited CSV files. The CLI reads the bundled study data for CV; use `fit --data` and `predict --query` for external data prediction.

## Model fitting and outputs

`python run.py fit --warmup 3000 --samples 3000 --chains 4` uses the full-data sampling budget. The API and CLI default to the core no-GP configuration. `posterior.npz` stores sampled parameters, training coordinates, response order, and configuration without Python pickle serialization. Predictions report overall sampled means; they are not conditional positive-response means.

The archived final-fit diagnostics belong to the full-data fit, not to every CV fold. Short smoke-test chains provide execution evidence only. Check convergence and effective sample sizes for any new inferential run.

## PFN

`checkpoints/pfn_rsm.pt` is the saved paper checkpoint. It was trained for 50,000 steps with batch size 256. The original recorded duration was about 3.10 hours on the DGX GB10 GPU; this is not a runtime guarantee for another machine. Pretraining consumes synthetic experiments and does not increase the 15 measured runs.

```bash
python run.py pfn-train --steps 50000 --batch 256 --device cuda
python run.py predict --model pfn
```

New training output is written under `results/`. `PFN_CHECKPOINT` can select a trusted checkpoint. Loading uses PyTorch's weights-only mode. The network cache is keyed by checkpoint path and modification time as well as device.

## Archived figure generation

`python run.py figures` regenerates the companion plots with English labels from `paper_figures/figure_data/`. Numerical inputs are copied unchanged from the manuscript project.

For original experimental figures, first copy the archived results into a separate working checkout's `results/` directory, then run:

```bash
python code/data_check.py
python code/baseline.py
python code/fig_data.py
python code/fig_results.py
```

`reference_results/` contains the pooled/per-fold metrics, full-data summaries and grids, and baseline outputs required by these plot routines. Keep it immutable. Original figure routines use the original labels and historic pooled-model framing; the final manuscript's captions and main-model comparison define the interpretation. No article-downloading or literature-screening scripts are needed for model use.

## Historical supplementary extensions

`code/stack.py`, `code/stack_oof.py`, and `code/final_fit.py` retain pooling and distillation machinery. Cached outer-fold pooling has indirect test-condition dependence, described in `MODEL.md`, and is excluded from `run.py cv`. Its saved scores in `reference_results/` are labeled historical and should not be presented as unbiased generalization estimates. A fully nested evaluation would generate inner predictions and select weights entirely within each outer training fold.

The legacy final-fit script combines multiple fitted members, derives a compact logit-quadratic teacher approximation, and computes conditional positive-response maxima. It requires the full dependencies and can be expensive. It does not replace the simpler main-model-only `run.py fit` command. Grid distillation errors measure agreement with the teacher function rather than error against new experiments.

## Provenance

The archived dataset, checkpoint, and numeric results were copied from the original DGX research directory. `source_manifest.json` records original code hashes; `release_manifest.json` records data and checkpoint hashes for this repository. Comments that overstated model or evaluation claims were aligned with the final manuscript. Model likelihood and prior code remain as implemented in the research project.
