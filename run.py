"""Command-line entry point. Run `python run.py --help` from any directory."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, shutil, subprocess, sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

def script(name, *arguments, env=None):
    subprocess.run([sys.executable, str(ROOT / "code" / name), *arguments],
                   cwd=ROOT, env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description="CM-BARS paper experiments and predictions")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Validate the paper dataset and source coefficients")
    sub.add_parser("doctor", help="Report available dependencies")
    sub.add_parser("reference", help="Print archived paper metrics without retraining")
    sub.add_parser("figures", help="Regenerate translated paper plots from archived results")
    for command in ["demo", "fit"]:
        a = sub.add_parser(command, help="Short smoke fit" if command == "demo" else "Fit main hurdle-Beta model")
        a.add_argument("--data", type=Path, default=ROOT / "data/bbd_design_responses.csv")
        a.add_argument("--output", type=Path, default=ROOT / "results" / command)
        a.add_argument("--warmup", type=int, default=30 if command == "demo" else 1000)
        a.add_argument("--samples", type=int, default=40 if command == "demo" else 1000)
        a.add_argument("--chains", type=int, default=1 if command == "demo" else 2)
        a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("predict", help="Predict with a saved CM-BARS posterior or bundled PFN")
    a.add_argument("--model", choices=["cmbars", "pfn"], default="cmbars")
    a.add_argument("--posterior", type=Path, default=ROOT / "results/fit/posterior.npz")
    a.add_argument("--data", type=Path, default=ROOT / "data/bbd_design_responses.csv")
    a.add_argument("--query", type=Path, default=ROOT / "examples/query.csv")
    a.add_argument("--output", type=Path, default=ROOT / "results/predictions.csv")
    a.add_argument("--draws", type=int, default=2000)
    a = sub.add_parser("cv", help="Run grouped validation for named models")
    a.add_argument("--methods", nargs="+", default=["CM-BARS", "OLS-Quad (published)", "ExtraTrees"])
    a.add_argument("--protocols", default="LOCO")
    a.add_argument("--repeats", type=int, default=20)
    a.add_argument("--workers", type=int, default=1)
    a.add_argument("--fast", action="store_true", help="300+300 MCMC iterations; not paper scores")
    a = sub.add_parser("pfn-train", help="Train PFN-RSM on synthetic tasks")
    a.add_argument("--steps", type=int, default=50000)
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--device", choices=["cpu", "cuda"], default=None)
    a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("analyze", help="Summarize newly computed validation results")
    a.add_argument("--method", default="CM-BARS")
    args = parser.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    if args.command == "doctor":
        print("Python", sys.version.split()[0])
        for name in ["numpy", "pandas", "scipy", "scikit-learn", "jax", "jaxlib", "numpyro", "torch", "lightgbm", "xgboost"]:
            try: print(name, importlib.metadata.version(name))
            except importlib.metadata.PackageNotFoundError: print(name, "not installed (see requirements files)")
    elif args.command == "check":
        script("data_check.py")
        import pandas as pd
        table = pd.read_csv(ROOT / "results/data_check.csv")
        if not (table.status == "PASS").all():
            raise SystemExit("Dataset checks failed.")
    elif args.command == "reference":
        import pandas as pd
        d = pd.read_csv(ROOT / "reference_results/cv_pooled.csv")
        d = d[(d.protocol == "LOCO") & d.method.isin(["CM-BARS", "OLS-Quad (published)", "ExtraTrees", "PFN-RSM (prior-fitted)"])]
        print("Archived paper results (not a new fit)")
        print(d.pivot(index="method", columns="response", values="RMSE").round(3))
        print("\nMean RMSE:\n", d.groupby("method").RMSE.mean().round(3))
    elif args.command == "figures":
        subprocess.run([sys.executable, str(ROOT / "paper_figures/build_figures.py")], check=True, cwd=ROOT)
        print(ROOT / "paper_figures/figures")
    elif args.command in ["demo", "fit"]:
        from api import read_data, factors, model, save_posterior, predictions
        from methods import RESPONSES
        d = read_data(args.data)
        if min(args.warmup, args.samples, args.chains) < 1:
            raise ValueError("Iteration counts and chain count must be positive.")
        args.output.mkdir(parents=True, exist_ok=True)
        config = dict(warmup=args.warmup, samples=args.samples, chains=args.chains, seed=args.seed)
        print("Fitting CM-BARS", config, flush=True)
        fitted = model(**config).fit(factors(d), d[RESPONSES].to_numpy())
        save_posterior(fitted, args.output / "posterior.npz")
        q = read_data(ROOT / "examples/query.csv", responses=False)
        predictions(fitted, q).to_csv(args.output / "predictions.csv", index=False)
        import pandas as pd
        pd.DataFrame(fitted.diagnostics()).to_csv(args.output / "diagnostics.csv", index=False)
        (args.output / "run.json").write_text(json.dumps({**config, "smoke_only": args.command == "demo", "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest()}, indent=2))
        print("Saved posterior, predictions, and diagnostics to", args.output)
        if args.command == "demo": print("Smoke test only: these short chains are not inferential results.")
    elif args.command == "predict":
        from api import read_data, factors, load_posterior, predictions
        if args.draws < 2: raise ValueError("Use at least two predictive draws.")
        q = read_data(args.query, responses=False)
        if args.model == "cmbars": fitted = load_posterior(args.posterior)
        else:
            from pfn import PFNRSM
            from methods import RESPONSES
            d = read_data(args.data)
            fitted = PFNRSM(device="cpu").fit(factors(d), d[RESPONSES].to_numpy())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions(fitted, q, args.draws).to_csv(args.output, index=False)
        print("Predictions saved to", args.output)
    elif args.command == "cv":
        if "CM-BARS+ (stacked)" in args.methods:
            raise ValueError("Cached pooling is a historical extension, not a core validation method. See docs/REPRODUCIBILITY.md.")
        if args.workers < 1 or args.repeats < 1:
            raise ValueError("Workers and repeats must be positive.")
        protocols = args.protocols.split(",")
        if not set(protocols) <= {"LOCO", "LOO", "RKF"}:
            raise ValueError("Protocols must be LOCO, LOO, or RKF.")
        config = {"methods": args.methods, "protocols": protocols, "repeats": args.repeats, "fast": args.fast}
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        files = sorted((ROOT / "code").glob("*.py")) + sorted((ROOT / "data").glob("*.csv"))
        if "PFN-RSM (prior-fitted)" in args.methods:
            ckpt = Path(os.environ.get("PFN_CHECKPOINT", str(ROOT / "results/pfn_rsm.pt")))
            files.append(ckpt if ckpt.exists() else ROOT / "checkpoints/pfn_rsm.pt")
        for path in files: digest.update(path.read_bytes())
        namespace = digest.hexdigest()[:20]
        env = dict(os.environ, CMBARS_CACHE_NAMESPACE=namespace)
        command = ["--only", ",".join(args.methods), "--protocols", args.protocols, "--n-rep", str(args.repeats), "--workers", str(args.workers)]
        if args.fast: command.append("--fast")
        script("experiments.py", *command, env=env)
        target = ROOT / "results/runs" / namespace
        target.mkdir(parents=True, exist_ok=True)
        for name in (["cv_pooled.csv"] if any(x in protocols for x in ["LOCO", "LOO"]) else []) + (["cv_perfold.csv"] if "RKF" in protocols else []):
            source = ROOT / "results" / name
            if source.exists(): shutil.copy2(source, target / name)
        (target / "config.json").write_text(json.dumps(config, indent=2))
        print("Saved run snapshot to", target)
    elif args.command == "pfn-train":
        if args.steps < 20 or args.batch < 1: raise ValueError("Use >=20 steps and a positive batch size.")
        from pfn import train
        train(steps=args.steps, B=args.batch, device=args.device, seed=args.seed)
    elif args.command == "analyze":
        script("analysis.py", env=dict(os.environ, PROPOSED=args.method))

if __name__ == "__main__":
    main()
