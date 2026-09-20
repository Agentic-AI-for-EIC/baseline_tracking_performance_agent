"""Command line: ``python -m pid <command> ...``.

Flag vocabulary deliberately mirrors ``trkperf`` (``--file``, ``--file-list``,
``--dataset-tag``, ``--min-q2-tier``, ``--max-file-failures``, ``--cache-dir``,
``--out-dir``) so the same runbook works for both halves of the project, and file
discovery stays outside the code (rucio/xrootd MCP tools build the lists - AGENTS.md).

Every command has a smoke path over ``data/dataset_small/`` so a human can test
the whole chain in a couple of minutes without touching the grid::

    python -m pid all

(``all`` uses the fixed local reference pair, so it takes no file-discovery
flags by design - the other commands do.)
"""

from __future__ import annotations

import argparse
import os
import sys

from . import config, dataset, schema


def _read_file_list(args) -> list[str]:
    files = list(getattr(args, "file", None) or [])
    if getattr(args, "file_list", None):
        with open(args.file_list) as fh:
            files.extend(line.strip() for line in fh if line.strip() and not line.startswith("#"))
    return files


def _add_io_args(p: argparse.ArgumentParser, *, needs_tag: bool = True) -> None:
    p.add_argument("--file", action="append", help="ROOT file path or root:// URL (repeatable).")
    p.add_argument("--file-list", help="Text file with one path/URL per line.")
    if needs_tag:
        p.add_argument("--dataset-tag", required=True, choices=sorted(config.CAMPAIGN_BY_DATASET_TAG),
                       help="Which project dataset this run is (sets the campaign/schema).")
        p.add_argument("--min-q2-tier", default=None,
                       help="minQ2 tier(s) used; recorded in output metadata only.")
    p.add_argument("--out-dir", default=config.OUTPUT_DIR)
    p.add_argument("--limit-files", type=int, default=None,
                   help="Process only the first N files (smoke tests).")
    p.add_argument("--max-file-failures", type=int, default=0,
                   help="Tolerate up to N failing files (flaky endpoints) instead of aborting.")
    p.add_argument("--cache-dir", default=config.FEATURE_CACHE_DIR,
                   help="Per-file feature-table cache (default: %(default)s); '' disables.")


# ---------------------------------------------------------------------------
# schema-check
# ---------------------------------------------------------------------------

def cmd_schema_check(args) -> int:
    print("== ML environment ==")
    env = schema.assert_ml_env(strict=False)
    for key, value in env.items():
        print(f"  {key:16s} {value}")
    campaign = schema.campaign_of(args.dataset) if args.dataset else None
    if args.dataset:
        print(f"\n== dataset {args.dataset} -> campaign {campaign} ==")

    print("\n== verified dead ends (never used as features) ==")
    print(schema.dead_link_report())

    if not args.file:
        print("\n(no --file given: schema/dead-link report only)")
        return 0
    from . import links

    path = args.file
    print(f"\n== link audit for {os.path.basename(path)} ==")
    audit = links.audit_links(path)
    registry = links.read_registry(path)
    print(f"  registered collections: {len(registry)}")
    usable = int(audit["usable"].sum())
    print(audit[["relation", "family", "n", "resolves_to", "usable", "status"]].to_string(index=False))
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"pid-schema-audit_{args.dataset or 'file'}.{args.fmt}")
        if args.fmt == "json":
            audit.to_json(out, orient="records", indent=2)
        else:
            audit.to_markdown(out, index=False)
        print(f"  wrote {out}")
    broken = audit[(~audit["usable"]) & (~audit["optional"].astype(bool))
                   & (~audit["status"].str.contains("expected broken"))]
    if not broken.empty:
        print(f"\nGATE FAIL: {len(broken)} relation(s) unusable but NOT expected-broken:")
        print(broken[["relation", "status"]].to_string(index=False))
        print("Adjust pid.schema/pid.features or drop the affected feature family.")
        return 1
    print(f"\nGATE PASS: {usable} audited relations usable; the rest are the "
          f"documented dead ends in pid.schema.DEAD_LINKS.")
    return 0


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def cmd_features(args) -> int:
    from . import features as pid_features

    files = _read_file_list(args)
    if not files:
        raise SystemExit("features: no input files (use --file and/or --file-list)")
    legs = tuple(args.legs.split(",")) if args.legs else ("backward", "forward")
    df = pid_features.build_features(
        files, dataset_tag=args.dataset_tag, legs=legs, max_failures=args.max_file_failures,
        cache_dir=args.cache_dir or None, limit_files=args.limit_files,
        enable_ionisation=args.enable_ionisation)
    if df.empty:
        raise SystemExit("features: no rows produced (all files failed?)")
    out = args.out or os.path.join(args.out_dir, f"pid-features_{args.dataset_tag}.pkl")
    dataset.save_table(df, out)
    skipped = df.attrs.get("skipped_files", [])
    print(f"[features] wrote {out}: {len(df)} rows x {len(df.columns)} cols from "
          f"{df.attrs.get('n_files', len(files))} files ({len(skipped)} skipped)")

    # M1 gate evidence, printed every run so a bad table cannot be missed.
    table = dataset.summary(df)
    all_nan = dataset.empty_numeric_columns(table)
    print("\n[features] feature summary (family / NaN fraction):")
    print(table.groupby("family")["column"].count().rename("n_columns").to_string())
    if args.report_nan:
        print(table[["column", "family", "nan_fraction", "min", "max"]].to_string(index=False))
    if not all_nan.empty:
        print(f"[features] WARNING: {len(all_nan)} column(s) are empty in this sample: "
              f"{', '.join(all_nan['column'].head(8))}")
    for leg in ("backward", "forward"):
        sub = df[(df["leg"] == leg) & df["truth_class"].notna()]
        col = f"e_over_p_{leg}"
        if col in sub.columns:
            med = sub.groupby("truth_class")[col].median().round(3).to_dict()
            cnt = sub["truth_class"].value_counts().to_dict()
            print(f"[features] median {col} by truth class: {med}  (n={cnt})")
            ok = True
            if leg == "backward" and "e" in med and "pi" in med:
                if not (med["e"] > 0.8 and med["pi"] < 0.4):
                    ok = False
                    print("[gate:FAIL] backward-leg E/p: expected electron peak > 0.8 "
                          "and pion < 0.4")
            if leg == "forward" and "pi" in med:
                if not (med["pi"] < 0.4):
                    ok = False
                    print("[gate:FAIL] forward-leg E/p: expected pion median < 0.4 "
                          "(MIP-like)")
            if not ok and not args.relax_gates:
                return 1
    print("[gate:PASS] feature-table sanity (E/p peaks where physics expects them)"
          if not args.relax_gates else "[features] gates relaxed")
    return 0


# ---------------------------------------------------------------------------
# train / evaluate / importance / compare
# ---------------------------------------------------------------------------

def cmd_train(args) -> int:
    from . import train as pid_train

    files = _read_file_list(args)
    if not args.features and not files:
        raise SystemExit("train: pass --features <table.pkl> or a file list")
    pid_train.train(args.task, model=args.model, features=args.features, files=files or None,
                    dataset_tag=args.dataset_tag, legs=tuple(args.legs.split(",")),
                    limit_files=args.limit_files, max_failures=args.max_file_failures,
                    cache_dir=args.cache_dir or None, n_iter=args.n_iter, n_splits=args.n_splits,
                    seed=args.seed, out_dir=args.model_dir, n_jobs=args.n_jobs,
                    include_event_level=bool(args.include_event_level),
                    enable_ionisation=args.enable_ionisation,
                    calibrate=not args.no_calibrate, gates=not args.relax_gates,
                    class_weight=args.class_weight, use_charge=args.use_charge,
                    allow_small_sample=args.allow_small_sample,
                    include_kinematics=args.include_kinematics,
                    include_track_time=args.include_track_time,
                    min_q2_tier=args.min_q2_tier)
    return 0


def cmd_evaluate(args) -> int:
    from . import evaluate as pid_evaluate
    from . import plots

    res = pid_evaluate.evaluate(args.scores, task=args.task, dataset_tag=args.dataset_tag,
                                model=args.model, signal=args.signal,
                                out_dir=args.out_dir,
                                by=tuple(args.by.split(",")))
    if args.plot:
        stem = os.path.join(args.out_dir, "plots", f"pid-{res['tag']}")
        plots.roc(res["roc"], stem + "-roc.png", title=res["tag"])
        plots.efficiency_vs_fake(res["overall"], stem + "-eff_vs_fake.png", title=res["tag"])
        for variable in ("pt", "eta"):
            if f"vs_{variable}" in res:
                plots.metric_vs_bin(res[f"vs_{variable}"], stem + f"-auc_vs_{variable}.png",
                                    column="auc", title=res["tag"])
    return 0


def cmd_importance(args) -> int:
    from . import importance as pid_importance
    from . import plots

    res = pid_importance.importance(args.dir, model_name=args.model, task=args.task,
                                    kind=args.kind, out_dir=args.out_dir,
                                    dataset_tag=args.dataset_tag, write=not args.no_write)
    check = res["check"]
    print(f"\n[importance] leading feature: {check['top']}")
    for note in check["notes"]:
        print(f"[importance] NOTE: {note}")
    if args.plot:
        stem = os.path.join(args.out_dir, "plots", f"pid-{args.task}_{args.model}_{args.dataset_tag}")
        plots.importance(res["tables"], stem + "-importance.png", title=stem.split("/")[-1])
    return 0 if check["passed"] or args.relax_gates else 1


def cmd_performance(args) -> int:
    from . import performance as pid_performance

    channels = tuple(args.channels.split(","))
    scores_map = None
    if args.scores:
        scores_map = {}
        for item in args.scores:
            if "=" not in item:
                raise SystemExit("performance: --scores expects channel=path "
                                 "(e.g. --scores eid=path/test_scores.pkl)")
            key, path = item.split("=", 1)
            scores_map[key] = path
    result = pid_performance.run(
        channels=channels, dataset_tag=args.dataset_tag, model=args.model,
        out_dir=args.out_dir, plots_dir=args.plots_dir, scores_map=scores_map,
        model_root=args.model_root, binned_curves=not args.no_binned_curves,
        n_thresholds=args.n_thresholds, bin_source=args.bin_source,
        nsigma_method=args.nsigma_method, require_figures=args.require_figures,
        mode=args.mode, lumi_scale=args.lumi_scale, bkg_scale=args.bkg_scale,
        write=not args.no_write)
    summary = result["summary"]
    if summary.empty:
        raise SystemExit(
            "pid performance: no channel could be evaluated - train the matching "
            "tasks first (python -m pid train --task eid|ehad|hadpid "
            f"--dataset-tag {args.dataset_tag})")
    bad = summary[(~summary["valid"].astype(bool))]
    if not bad.empty:
        print(f"[performance] WARNING: {len(bad)} channel(s) have no valid working point: "
              f"{', '.join(bad['channel'])}")
    caution = summary[summary["caution"].astype(str).str.len() > 0]
    for _, r in caution.iterrows():
        print(f"[performance] CAUTION {r['channel']}: {r['caution']}")
    return 0


def cmd_compare(args) -> int:
    from . import compare as pid_compare

    if args.artifact == "all":
        pid_compare.compare_all(task=args.task, model=args.model, out_dir=args.out_dir)
    else:
        pid_compare.compare_artifact(args.artifact, task=args.task, model=args.model,
                                     clean=args.clean, bkg=args.bkg, out_dir=args.out_dir)
    return 0


def cmd_all(args) -> int:
    """One-shot pipeline for the local pair: features -> train -> evaluate -> importance."""
    reference = config.LOCAL_REFERENCE_FILES[args.dataset_tag]
    files = [reference]
    print(f"[all] dataset_tag={args.dataset_tag} using {reference}")
    feats = os.path.join(args.out_dir, f"pid-features_{args.dataset_tag}.pkl")

    from . import features as pid_features

    df = pid_features.build_features(
        files, dataset_tag=args.dataset_tag, legs=("backward", "forward"),
        max_failures=args.max_file_failures,
        cache_dir=args.cache_dir or None, limit_files=1,
        enable_ionisation=args.enable_ionisation)
    feats = os.path.join(args.out_dir, f"pid-features_{args.dataset_tag}.pkl")
    dataset.save_table(df, feats)
    print(f"[all] features: {len(df)} rows x {len(df.columns)} cols -> {feats}")

    tasks = (args.tasks or "eid,ehad,hadpid,pooled").split(",")
    libraries = args.models.split(",") if isinstance(args.models, str) and args.models \
        else [config.MODEL_LIBRARY_DEFAULT]
    for task in tasks:
        for model in libraries:
            stem = os.path.join(args.model_dir, f"{task}_{model}_{args.dataset_tag}")
            cmd_train(argparse.Namespace(
                task=task, model=model, features=feats, file=None, file_list=None,
                dataset_tag=args.dataset_tag, legs="backward,forward", limit_files=None,
                max_file_failures=args.max_file_failures, cache_dir=args.cache_dir,
                out_dir=args.out_dir,
                model_dir=args.model_dir, n_iter=args.n_iter, n_splits=args.n_splits,
                seed=args.seed, include_event_level=args.include_event_level,
                include_track_time=args.include_track_time,
                enable_ionisation=args.enable_ionisation, no_calibrate=args.no_calibrate,
                relax_gates=args.relax_gates, class_weight=args.class_weight,
                use_charge=False, allow_small_sample=True,
                include_kinematics=args.include_kinematics,
                min_q2_tier=None, n_jobs=args.n_jobs))
            cmd_evaluate(argparse.Namespace(
                scores=os.path.join(stem, "test_scores.pkl"), task=task, model=model,
                dataset_tag=args.dataset_tag, signal=None, out_dir=args.out_dir,
                by=args.by, plot=args.plot))
            cmd_importance(argparse.Namespace(
                dir=stem, task=task, model=model, kind="all", dataset_tag=args.dataset_tag,
                out_dir=args.out_dir, no_write=False, plot=args.plot, relax_gates=True))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pid", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("schema-check", help="Verify the ML stack, the branch schema and "
                                            "every track<->detector link in a file.")
    p.add_argument("--dataset", default=None, choices=sorted(config.CAMPAIGN_BY_DATASET_TAG))
    p.add_argument("--file", default=None, help="One ROOT file or root:// URL to audit.")
    p.add_argument("--out-dir", default=config.OUTPUT_DIR)
    p.add_argument("--fmt", default="csv", choices=("csv", "json"))
    p.set_defaults(func=cmd_schema_check)

    p = sub.add_parser("features", help="Build the per-track PID feature table.")
    _add_io_args(p)
    p.add_argument("--legs", default="backward,forward", help="Comma list of legs to build.")
    p.add_argument("--out", default=None, help="Feature-table output path (.pkl).")
    p.add_argument("--enable-ionisation", action="store_true",
                   help="Add the reco-side ionisation proxy columns (off by default).")
    p.add_argument("--report-nan", action="store_true", help="Print the full per-column summary.")
    p.add_argument("--relax-gates", action="store_true")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("train", help="Train + cross-validate one task/model.")
    _add_io_args(p)
    p.add_argument("--task", required=True, choices=sorted(config.TASKS))
    p.add_argument("--model", default=config.MODEL_LIBRARY_DEFAULT, choices=models_choices())
    p.add_argument("--features", default=None, help="Pre-built feature table (.pkl).")
    p.add_argument("--legs", default="backward,forward")
    p.add_argument("--n-iter", type=int, default=config.N_SEARCH_ITERATIONS)
    p.add_argument("--n-splits", type=int, default=config.N_CV_SPLITS)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--n-jobs", type=int, default=config.N_JOBS,
                   help="Threads for the search and learners (default 1: larger values "
                        "have deadlocked OpenMP on small tables - see config.N_JOBS).")
    p.add_argument("--model-dir", default=config.MODEL_DIR)
    p.add_argument("--class-weight", default="balanced", choices=("none", "balanced"))
    p.add_argument("--include-event-level", action="store_true",
                   help="Feed event-occupancy columns to the classifier (ablation, "
                        "not the baseline).")
    p.add_argument("--no-event-level", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--include-track-time", action="store_true",
                   help="Feed CentralCKFTracks.time/timeError (an event timestamp here).")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--use-charge", action="store_true",
                   help="Include signed track charge (NOT recommended: beam-charge dependent).")
    p.add_argument("--enable-ionisation", action="store_true")
    p.add_argument("--relax-gates", action="store_true")
    p.add_argument("--allow-small-sample", action="store_true",
                   help="Train despite a class below the statistics floor (smoke tests).")
    p.add_argument("--include-kinematics", action="store_true",
                   help="Feed p/pT/eta to the classifier (flux shortcut; see pid.config).")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Performance tables from a saved score table.")
    p.add_argument("--scores", required=True, help="test_scores.pkl written by train.")
    p.add_argument("--task", required=True, choices=sorted(config.TASKS))
    p.add_argument("--model", default="")
    p.add_argument("--dataset-tag", required=True, choices=sorted(config.CAMPAIGN_BY_DATASET_TAG))
    p.add_argument("--signal", default=None, help="Class of interest for multiclass tasks.")
    p.add_argument("--by", default="pt,eta", help="Comma list of slicing variables (pt,p,eta).")
    p.add_argument("--out-dir", default=config.OUTPUT_DIR)
    p.add_argument("--plot", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("importance", help="Gain / exact SHAP / permutation importance.")
    p.add_argument("--dir", required=True, help="Artifact directory written by train.")
    p.add_argument("--task", required=True, choices=sorted(config.TASKS))
    p.add_argument("--model", default=config.MODEL_LIBRARY_DEFAULT, choices=models_choices())
    p.add_argument("--dataset-tag", required=True, choices=sorted(config.CAMPAIGN_BY_DATASET_TAG))
    p.add_argument("--kind", default="all", choices=("all", "gain", "shap", "permutation"))
    p.add_argument("--out-dir", default=config.OUTPUT_DIR)
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--relax-gates", action="store_true")
    p.set_defaults(func=cmd_importance)

    p = sub.add_parser("performance",
                       help="Maximum-significance working points and the full "
                            "differential PID performance package (FOM(c), c*, c*(pT), "
                            "efficiency/fake/purity at c*, 2D maps, rejection vs p, "
                            "n-sigma vs p, confusion matrix, overtraining check).")
    p.add_argument("--dataset-tag", required=True, choices=sorted(config.CAMPAIGN_BY_DATASET_TAG))
    p.add_argument("--model", default=config.MODEL_LIBRARY_DEFAULT, choices=models_choices())
    p.add_argument("--channels", default="eid,ehad,Kpi,pK",
                   help="Comma list from pid.config.CHANNELS.")
    p.add_argument("--scores", action="append", default=None,
                   help="Override a channel's score table: channel=/path/to/all_scores.pkl")
    p.add_argument("--mode", default="s_over_sqrt",
                   choices=list(config.SIGNIFICANCE_MODES))
    p.add_argument("--lumi-scale", type=float, default=config.DEFAULT_LUMI_SCALE,
                   help="Scale both classes (FOM ~ sqrt(L)); recorded in every output.")
    p.add_argument("--bkg-scale", type=float, default=1.0,
                   help="Rescale the background yield relative to the measured "
                        "composition (realistic-flux studies); recorded in outputs.")
    p.add_argument("--out-dir", default=config.OUTPUT_DIR)
    p.add_argument("--plots-dir", default=None, help="Default: <out-dir>/plots")
    p.add_argument("--no-binned-curves", action="store_true",
                   help="Skip the per-pT-bin FOM(c) curves (faster, fewer figures).")
    p.add_argument("--n-thresholds", type=int, default=None,
                   help="Threshold grid for the per-bin curves (default from pid.config).")
    p.add_argument("--model-root", default="", help="Root directory of model artifacts "
                                                   "(default: pid.config.MODEL_DIR).")
    p.add_argument("--nsigma-method", default=None,
                   choices=list(config.NSIGMA_METHODS),
                   help="n_sigma estimator (default from pid.config: quantile). "
                        "'raw' fits the bounded score and is deprecated.")
    p.add_argument("--require-figures", action="store_true",
                   help="Exit non-zero if any deliverable figure is missing.")
    p.add_argument("--bin-source", default="reco", choices=("reco", "truth", "both"),
                   help="Kinematic basis for the binned tables/figures: reconstructed "
                        "track quantities (default, what a cut is applied to), truth "
                        "partner's (comparable to the tracking metrics, migration-free), "
                        "or both (truth outputs get a *_truthpt suffix).")
    p.add_argument("--no-write", action="store_true", help="Print summary only.")
    p.set_defaults(func=cmd_performance)

    p = sub.add_parser("compare", help="Clean vs +background comparison of PID artifacts.")
    p.add_argument("--task", required=True, choices=sorted(config.TASKS))
    p.add_argument("--model", default="")
    p.add_argument("--artifact", default="overall",
                   help="'overall', 'vs_pt', 'vs_eta', 'confusion', or 'all'.")
    p.add_argument("--clean", default=None)
    p.add_argument("--bkg", default=None)
    p.add_argument("--out-dir", default=config.OUTPUT_DIR)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("all", help="End-to-end smoke run on the local reference file.")
    p.add_argument("--dataset-tag", default="clean", choices=sorted(config.CAMPAIGN_BY_DATASET_TAG))
    p.add_argument("--tasks", default=None, help="Comma list (default: all four tasks).")
    p.add_argument("--models", default=None, help="Comma list of libraries (default: lightgbm).")
    p.add_argument("--out-dir", default=config.OUTPUT_DIR)
    p.add_argument("--model-dir", default=config.MODEL_DIR)
    p.add_argument("--cache-dir", default=config.FEATURE_CACHE_DIR)
    p.add_argument("--enable-ionisation", action="store_true")
    p.add_argument("--include-event-level", action="store_true")
    p.add_argument("--no-event-level", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--include-track-time", action="store_true")
    p.add_argument("--class-weight", default="balanced", choices=("none", "balanced"))
    p.add_argument("--n-iter", type=int, default=4)
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=config.N_JOBS)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--by", default="pt,eta")
    p.add_argument("--include-kinematics", action="store_true")
    p.add_argument("--no-calibrate", action="store_true",
                   help="Skip probability calibration (smoke runs: tiny samples).")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--relax-gates", action="store_true")
    p.add_argument("--max-file-failures", type=int, default=0)
    p.set_defaults(func=cmd_all)

    return parser


def models_choices() -> list[str]:
    from . import models

    return models.available()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
