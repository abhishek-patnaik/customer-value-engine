"""Command line entry point.

    python -m clv fetch                 download the UCI workbook into data/raw/
    python -m clv run                   run the pipeline on the real data
    python -m clv run --synthetic       same pipeline on simulated data, writes to outputs_synthetic/
    python -m clv powerbi-project       write the Power BI report as a .pbip project to powerbi/
"""

from __future__ import annotations

import argparse
import json
import time

from clv import clean, data
from clv.config import ROOT, load_config, paths_for


def _stage(name: str) -> None:
    print(f"\n[{time.strftime('%H:%M:%S')}] {name}", flush=True)


def cmd_fetch(args) -> None:
    data.fetch(load_config(), force=args.force)


def cmd_run(args) -> None:
    cfg = load_config()
    kind = "synthetic" if args.synthetic else "real"
    paths = paths_for(kind)
    if args.quick:
        cfg["predict"]["bootstrap_rounds"] = 10
        cfg["decide"]["simulations"] = 200

    _stage(f"Loading raw data ({kind})")
    if args.synthetic:
        from clv.synthetic import build

        raw, _ = build()
    else:
        raw = data.load_raw(cfg, refresh=args.refresh)
    print(f"  {len(raw):,} raw rows")

    _stage("Cleaning")
    out = clean.clean(raw, cfg)
    for name in ("lines", "returns", "invoices"):
        out[name].to_parquet(paths.data / f"{name}.parquet", index=False)
    out["audit"].to_csv(paths.tables / "clean_audit.csv", index=False)
    summ = clean.summary(raw, out)
    (paths.tables / "clean_summary.json").write_text(
        json.dumps({**summ, "checks": out["checks"]}, indent=2, default=str))
    print(f"  {summ['customers']:,} customers, {summ['invoices']:,} invoices, "
          f"{summ['first_date']:%d %b %Y} to {summ['last_date']:%d %b %Y}")

    from clv import decide, diagnose, powerbi, predict, report

    _stage("Layer 1: diagnose")
    diag = diagnose.run(out, cfg)
    diagnose.write(diag, paths)
    _stage("Layer 2: predict (two backtests, then the forecast; takes a minute or two)")
    pred = predict.run(out, diag, cfg)
    predict.write(pred, paths)
    lo, hi = pred["planning_range"]
    print(f"  next 26 weeks: model {pred['forecast_total']:,.0f}, planning range {lo:,.0f} to {hi:,.0f}")
    _stage("Layer 3: decide")
    dec = decide.run(pred, diag, cfg)
    decide.write(dec, paths)
    print(f"  spend that pays back: {dec['allocation']['spend'].sum():,.0f} of {dec['budget']:,.0f}")
    _stage("Power BI tables")
    powerbi.write(powerbi.build(out, diag, pred, dec), paths.base / "powerbi")
    _stage("Writing report")
    for p in report.write(summ, out, diag, pred, dec, cfg, paths, synthetic=args.synthetic):
        print(f"  wrote {p.relative_to(ROOT)}")
    print(f"\nDone. Everything is in {paths.base.relative_to(ROOT)}/")


def cmd_pbip(args) -> None:
    from clv import pbip

    folder = args.data_folder or pbip.default_folder()
    for p in pbip.write(ROOT / "powerbi", folder):
        print(f"wrote {p.relative_to(ROOT)}")
    print(f"\nData folder: {folder}\nOpen powerbi/{pbip.NAME}.pbip in Power BI Desktop and click Refresh.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m clv", description="Customer Value Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download the raw workbook from UCI")
    f.add_argument("--force", action="store_true", help="download again even if the file exists")
    f.set_defaults(func=cmd_fetch)

    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("--synthetic", action="store_true", help="use simulated data")
    r.add_argument("--refresh", action="store_true", help="re-read the workbook instead of the cache")
    r.add_argument("--quick", action="store_true", help="fewer bootstrap and sensitivity runs, for a fast check")
    r.set_defaults(func=cmd_run)

    b = sub.add_parser("powerbi-project", help="write the Power BI project (.pbip) to powerbi/")
    b.add_argument("--data-folder", help="folder holding the Power BI CSVs, as Power BI will see it "
                                         "(default: outputs/powerbi in this project)")
    b.set_defaults(func=cmd_pbip)

    args = parser.parse_args(argv)
    args.func(args)
