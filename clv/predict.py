"""Layer 2: what each customer is likely to be worth over the next six months.

How it works
    1. Hold out the last 26 weeks. Fit on everything before, predict the
       holdout, compare with what customers actually did, next to simple
       baselines a business would otherwise use.
    2. Refit on all two years and forecast the 26 weeks after the data ends.

The store is strongly seasonal and BG/NBD assumes a purchase rate that is
flat through the year. Instead of bolting a correction on afterwards, time is
measured in "seasonal weeks": a week in November counts for more than a week
in January, in proportion to how much customers who are still active buy in
that month. The model is fitted and forecasts on that clock. The plain calendar
version is kept as a comparison so the effect of this choice is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from clv.diagnose import purchase_days
from clv.models import BGNBD, MBGNBD, GammaGamma

MODELS = {"bgnbd": (BGNBD, "BG/NBD"), "mbgnbd": (MBGNBD, "MBG/NBD")}

DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


# ---------------------------------------------------------------- the clock

def seasonal_index(days: pd.DataFrame, end: pd.Timestamp, proof_weeks: int = 4, prior: float = 50.0,
                   iterations: int = 200) -> pd.Series:
    """Relative purchase intensity by calendar month, 1 = an average month.

    The trap here is that a group of customers buys less over time simply
    because some of them leave, and with a year or less of history that
    decline cannot be told apart from seasonality. My first version fell
    into it, and the simulated store (which has no seasonality at all)
    came out with a strong one.

    So the index is estimated only from customers known to be active for the
    whole stretch: they bought at least once in the `proof_weeks` before
    `end`. Under the model, someone still active buys at a steady personal
    rate, so any month to month pattern in their purchases is seasonality.
    The counts are fitted with a Poisson model that has a fixed effect per
    customer, so a heavy buyer joining the panel cannot pass for a busy
    month. Each customer counts from the day after their first purchase to
    the start of the proof window. Each month is pulled toward 1 by a prior
    worth `prior` purchase days, so a month seen only a handful of times
    cannot swing the clock. Months never observed take the average of their
    neighbours.
    """
    proof_start = end - pd.Timedelta(weeks=proof_weeks)
    alive = days.loc[(days["day"] >= proof_start) & (days["day"] < end), "customer_id"].unique()
    d = days[days["customer_id"].isin(alive) & (days["day"] < proof_start)]
    first = d.groupby("customer_id")["day"].min()
    d = d[d["day"] > d["customer_id"].map(first)]
    # exposure: customer days per calendar month, from day after first purchase to proof_start
    cal = pd.date_range(first.min(), proof_start - pd.Timedelta(days=1), freq="D")
    month_of_day = cal.to_period("M")
    months = month_of_day.unique()
    m_index = {m: i for i, m in enumerate(months)}
    day_pos = (cal - cal[0]).days.to_numpy()
    cum = np.zeros((len(months), len(cal) + 1))
    for i, m in enumerate(months):
        cum[i, 1:] = np.cumsum(month_of_day == m)
    start_pos = ((first + pd.Timedelta(days=1)) - cal[0]).dt.days.clip(lower=0, upper=len(cal)).to_numpy()
    E = (cum[:, -1][None, :] - cum[:, start_pos].T)          # customers x months
    cust_pos = {c: i for i, c in enumerate(first.index)}
    N = np.zeros_like(E)
    cm = d.assign(m=d["day"].dt.to_period("M")).groupby(["customer_id", "m"]).size()
    for (c, m), v in cm.items():
        N[cust_pos[c], m_index[m]] += v
    moy = np.array([m.month for m in months]) - 1
    seen = np.bincount(moy, weights=E.sum(axis=0), minlength=12) > 0
    s_ = np.ones(12)
    for _ in range(iterations):
        rate = N.sum(axis=1) / np.maximum((E * s_[moy][None, :]).sum(axis=1), 1e-12)
        num = np.bincount(moy, weights=N.sum(axis=0), minlength=12)
        den = np.bincount(moy, weights=(E * rate[:, None]).sum(axis=0), minlength=12)
        s_ = np.where(seen, (num + prior) / (den + prior), np.nan)
        s_ = s_ / np.nanmean(s_)
        s_ = np.where(seen, s_, 1.0)
    idx = np.where(seen, s_, np.nan)
    for k in np.where(~seen)[0]:
        idx[k] = np.nanmean([idx[(k - 1) % 12], idx[(k + 1) % 12]])
    idx = idx / ((idx * DAYS_IN_MONTH).sum() / DAYS_IN_MONTH.sum())
    return pd.Series(idx, index=range(1, 13), name="index")


class Clock:
    """Turns dates into weeks. Seasonal weeks if an index is given."""

    def __init__(self, start: pd.Timestamp, stop: pd.Timestamp, index: pd.Series | None = None):
        self.start = start.normalize()
        days = pd.date_range(self.start, stop.normalize(), freq="D")
        w = np.ones(len(days)) if index is None else index.reindex(days.month).to_numpy()
        self.cum = pd.Series(np.r_[0.0, np.cumsum(w)][:-1] / 7.0, index=days)

    def __call__(self, when) -> np.ndarray:
        when = pd.to_datetime(pd.Series(when)).dt.normalize()
        return self.cum.reindex(when).to_numpy()


# ---------------------------------------------------------------- data

def summarise(days: pd.DataFrame, cal_end: pd.Timestamp, clock: Clock) -> pd.DataFrame:
    """One row per customer seen before cal_end: x, t_x, T, average repeat spend."""
    d = days[days["day"] < cal_end].sort_values(["customer_id", "day"]).copy()
    d["n"] = d.groupby("customer_id").cumcount()
    g = d.groupby("customer_id")
    s = pd.DataFrame({"first": g["day"].min(), "last": g["day"].max(), "x": g["day"].size() - 1,
                      "revenue": g["revenue"].sum()})
    s["first_value"] = d[d["n"] == 0].set_index("customer_id")["revenue"]
    rep = d[d["n"] > 0].groupby("customer_id")["revenue"]
    s["m"] = rep.mean().reindex(s.index).fillna(0.0)
    end_w = clock([cal_end])[0]
    s["t_first"] = clock(s["first"].to_numpy())
    s["tx"] = clock(s["last"].to_numpy()) - s["t_first"]
    s["T"] = end_w - s["t_first"]
    s["T_calendar_weeks"] = (cal_end - s["first"]).dt.days / 7
    # revenue in the last 26 calendar weeks of the window, for the baseline
    recent = d[d["day"] >= cal_end - pd.Timedelta(weeks=26)].groupby("customer_id")["revenue"].sum()
    s["revenue_last_26w"] = recent.reindex(s.index).fillna(0.0)
    s["purchases_last_26w"] = (d[d["day"] >= cal_end - pd.Timedelta(weeks=26)]
                               .groupby("customer_id").size().reindex(s.index).fillna(0))
    return s


def actuals(days: pd.DataFrame, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    d = days[(days["day"] >= start) & (days["day"] < stop)]
    return d.groupby("customer_id").agg(actual_purchases=("day", "size"), actual_revenue=("revenue", "sum"))


# ---------------------------------------------------------------- fit and score

def fit(s: pd.DataFrame, model: str = "mbgnbd") -> tuple[BGNBD, GammaGamma]:
    bg = MODELS[model][0]().fit(s["x"], s["tx"], s["T"])
    rep = s[s["x"] > 0]
    gg = GammaGamma().fit(rep["x"], rep["m"])
    return bg, gg


def score(bg: BGNBD, gg: GammaGamma, s: pd.DataFrame, horizon_weeks: float) -> pd.DataFrame:
    out = pd.DataFrame(index=s.index)
    out["p_alive"] = bg.p_alive(s["x"], s["tx"], s["T"])
    out["expected_purchases"] = bg.expected_purchases(horizon_weeks, s["x"], s["tx"], s["T"])
    out["expected_spend"] = gg.expected_spend(s["x"], s["m"])
    out["expected_value"] = out["expected_purchases"] * out["expected_spend"]
    return out


# ---------------------------------------------------------------- evaluation

def _metrics(pred: pd.Series, actual: pd.Series) -> dict:
    n_top = max(1, int(round(len(pred) * 0.1)))
    top = pred.sort_values(ascending=False, kind="stable").index[:n_top]
    return {
        "predicted_total": float(pred.sum()),
        "actual_total": float(actual.sum()),
        "total_error_pct": float(pred.sum() / actual.sum() - 1),
        "mae": float((pred - actual).abs().mean()),
        "spearman": float(stats.spearmanr(pred, actual).statistic),
        "top10_capture": float(actual.loc[top].sum() / actual.sum()),
    }


def evaluate(s: pd.DataFrame, act: pd.DataFrame, preds: dict[str, pd.DataFrame], holdout_weeks: float):
    a = act.reindex(s.index).fillna(0.0)
    rows, purchases = [], []
    for name, p in preds.items():
        rows.append({"method": name, **_metrics(p["expected_value"], a["actual_revenue"])})
        purchases.append({"method": name, **_metrics(p["expected_purchases"], a["actual_purchases"])})
    # baselines a team would use without a model
    base = {
        "Same as last 26 weeks": (s["revenue_last_26w"], s["purchases_last_26w"]),
        "Historical average rate": (s["revenue"] / s["T_calendar_weeks"].clip(lower=1) * holdout_weeks,
                                    (s["x"] + 1) / s["T_calendar_weeks"].clip(lower=1) * holdout_weeks),
    }
    for name, (rev, n) in base.items():
        rows.append({"method": name, **_metrics(rev, a["actual_revenue"])})
        purchases.append({"method": name, **_metrics(n, a["actual_purchases"])})
    return pd.DataFrame(rows), pd.DataFrame(purchases)


def deciles(pred: pd.Series, actual: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"pred": pred, "actual": actual})
    df["decile"] = pd.qcut(df["pred"].rank(method="first", ascending=False), 10, labels=range(1, 11)).astype(int)
    out = df.groupby("decile").agg(customers=("pred", "size"), predicted=("pred", "sum"), actual=("actual", "sum"))
    out["predicted_per_customer"] = out["predicted"] / out["customers"]
    out["actual_per_customer"] = out["actual"] / out["customers"]
    out["share_of_actual"] = out["actual"] / out["actual"].sum()
    return out.reset_index()


def segment_check(s, pred, act, base_ids) -> pd.DataFrame:
    """Does the model work as well for the existing base as for new customers?"""
    a = act.reindex(s.index).fillna(0.0)
    grp = np.where(s.index.isin(base_ids), "Existing base", "Acquired during the data")
    df = pd.DataFrame({"group": grp, "pred": pred["expected_value"], "actual": a["actual_revenue"],
                       "pred_n": pred["expected_purchases"], "actual_n": a["actual_purchases"]})
    out = df.groupby("group").agg(customers=("pred", "size"), predicted=("pred", "sum"), actual=("actual", "sum"),
                                  predicted_purchases=("pred_n", "sum"), actual_purchases=("actual_n", "sum"))
    out["error_pct"] = out["predicted"] / out["actual"] - 1
    return out.reset_index()


def alive_check(pred: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    a = act.reindex(pred.index).fillna(0.0)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001]
    labels = ["under 20%", "20 to 40%", "40 to 60%", "60 to 80%", "80 to 90%", "90% and over"]
    b = pd.cut(pred["p_alive"], bins, labels=labels, right=False)
    df = pd.DataFrame({"band": b, "p_alive": pred["p_alive"], "bought": a["actual_purchases"] > 0,
                       "exp_purchases": pred["expected_purchases"], "actual_purchases": a["actual_purchases"]})
    return (df.groupby("band", observed=False)
            .agg(customers=("bought", "size"), mean_p_alive=("p_alive", "mean"), share_bought=("bought", "mean"),
                 expected_purchases=("exp_purchases", "mean"), actual_purchases=("actual_purchases", "mean"))
            .reset_index())


def bootstrap_total(s: pd.DataFrame, horizon: float, rounds: int, seed: int, model: str) -> np.ndarray:
    """Parameter uncertainty on the total forecast: refit on resampled customers."""
    rng = np.random.default_rng(seed)
    totals = []
    for _ in range(rounds):
        idx = rng.integers(0, len(s), len(s))
        b = s.iloc[idx]
        bg, gg = fit(b, model)
        totals.append(float(score(bg, gg, s, horizon)["expected_value"].sum()))
    return np.array(totals)


# ---------------------------------------------------------------- run

def backtest(days, base_ids, start, far, cal_end, hold_end, holdout_weeks, main: str) -> dict:
    """Fit on everything before cal_end, predict [cal_end, hold_end).

    Every model and clock combination is scored, so the choice of the main
    one is visible rather than asserted. `main` is the one the detailed
    checks are run on.
    """
    idx = seasonal_index(days, cal_end)
    clocks = {"seasonal": Clock(start, far, idx), "calendar": Clock(start, far)}
    preds, fits, summaries = {}, {}, {}
    for kind, clock in clocks.items():
        s = summarise(days, cal_end, clock)
        summaries[kind] = s
        h = clock([hold_end])[0] - clock([cal_end])[0]
        for model, (_, label) in MODELS.items():
            bg, gg = fit(s, model)
            name = f"{label} + Gamma-Gamma, {kind} clock"
            preds[name] = score(bg, gg, s, h)
            fits[name] = {**bg.params, **gg.params, "converged": bool(bg.converged and gg.converged)}
    s_main = summaries["seasonal"]
    main_name = f"{MODELS[main][1]} + Gamma-Gamma, seasonal clock"
    ordered = {main_name: preds[main_name], **{k: v for k, v in preds.items() if k != main_name}}
    act = actuals(days, cal_end, hold_end)
    eval_rev, eval_n = evaluate(s_main, act, ordered, holdout_weeks)
    a = act.reindex(s_main.index).fillna(0.0)
    bought = a["actual_purchases"] > 0
    rep = s_main[s_main["x"] > 0]
    m = preds[main_name]
    return {
        "cal_end": cal_end, "hold_end": hold_end, "season_index": idx, "params": fits, "main": main_name,
        "eval_revenue": eval_rev, "eval_purchases": eval_n,
        "deciles": deciles(m["expected_value"], a["actual_revenue"]),
        "segment_check": segment_check(s_main, m, act, base_ids),
        "alive_check": alive_check(m, act),
        "independence_spearman": float(stats.spearmanr(rep["x"], rep["m"]).statistic),
        "spend_check": {
            "predicted_mean_order": float(m.loc[bought, "expected_spend"].mean()),
            "actual_mean_order": float((a.loc[bought, "actual_revenue"] / a.loc[bought, "actual_purchases"]).mean()),
        },
        "scores": m.join(a),
    }


def run(clean_out: dict, diag: dict, cfg: dict) -> dict:
    pc = cfg["predict"]
    days = purchase_days(clean_out["invoices"])
    snapshot = diag["snapshot"]
    hw = pc["holdout_weeks"]
    start = days["day"].min()
    base_ids = diag["customers"].index[diag["customers"]["is_existing_base"]]
    far = snapshot + pd.Timedelta(weeks=pc["horizon_weeks"] + 8)

    # 1. two backtests: the busy half of the year and the quiet half
    folds = {}
    for name, back in (("Jun to Dec 2011 (busy season)", 0), ("Dec 2010 to Jun 2011 (quiet season)", 1)):
        hold_end = snapshot - pd.Timedelta(weeks=hw * back)
        cal_end = hold_end - pd.Timedelta(weeks=hw)
        folds[name] = backtest(days, base_ids, start, far, cal_end, hold_end, hw, pc["model"])

    # 2. forecast: refit on everything, predict the next 26 weeks
    idx_full = seasonal_index(days, snapshot)
    clock = Clock(start, far, idx_full)
    s_full = summarise(days, snapshot, clock)
    bg, gg = fit(s_full, pc["model"])
    fend = snapshot + pd.Timedelta(weeks=pc["horizon_weeks"])
    h = clock([fend])[0] - clock([snapshot])[0]
    fc = score(bg, gg, s_full, h)
    cal_clock = Clock(start, far)
    s_flat = summarise(days, snapshot, cal_clock)
    fc["expected_value_calendar_clock"] = score(*fit(s_flat, pc["model"]), s_flat, pc["horizon_weeks"])["expected_value"]
    boot = bootstrap_total(s_full, h, pc["bootstrap_rounds"], pc["random_seed"], pc["model"])
    lo, hi = np.quantile(boot, [(1 - pc["interval"]) / 2, (1 + pc["interval"]) / 2])

    # month by month split of the forecast, for the dashboard line
    bounds = list(pd.date_range(snapshot, fend, freq="MS"))
    bounds = [snapshot] + [b_ for b_ in bounds if snapshot < b_ < fend] + [fend]
    cum = [float((bg.expected_purchases(clock([b_])[0] - clock([snapshot])[0], s_full["x"], s_full["tx"],
                                        s_full["T"]) * fc["expected_spend"]).sum()) for b_ in bounds]
    monthly_fc = pd.DataFrame({"period_start": bounds[:-1], "period_end": bounds[1:],
                               "forecast_revenue": np.diff(cum)})

    cust = s_full[["first", "last", "x", "revenue", "m"]].rename(
        columns={"first": "first_purchase", "last": "last_purchase", "x": "repeat_purchases",
                 "m": "avg_repeat_order"}).join(fc)
    cust["shrinkage_to_own_average"] = gg.shrinkage(s_full["x"])
    cust = cust.sort_values("expected_value", ascending=False)
    cust["value_rank"] = np.arange(1, len(cust) + 1)

    # Planning range: the forecast corrected by the worst and best backtest
    # errors. The quiet season fold matches the forecast window, so its
    # correction is the conservative case the budget is planned on.
    errs = {k: f["eval_revenue"].iloc[0]["total_error_pct"] for k, f in folds.items()}
    total = float(fc["expected_value"].sum())
    planning = (total / (1 + max(errs.values())), total / (1 + min(errs.values())))
    quiet = next(k for k in folds if "quiet" in k)
    cust["planning_value"] = cust["expected_value"] / (1 + errs[quiet])

    return {
        "snapshot": snapshot, "forecast_end": fend, "folds": folds,
        "season_index_forecast": idx_full, "forecast_params": {**bg.params, **gg.params},
        "converged": bool(bg.converged and gg.converged
                          and all(p["converged"] for f in folds.values() for p in f["params"].values())),
        "horizon_seasonal_weeks": float(h),
        "new_customer_expected_repeat_26w": float(bg.expected_purchases_new(h)),
        "population_mean_order": gg.population_mean,
        "independence_spearman": float(stats.spearmanr(s_full.loc[s_full["x"] > 0, "x"],
                                                       s_full.loc[s_full["x"] > 0, "m"]).statistic),
        "customers": cust, "forecast_total": total,
        "forecast_total_calendar_clock": float(fc["expected_value_calendar_clock"].sum()),
        "bootstrap_interval": (float(lo), float(hi)), "bootstrap": boot, "planning_range": planning,
        "backtest_errors": errs, "planning_scale": 1 / (1 + errs[quiet]),
        "monthly_forecast": monthly_fc.assign(planning_revenue=monthly_fc["forecast_revenue"] / (1 + errs[quiet])),
    }


# ---------------------------------------------------------------- outputs

def write(p: dict, paths) -> dict:
    import matplotlib.pyplot as plt

    from clv import viz

    t, f = paths.tables, paths.figures
    rows = []
    for fold, r in p["folds"].items():
        for kind, tab in (("revenue", r["eval_revenue"]), ("purchases", r["eval_purchases"])):
            rows.append(tab.assign(fold=fold, target=kind))
    pd.concat(rows).to_csv(t / "predict_backtest.csv", index=False)
    for fold, r in p["folds"].items():
        tag = "busy" if "busy" in fold else "quiet"
        r["deciles"].to_csv(t / f"predict_deciles_{tag}.csv", index=False)
        r["alive_check"].to_csv(t / f"predict_alive_check_{tag}.csv", index=False)
        r["segment_check"].to_csv(t / f"predict_group_check_{tag}.csv", index=False)
    params = pd.DataFrame({f"{fold} | {name}": v for fold, r in p["folds"].items() for name, v in r["params"].items()}).T
    params.loc["forecast"] = pd.Series(p["forecast_params"])
    params.to_csv(t / "predict_parameters.csv")
    pd.DataFrame({"month": range(1, 13),
                  **{f"index_{('busy' if 'busy' in k else 'quiet')}_fold": r["season_index"].to_numpy()
                     for k, r in p["folds"].items()},
                  "index_forecast": p["season_index_forecast"].to_numpy()}).to_csv(t / "predict_season_index.csv",
                                                                                     index=False)
    p["monthly_forecast"].to_csv(t / "predict_monthly_forecast.csv", index=False)
    c = p["customers"].copy()
    c.index.name = "customer_id"
    c.to_csv(t / "predict_customer_values.csv")
    figs = {}

    # 7. backtest: total error by method and season
    short = {"seasonal clock": "Model, seasonal clock", "calendar clock": "Model, calendar clock"}
    folds = list(p["folds"])
    methods = []
    for m in p["folds"][folds[0]]["eval_revenue"]["method"]:
        if m.startswith("MBG"):
            continue
        methods.append(m)
    labels = []
    for m in methods:
        lab = m
        for k, v in short.items():
            if k in m:
                lab = v
        labels.append(lab)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(methods))
    h = 0.36
    for i, (fold, colour) in enumerate(zip(folds, [viz.SERIES[1], viz.SERIES[0]])):
        e = p["folds"][fold]["eval_revenue"].set_index("method").loc[methods, "total_error_pct"].to_numpy()
        yy = y + (h / 2 if i == 0 else -h / 2)
        ax.barh(yy, e, height=h, color=colour, label=fold)
        for yi, v in zip(yy, e):
            ax.text(v + (0.03 if v >= 0 else -0.03), yi, f"{v * 100:+.0f}%", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=viz.TEXT_2)
    ax.axvline(0, color=viz.TEXT_2, lw=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(viz.pct)
    lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1])) + 0.25
    ax.set_xlim(-0.6, lim)
    ax.grid(axis="y", visible=False)
    ax.set_title("How far off each method was on the total", pad=40)
    viz.legend_top(ax, 2)
    viz.subtitle(ax, "Predicted vs actual 26-week revenue from customers known at the cutoff. 0% is perfect.",
                 above_legend=True)
    figs["backtest"] = viz.save(fig, f / "07_backtest.png")

    # 8. deciles: predicted vs actual per customer
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, fold in zip(axes, folds):
        d = p["folds"][fold]["deciles"]
        x = d["decile"].to_numpy()
        ax.bar(x - 0.2, d["predicted_per_customer"], width=0.4, color=viz.SERIES[0], label="Predicted")
        ax.bar(x + 0.2, d["actual_per_customer"], width=0.4, color=viz.SERIES[1], label="Actual")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(viz.gbp)
        ax.set_xticks(x)
        ax.set_xlabel("Customers ranked by predicted value, decile 1 = top 10%")
        ax.grid(axis="x", visible=False)
        ax.text(0, 1.02, fold, transform=ax.transAxes, color=viz.TEXT, fontsize=10.5, va="bottom")
        top = d.iloc[0]["share_of_actual"]
        ax.text(0.98, 0.95, f"top 10% held {top * 100:.0f}% of actual revenue", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=viz.TEXT_2)
    axes[0].set_ylabel("Revenue per customer over 26 weeks (log scale)")
    fig.suptitle("The ranking holds up in both seasons", x=0.01, ha="left", fontsize=13, fontweight="bold",
                 color=viz.TEXT, y=1.06)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncols=2, borderaxespad=0)
    figs["deciles"] = viz.save(fig, f / "08_deciles.png")

    # 9. seasonal index
    fig, ax = plt.subplots(figsize=(9, 4.2))
    idx = p["season_index_forecast"]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    window = [12, 1, 2, 3, 4, 5, 6]
    colours = [viz.SERIES[0] if m in window else viz.NEUTRAL for m in idx.index]
    ax.bar(range(12), idx.to_numpy(), color=colours, width=0.7)
    ax.axhline(1, color=viz.TEXT_2, lw=1, ls=(0, (4, 3)), zorder=0)
    for i, v in enumerate(idx.to_numpy()):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8.5, color=viz.TEXT_2)
    ax.set_xticks(range(12), months)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, idx.max() * 1.15)
    ratio = idx.max() / idx.min()
    full = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December"]
    ax.set_title(f"A week in {full[int(idx.idxmax()) - 1]} counts for {ratio:.1f} weeks in "
                 f"{full[int(idx.idxmin()) - 1]}", pad=26)
    viz.subtitle(ax, "Seasonal index from customers known to be active, 1 = average month. "
                     "Blue = months in the forecast window.")
    figs["season"] = viz.save(fig, f / "09_season_index.png")

    # 10. P(alive) by recency and frequency
    c = p["customers"]
    rec_w = (p["snapshot"] - c["last_purchase"]).dt.days / 7
    fb = pd.cut(c["repeat_purchases"], [-1, 0, 1, 3, 7, 15, 10_000], labels=["0", "1", "2 to 3", "4 to 7", "8 to 15", "16+"])
    rb = pd.cut(rec_w, [-1, 4, 13, 26, 52, 78, 200], labels=["under 1m", "1 to 3m", "3 to 6m", "6 to 12m", "12 to 18m", "18m+"])
    grid = c.assign(fb=fb, rb=rb).pivot_table(index="fb", columns="rb", values="p_alive", aggfunc="mean", observed=False)
    n = c.assign(fb=fb, rb=rb).pivot_table(index="fb", columns="rb", values="p_alive", aggfunc="size", observed=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(grid.to_numpy()[::-1], cmap=viz.BLUES, vmin=0, vmax=1, aspect="auto")
    G, N = grid.to_numpy()[::-1], n.to_numpy()[::-1]
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if N[i, j] >= 5 and not np.isnan(G[i, j]):
                ax.text(j, i, f"{G[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if G[i, j] > 0.55 else viz.TEXT)
    ax.set_xticks(range(G.shape[1]), grid.columns)
    ax.set_yticks(range(G.shape[0]), list(grid.index)[::-1])
    ax.set_xlabel("Time since last purchase")
    ax.set_ylabel("Repeat purchases")
    ax.grid(False)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("A frequent buyer who goes quiet is a warning sign", pad=40)
    viz.subtitle(ax, "Model's average probability the customer is still active, as of 10 Dec 2011. Cells under 5 customers "
                     "left blank.\nOne-off buyers get the benefit of the doubt, which is probably too generous.")
    figs["p_alive"] = viz.save(fig, f / "10_p_alive.png")
    return figs
