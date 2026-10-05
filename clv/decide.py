"""Layer 3: where a fixed retention budget should go, and how sure we can be.

There is no campaign history in this data, so nothing here can measure how
customers respond to marketing. What the data does give is how much each
customer is likely to be worth and how likely they are to have already gone.
The response side is a set of assumptions, written down in config.toml with
a range for each, and every result is re-run across those ranges. A
recommendation that only holds at one exact set of guesses is not a
recommendation.

The response model, per customer i and spend x:

    incremental revenue(x) = M_i * (1 - exp(-x / k))

    M_i = lift * V_i + recovery * R_i
    V_i = planning value: expected revenue over the next 26 weeks
    R_i = value at risk: what the customer would be worth if still active,
          times the probability they have already left
          = V_i * (1 - p_alive) / p_alive

`lift` is the most a campaign can add to a customer who is still buying,
`recovery` is the share of at-risk value a win-back can bring back, and `k`
is the spend per account at which about two thirds of that maximum is
reached. Returns diminish, so the best plan spreads money until the last
pound spent on every account earns the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clv.diagnose import SEGMENT_ORDER


def customer_frame(pred: dict, rfm: pd.DataFrame) -> pd.DataFrame:
    c = pred["customers"][["planning_value", "expected_value", "p_alive", "revenue", "repeat_purchases"]].copy()
    c["segment"] = rfm["segment"].reindex(c.index)
    p = c["p_alive"].clip(lower=0.01)
    c["value_at_risk"] = c["planning_value"] * (1 - p) / p
    return c


def max_lift(c: pd.DataFrame, a: dict) -> np.ndarray:
    return a["lift"] * c["planning_value"].to_numpy() + a["recovery"] * c["value_at_risk"].to_numpy()


def optimal_spend(M: np.ndarray, k: float, margin: float, budget: float | None, force: bool = False) -> np.ndarray:
    """Spend per customer that maximises margin minus spend, within budget.

    Setting the return on the last pound equal across customers gives
        x_i = k * ln(margin * M_i / (k * m))   when positive,
    where m is that return. Left alone, spending stops where m = 1, the
    point past which a pound spent brings back less than a pound of margin.
    With a budget, m is raised until the plan fits. With force=True the
    whole budget is spent even past m = 1, which is how a "use it or lose
    it" budget behaves and is shown only for comparison.
    """
    def spend(m):
        with np.errstate(divide="ignore"):
            x = k * np.log(margin * M / (k * m))
        return np.clip(np.nan_to_num(x, nan=0.0, neginf=0.0), 0, None)

    x = spend(1.0)
    if budget is None or (x.sum() <= budget and not force):
        return x
    lo, hi = 1e-9, max(1.0, float(margin * M.max() / k))
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if spend(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
    return spend(hi)


def outcome(x: np.ndarray, M: np.ndarray, k: float, margin: float) -> dict:
    inc = M * (1 - np.exp(-x / k))
    spend = float(x.sum())
    profit = float(margin * inc.sum() - spend)
    return {"spend": spend, "incremental_revenue": float(inc.sum()), "incremental_profit": profit,
            "roi": profit / spend if spend else np.nan, "inc": inc}


def strategies(c: pd.DataFrame, M: np.ndarray, a: dict, budget: float) -> dict[str, np.ndarray]:
    """The plan from the model next to the plans a team might default to."""
    n = len(c)
    seg = c["segment"].to_numpy()
    out = {"Optimised by expected return": optimal_spend(M, a["k"], a["margin"], budget),
           "Optimised, but forced to spend it all": optimal_spend(M, a["k"], a["margin"], budget, force=True)}
    out["Same spend on every customer"] = np.full(n, budget / n)
    best = np.isin(seg, ["Champions", "Loyal"])
    out["Only the best customers"] = np.where(best, budget / best.sum(), 0.0)
    lapsed = np.isin(seg, ["At risk", "Can't lose them", "Hibernating", "Lost", "About to sleep"])
    out["Win back everyone lapsing"] = np.where(lapsed, budget / lapsed.sum(), 0.0)
    return out


def by_segment(c: pd.DataFrame, x: np.ndarray, inc: np.ndarray, margin: float) -> pd.DataFrame:
    df = c.assign(spend=x, inc=inc, targeted=x > 0)
    g = df.groupby("segment").agg(
        customers=("spend", "size"), targeted=("targeted", "sum"), spend=("spend", "sum"),
        incremental_revenue=("inc", "sum"), planning_value=("planning_value", "sum"),
        value_at_risk=("value_at_risk", "sum"), mean_p_alive=("p_alive", "mean"))
    g["mean_p_alive_targeted"] = df[df["targeted"]].groupby("segment")["p_alive"].mean()
    g["incremental_profit"] = margin * g["incremental_revenue"] - g["spend"]
    g["roi"] = g["incremental_profit"] / g["spend"].where(g["spend"] > 0)
    g["share_of_budget"] = g["spend"] / g["spend"].sum()
    g["spend_per_targeted"] = g["spend"] / g["targeted"].where(g["targeted"] > 0)
    return g.reindex([s for s in SEGMENT_ORDER if s in g.index]).reset_index()


def draw_assumptions(rng, ranges: dict, n: int, planning_scale: float) -> pd.DataFrame:
    """Each assumption is drawn uniformly from its range. The value scale is
    drawn between the conservative planning case and the raw model forecast."""
    d = {k: rng.uniform(lo, hi, n) for k, (lo, hi) in ranges.items()}
    d["value_scale"] = rng.uniform(1.0, 1.0 / planning_scale, n)
    return pd.DataFrame(d)


def budget_curve(c, a, budgets) -> pd.DataFrame:
    M = max_lift(c, a)
    rows = []
    for b in budgets:
        x = optimal_spend(M, a["k"], a["margin"], b, force=True)
        o = outcome(x, M, a["k"], a["margin"])
        rows.append({"budget": b, **{k: v for k, v in o.items() if k != "inc"}})
    df = pd.DataFrame(rows).drop_duplicates("budget").sort_values("budget").reset_index(drop=True)
    # return on the last pound: change in profit per extra pound of spend
    df["marginal_return"] = (df["incremental_profit"].diff() / df["spend"].diff()).fillna(np.nan)
    return df


def run(pred: dict, diag: dict, cfg: dict) -> dict:
    dc = cfg["decide"]
    a = {"lift": dc["lift"], "recovery": dc["recovery"], "k": dc["spend_scale"], "margin": dc["gross_margin"]}
    budget = float(dc["budget"])
    c = customer_frame(pred, diag["rfm"])
    M = max_lift(c, a)

    plans = strategies(c, M, a, budget)
    base = {name: outcome(x, M, a["k"], a["margin"]) for name, x in plans.items()}
    x_opt = plans["Optimised by expected return"]
    alloc = by_segment(c, x_opt, base["Optimised by expected return"]["inc"], a["margin"])
    unconstrained = optimal_spend(M, a["k"], a["margin"], None).sum()

    # sensitivity: redo everything across the assumption ranges
    rng = np.random.default_rng(dc["random_seed"])
    ranges = {"lift": dc["lift_range"], "recovery": dc["recovery_range"], "k": dc["spend_scale_range"],
              "margin": dc["gross_margin_range"]}
    draws = draw_assumptions(rng, ranges, dc["simulations"], pred["planning_scale"])
    seg = c["segment"].to_numpy()
    segs = [s for s in SEGMENT_ORDER if s in set(seg)]
    seg_idx = {s: seg == s for s in segs}
    share_rows, strat_rows, ceilings = [], [], []
    for _, d in draws.iterrows():
        ad = {"lift": d["lift"], "recovery": d["recovery"], "k": d["k"], "margin": d["margin"]}
        Md = max_lift(c, ad) * d["value_scale"]
        pd_ = strategies(c, Md, ad, budget)
        res = {name: outcome(x, Md, ad["k"], ad["margin"])["incremental_profit"] for name, x in pd_.items()}
        # the base case plan, judged under this draw's assumptions
        res["Base case plan under these assumptions"] = outcome(x_opt, Md, ad["k"], ad["margin"])["incremental_profit"]
        strat_rows.append(res)
        xo = pd_["Optimised by expected return"]
        tot = xo.sum()
        ceilings.append(float(optimal_spend(Md, ad["k"], ad["margin"], None).sum()))
        share_rows.append({s: xo[m].sum() for s, m in seg_idx.items()})
    spends = pd.DataFrame(share_rows)
    sims = pd.DataFrame(strat_rows)
    q = (1 - dc["interval"]) / 2
    share_summary = pd.DataFrame({
        "segment": segs,
        "base_case_spend": [alloc.set_index("segment")["spend"].get(s, 0.0) for s in segs],
        "median_spend": spends.median().reindex(segs).to_numpy(),
        "low_spend": spends.quantile(q).reindex(segs).to_numpy(),
        "high_spend": spends.quantile(1 - q).reindex(segs).to_numpy(),
        "funded_in_share_of_runs": (spends > 100.0).mean().reindex(segs).to_numpy(),
    })
    opt = "Optimised by expected return"
    strat_summary = pd.DataFrame([{
        "strategy": name,
        "base_case_profit": base[name]["incremental_profit"] if name in base else np.nan,
        "base_case_roi": base[name]["roi"] if name in base else np.nan,
        "median_profit": float(sims[name].median()),
        "low_profit": float(sims[name].quantile(q)),
        "high_profit": float(sims[name].quantile(1 - q)),
        "share_of_runs_losing_money": float((sims[name] < 0).mean()),
        "optimised_beats_it": float((sims[opt] > sims[name]).mean()) if name != opt else np.nan,
    } for name in sims.columns])
    regret = float(((sims[opt] - sims["Base case plan under these assumptions"]) / sims[opt].abs()).median())

    curve = budget_curve(c, a, np.r_[np.linspace(0, unconstrained * 2, 21), np.linspace(0, budget, 11)[1:]])

    target = c.assign(recommended_spend=x_opt, expected_incremental_revenue=base[opt]["inc"])
    target = target[target["recommended_spend"] > 0].copy()
    target["action"] = target["segment"].map(dc["actions"]).fillna("Personal follow up")
    target["return_per_pound"] = a["margin"] * target["expected_incremental_revenue"] / target["recommended_spend"]
    target = target.sort_values("expected_incremental_revenue", ascending=False)
    target.index.name = "customer_id"

    ceil = np.array(ceilings)
    ceiling = {"median": float(np.median(ceil)), "low": float(np.quantile(ceil, q)),
               "high": float(np.quantile(ceil, 1 - q)), "share_under_budget": float((ceil < budget).mean())}
    return {"assumptions": a, "budget": budget, "spend_ceiling": ceiling, "allocation": alloc, "plans": base,
            "unconstrained_spend": float(unconstrained), "share_summary": share_summary,
            "strategy_summary": strat_summary, "regret_median": regret, "draws": draws,
            "budget_curve": curve, "target_list": target, "customers": c}


# ---------------------------------------------------------------- outputs

def write(dec: dict, paths) -> dict:
    import matplotlib.pyplot as plt

    from clv import viz

    t, f = paths.tables, paths.figures
    dec["allocation"].to_csv(t / "decide_allocation.csv", index=False)
    dec["share_summary"].to_csv(t / "decide_allocation_sensitivity.csv", index=False)
    dec["strategy_summary"].to_csv(t / "decide_strategies.csv", index=False)
    dec["budget_curve"].to_csv(t / "decide_budget_curve.csv", index=False)
    cols = ["segment", "action", "recommended_spend", "expected_incremental_revenue", "return_per_pound",
            "planning_value", "value_at_risk", "p_alive", "revenue"]
    dec["target_list"][cols].round(4).to_csv(t / "decide_target_list.csv")
    figs = {}

    # 11. allocation by segment, with the range across assumptions
    s = dec["share_summary"].set_index("segment")
    al = dec["allocation"].set_index("segment")
    s = s[(s["high_spend"] > 50) | (s["base_case_spend"] > 50)]
    s = s.sort_values("base_case_spend", ascending=True)
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(s) + 1.9))
    ax.barh(y, s["base_case_spend"], color=viz.SERIES[0], height=0.55, label="Base case")
    ax.errorbar(s["median_spend"], y, xerr=[s["median_spend"] - s["low_spend"], s["high_spend"] - s["median_spend"]],
                fmt="o", color=viz.TEXT, ms=4, lw=1.2, capsize=3, label="Median and 90% range across assumptions")
    right = s[["base_case_spend", "high_spend"]].max(axis=1)
    for yi, seg in enumerate(s.index):
        n = int(al.loc[seg, "targeted"]) if seg in al.index else 0
        roi = al.loc[seg, "roi"] if seg in al.index else np.nan
        txt = (f"{viz.money(al.loc[seg, 'spend'])} on {n:,} accounts, £{1 + roi:,.2f} margin per £1"
               if n else "nothing in the base case")
        ax.text(right[seg] + right.max() * 0.02, yi, txt, va="center", fontsize=8.5, color=viz.TEXT_2)
    ax.set_yticks(y, s.index)
    ax.xaxis.set_major_formatter(viz.gbp)
    ax.set_xlim(0, right.max() * 1.9)
    ax.grid(axis="y", visible=False)
    spent = dec["allocation"]["spend"].sum()
    ax.set_title(f"Past {viz.money(spent)}, the next pound earns back less than a pound", pad=46)
    viz.legend_top(ax, 2)
    viz.subtitle(ax, "Six-month retention spend by RFM segment, placed where the last pound still earns a pound of margin",
                 above_legend=True)
    figs["allocation"] = viz.save(fig, f / "11_allocation.png")

    # 12. strategies compared
    ss = dec["strategy_summary"]
    ss = ss[ss["strategy"] != "Base case plan under these assumptions"].iloc[::-1]
    y = np.arange(len(ss))
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    colours = [viz.SERIES[0] if n.startswith("Optimised") else viz.NEUTRAL for n in ss["strategy"]]
    ax.barh(y, ss["median_profit"], color=colours, height=0.55)
    ax.errorbar(ss["median_profit"], y, xerr=[ss["median_profit"] - ss["low_profit"],
                                             ss["high_profit"] - ss["median_profit"]],
                fmt="none", ecolor=viz.TEXT, lw=1.2, capsize=3)
    for yi, (m, hi) in enumerate(zip(ss["median_profit"], ss["high_profit"])):
        ax.text(max(hi, 0) + abs(ss["high_profit"]).max() * 0.02, yi, viz.money(m), va="center", fontsize=9,
                color=viz.TEXT_2)
    ax.axvline(0, color=viz.TEXT_2, lw=1)
    ax.set_yticks(y, ss["strategy"])
    ax.xaxis.set_major_formatter(viz.gbp)
    ax.grid(axis="y", visible=False)
    ax.set_title("Where the money goes matters more than how much", pad=26)
    viz.subtitle(ax, "Incremental gross profit after spend, median and 90% range across the assumption ranges")
    figs["strategies"] = viz.save(fig, f / "12_strategies.png")

    # 13. how much to spend
    cv = dec["budget_curve"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(cv["budget"], cv["incremental_profit"], color=viz.SERIES[0])
    best = cv.loc[cv["incremental_profit"].idxmax()]
    ax.axvline(dec["budget"], color=viz.NEUTRAL, lw=1, ls=(0, (4, 3)))
    ax.text(dec["budget"], cv["incremental_profit"].max(), "budget on the table ", color=viz.TEXT_2, fontsize=9,
            va="top", ha="right")
    ax.scatter([best["budget"]], [best["incremental_profit"]], color=viz.SERIES[0], s=40, zorder=3,
               edgecolor=viz.SURFACE, linewidth=2)
    unc = dec["unconstrained_spend"]
    lab = (f"profit peaks around £{unc:,.0f}" if unc <= cv["budget"].max()
           else f"still rising at £{cv['budget'].max():,.0f}")
    ax.annotate(lab, (best["budget"], best["incremental_profit"]), xytext=(12, 4),
                textcoords="offset points", ha="left", fontsize=9, color=viz.TEXT_2)
    ax.xaxis.set_major_formatter(viz.gbp)
    ax.yaxis.set_major_formatter(viz.gbp)
    ax.set_xlabel("Six-month retention budget")
    ax.set_ylabel("Incremental gross profit after spend")
    ax.set_title("Past a point, more budget buys less", pad=26)
    viz.subtitle(ax, "Base case assumptions. Every budget is placed in the best way available and spent in full.")
    figs["budget_curve"] = viz.save(fig, f / "13_budget_curve.png")
    return figs
