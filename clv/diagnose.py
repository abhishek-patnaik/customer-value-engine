"""Layer 1: who comes back, how fast, and who the revenue really depends on.

Everything here is descriptive and computed straight from the clean invoices.
No model, no assumptions beyond the ones logged in config.toml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# RFM segment map. R is recency score 1 to 5 (5 = most recent), FM is the
# rounded mean of the frequency and monetary scores. Every one of the 25 cells
# belongs to exactly one segment.
SEGMENT_ORDER = [
    "Champions", "Loyal", "Potential loyalists", "New", "Promising",
    "Need attention", "About to sleep", "At risk", "Can't lose them", "Hibernating", "Lost",
]


def segment_for(r: int, fm: int) -> str:
    if r == 5:
        return "Champions" if fm >= 4 else ("Potential loyalists" if fm >= 2 else "New")
    if r == 4:
        return "Loyal" if fm >= 4 else ("Potential loyalists" if fm >= 2 else "Promising")
    if r == 3:
        return "Loyal" if fm >= 4 else ("Need attention" if fm >= 2 else "About to sleep")
    if fm == 5:
        return "Can't lose them"
    if fm >= 3:
        return "At risk"
    return "Hibernating" if r == 2 else "Lost"


def snapshot_date(invoices: pd.DataFrame) -> pd.Timestamp:
    """The day after the last invoice. Recency is measured from here."""
    return invoices["invoice_date"].max().normalize() + pd.Timedelta(days=1)


def purchase_days(invoices: pd.DataFrame) -> pd.DataFrame:
    """One row per customer per calendar day they bought anything.

    Several invoices on the same day are one purchase occasion. In this data
    that is usually an order split across invoices, not a second decision.
    """
    d = invoices.assign(day=invoices["invoice_date"].dt.normalize())
    return (d.groupby(["customer_id", "day"], as_index=False)
            .agg(revenue=("revenue", "sum"), n_invoices=("invoice", "size")))


def customers(invoices: pd.DataFrame, returns: pd.DataFrame, snapshot: pd.Timestamp,
              base_cutoff: pd.Timestamp) -> pd.DataFrame:
    days = purchase_days(invoices)
    g = days.groupby("customer_id")
    out = pd.DataFrame({
        "first_purchase": g["day"].min(),
        "last_purchase": g["day"].max(),
        "purchase_days": g["day"].size(),
        "invoices": invoices.groupby("customer_id").size(),
        "revenue": g["revenue"].sum(),
    })
    ret = returns.groupby("customer_id")["revenue"].sum() if len(returns) else pd.Series(dtype=float)
    out["returns"] = (-ret).reindex(out.index).fillna(0.0)
    out["net_revenue"] = out["revenue"] - out["returns"]
    by_country = invoices.groupby(["customer_id", "country"])["revenue"].sum().reset_index()
    top = by_country.sort_values("revenue", ascending=False).drop_duplicates("customer_id")
    out["country"] = top.set_index("customer_id")["country"].reindex(out.index)
    out["recency_days"] = (snapshot - out["last_purchase"]).dt.days
    out["tenure_days"] = (snapshot - out["first_purchase"]).dt.days
    out["avg_order_value"] = out["revenue"] / out["purchase_days"]
    out["cohort"] = out["first_purchase"].dt.to_period("M")
    # Customers first seen in the opening month were mostly buying before the
    # data starts. They are the existing base, not new customers.
    out["is_existing_base"] = out["first_purchase"] < base_cutoff
    return out.sort_index()


def monthly(invoices: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    d = invoices.assign(month=invoices["invoice_date"].dt.to_period("M"))
    d["cohort"] = d["customer_id"].map(cust["cohort"])
    d["kind"] = np.where(d["cohort"] == d["month"], "new", "returning")
    d.loc[d["customer_id"].map(cust["is_existing_base"]), "kind"] = "existing base"
    m = d.pivot_table(index="month", columns="kind", values="revenue", aggfunc="sum", fill_value=0.0)
    active = d.groupby("month")["customer_id"].nunique()
    new = d[d["kind"] == "new"].groupby("month")["customer_id"].nunique()
    out = pd.DataFrame({
        "revenue": d.groupby("month")["revenue"].sum(),
        "active_customers": active,
        "new_customers": new.reindex(active.index).fillna(0).astype(int),
        "revenue_new": m.get("new", 0.0),
        "revenue_returning": m.get("returning", 0.0),
        "revenue_existing_base": m.get("existing base", 0.0),
        "invoices": d.groupby("month").size(),
    })
    out["avg_order_value"] = out["revenue"] / out["invoices"]
    return out.reset_index()


def cohorts(invoices: pd.DataFrame, cust: pd.DataFrame, last_full_month: pd.Period) -> pd.DataFrame:
    """Long table: one row per cohort per month of age.

    Ages that land in a month the data does not fully cover are left out, so
    the last cohorts do not look like they collapsed.
    """
    d = invoices.assign(month=invoices["invoice_date"].dt.to_period("M"))
    d["cohort"] = d["customer_id"].map(cust["cohort"])
    d = d[d["month"] <= last_full_month]
    d["age"] = (d["month"] - d["cohort"]).apply(lambda x: x.n)
    size = cust[cust["cohort"] <= last_full_month].groupby("cohort").size().rename("cohort_size")
    cells = (d.groupby(["cohort", "age"]).agg(active=("customer_id", "nunique"), revenue=("revenue", "sum"))
             .reset_index())
    # make sure ages with nobody active still show up as zero
    full = []
    for c, n in size.items():
        ages = range(0, (last_full_month - c).n + 1)
        full.append(pd.DataFrame({"cohort": c, "age": list(ages), "cohort_size": n}))
    grid = pd.concat(full, ignore_index=True)
    t = grid.merge(cells, on=["cohort", "age"], how="left").fillna({"active": 0, "revenue": 0.0})
    t["active"] = t["active"].astype(int)
    t["retention"] = t["active"] / t["cohort_size"]
    first_rev = t[t["age"] == 0].set_index("cohort")["revenue"]
    t["revenue_retention"] = t["revenue"] / t["cohort"].map(first_rev)
    t["cum_revenue_per_customer"] = t.groupby("cohort")["revenue"].cumsum() / t["cohort_size"]
    t["is_existing_base"] = t["cohort"] == cust.loc[cust["is_existing_base"], "cohort"].min()
    return t


def kaplan_meier(durations: np.ndarray, observed: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """Share of customers who have NOT yet made a second purchase by each day.

    Customers whose first purchase is recent have not had the chance to come
    back yet. Counting them as "never returned" would understate repeat
    buying, so they are treated as censored, which is what Kaplan-Meier is for.
    """
    order = np.argsort(durations)
    t, e = durations[order], observed[order]
    times = np.unique(t[e])
    surv, s = [], 1.0
    n = len(t)
    at_risk_idx = 0
    for u in times:
        while at_risk_idx < n and t[at_risk_idx] < u:
            at_risk_idx += 1
        at_risk = n - at_risk_idx
        events = int(((t == u) & e).sum())
        s *= 1 - events / at_risk
        surv.append((u, s, at_risk))
    curve = pd.DataFrame(surv, columns=["day", "not_returned", "at_risk"])
    out = pd.DataFrame({"day": grid})
    out = pd.merge_asof(out, curve[["day", "not_returned", "at_risk"]], on="day")
    out["not_returned"] = out["not_returned"].fillna(1.0)
    out["returned"] = 1 - out["not_returned"]
    # do not report the curve past the point where almost nobody is left to observe
    last_ok = t[max(0, n - 50)] if n > 50 else t[-1]
    out.loc[out["day"] > last_ok, ["returned", "not_returned"]] = np.nan
    return out


def second_purchase(invoices: pd.DataFrame, cust: pd.DataFrame, snapshot: pd.Timestamp,
                    horizon_days: int = 365) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = purchase_days(invoices).sort_values(["customer_id", "day"])
    days["n"] = days.groupby("customer_id").cumcount()
    second = days[days["n"] == 1].set_index("customer_id")["day"]
    new = cust[~cust["is_existing_base"]].copy()
    new["second"] = second.reindex(new.index)
    new["observed"] = new["second"].notna()
    end = new["second"].where(new["observed"], snapshot)
    new["duration"] = (end - new["first_purchase"]).dt.days
    grid = np.arange(0, horizon_days + 1)
    overall = kaplan_meier(new["duration"].to_numpy(), new["observed"].to_numpy(), grid).assign(group="All new customers")
    curves = [overall]
    # split by the value of the first order: does a big first basket mean a returning customer?
    first_val = days[days["n"] == 0].set_index("customer_id")["revenue"].reindex(new.index)
    q = first_val.quantile([1 / 3, 2 / 3]).to_numpy()
    new["first_order_band"] = np.select([first_val <= q[0], first_val <= q[1]],
                                        ["Small first order", "Medium first order"], "Large first order")
    for band in ["Small first order", "Medium first order", "Large first order"]:
        s = new[new["first_order_band"] == band]
        curves.append(kaplan_meier(s["duration"].to_numpy(), s["observed"].to_numpy(), grid).assign(group=band))
    bands = new.groupby("first_order_band").agg(customers=("observed", "size"),
                                                 first_order_low=("first_purchase", "size"))
    bands["first_order_low"] = [first_val[new["first_order_band"] == b].min() for b in bands.index]
    bands["first_order_high"] = [first_val[new["first_order_band"] == b].max() for b in bands.index]
    return pd.concat(curves, ignore_index=True), bands.reset_index()


def acquisition_like_for_like(invoices: pd.DataFrame, first_month: int = 4, last_month: int = 11) -> pd.DataFrame:
    """Customers who buy in a month without having bought since December.

    The raw count of "first purchase in the data" is unfair across years: in
    2010 the data only looks back a few months, so a customer who last bought
    in 2008 looks brand new. Here both years get the same lookback (since the
    previous December), and the 2011 starters are split into first ever
    buyers and customers returning after a break, which 2010 cannot do.
    """
    d = invoices.assign(m=invoices["invoice_date"].dt.to_period("M"))
    active = {m: set(g) for m, g in d.groupby("m")["customer_id"]}
    first = d.groupby("customer_id")["m"].min()
    rows = []
    years = sorted({p.year for p in active if p.month == last_month})
    for y in years:
        start = pd.Period(f"{y - 1}-12", "M")
        for k in range(first_month, last_month + 1):
            m = pd.Period(f"{y}-{k:02d}", "M")
            if m not in active:
                continue
            seen = set().union(*[active[p] for p in active if start <= p < m])
            starters = active[m] - seen
            first_ever = sum(1 for c in starters if first[c] == m)
            rows.append({"year": y, "month": m, "starters": len(starters), "first_ever": first_ever,
                         "returning_after_break": len(starters) - first_ever})
    return pd.DataFrame(rows)


def _score(values: pd.Series, ascending: bool) -> pd.Series:
    """Quintile score 1 to 5. Ranking first breaks the many ties in frequency."""
    r = values.rank(method="first", ascending=ascending)
    return pd.qcut(r, 5, labels=[1, 2, 3, 4, 5]).astype(int)


def rfm(cust: pd.DataFrame) -> pd.DataFrame:
    t = cust[["recency_days", "purchase_days", "revenue", "net_revenue", "first_purchase", "country"]].copy()
    t["R"] = _score(t["recency_days"], ascending=False)
    t["F"] = _score(t["purchase_days"], ascending=True)
    t["M"] = _score(t["revenue"], ascending=True)
    t["FM"] = np.floor((t["F"] + t["M"]) / 2 + 0.5).astype(int)
    t["rfm_code"] = t["R"].astype(str) + t["F"].astype(str) + t["M"].astype(str)
    t["segment"] = [segment_for(r, fm) for r, fm in zip(t["R"], t["FM"])]
    return t


def segment_summary(r: pd.DataFrame) -> pd.DataFrame:
    s = (r.groupby("segment")
         .agg(customers=("R", "size"), revenue=("revenue", "sum"),
              median_recency_days=("recency_days", "median"), median_purchase_days=("purchase_days", "median"),
              median_revenue=("revenue", "median"))
         .reindex(SEGMENT_ORDER).dropna(subset=["customers"]))
    s["customers"] = s["customers"].astype(int)
    s["share_customers"] = s["customers"] / s["customers"].sum()
    s["share_revenue"] = s["revenue"] / s["revenue"].sum()
    s["revenue_per_customer"] = s["revenue"] / s["customers"]
    return s.reset_index()


def concentration(cust: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    v = np.sort(cust["revenue"].to_numpy())[::-1]
    cum = np.cumsum(v) / v.sum()
    share = np.arange(1, len(v) + 1) / len(v)
    curve = pd.DataFrame({"share_customers": np.r_[0, share], "share_revenue": np.r_[0, cum]})
    asc = np.sort(v)
    n = len(asc)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ asc / (n * asc.sum()))
    tops = {f"top_{p}pct": float(cum[max(0, int(np.ceil(len(v) * p / 100)) - 1)]) for p in (1, 10, 20)}
    customers_for_half = float(np.searchsorted(cum, 0.5) + 1) / len(v)
    return curve, {**tops, "gini": gini, "customers_for_half_revenue": customers_for_half}


def headlines(d: dict) -> dict:
    """The numbers the write up quotes, computed in one place."""
    c = d["cohorts"]
    n = c[~c["is_existing_base"]]

    def rate(ages, months=None, years=None):
        x = n[n["age"].isin(list(ages))]
        if months is not None:
            x = x[x["cohort"].dt.month.isin(months)]
        if years is not None:
            x = x[x["cohort"].dt.year.isin(years)]
        return float(x["active"].sum() / x["cohort_size"].sum()) if len(x) else float("nan")

    km = d["second_purchase"]
    allc = km[km["group"] == "All new customers"].set_index("day")["returned"]
    band = {g: float(km[km["group"] == g].set_index("day")["returned"].dropna().iloc[-1])
            for g in ("Small first order", "Medium first order", "Large first order")}
    years = sorted({p.year for p in n["cohort"]})
    first_year = years[0]
    acq = d["acquisition"].groupby("year")[["starters", "first_ever", "returning_after_break"]].sum()
    naive = d["monthly"].assign(year=d["monthly"]["month"].dt.year)
    naive = naive[naive["month"].dt.month.between(4, 11)].groupby("year")["new_customers"].sum()
    seg = d["segments"].set_index("segment")
    return {
        "month1_retention": rate([1]),
        "month1_by_year": {y: rate([1], years=[y]) for y in years if not np.isnan(rate([1], years=[y]))},
        "autumn_m2_10": rate(range(2, 11), months=[9, 10, 11], years=[first_year]),
        "other_m2_10": rate(range(2, 11), months=list(range(1, 9)), years=[first_year]),
        "autumn_m11_13": rate(range(11, 14), months=[9, 10, 11], years=[first_year]),
        "autumn_year": first_year,
        "second_30": float(allc.get(30, np.nan)), "second_90": float(allc.get(90, np.nan)),
        "second_365": float(allc.get(365, np.nan)),
        "median_days_to_second": int(allc[allc >= 0.5].index.min()) if (allc >= 0.5).any() else None,
        "second_by_band": band,
        "acquisition": acq.to_dict("index"), "acquisition_naive": naive.to_dict(),
        "segments": seg.to_dict("index"),
    }


def run(clean_out: dict, cfg: dict) -> dict:
    inv, ret = clean_out["invoices"], clean_out["returns"]
    snap = snapshot_date(inv)
    first_month = inv["invoice_date"].min().to_period("M")
    base_cutoff = (first_month + 1).to_timestamp()
    last_day = inv["invoice_date"].max()
    month_end = (last_day.to_period("M") + 1).to_timestamp() - pd.Timedelta(days=1)
    last_full = last_day.to_period("M") - (1 if last_day.normalize() < month_end.normalize() else 0)

    cust = customers(inv, ret, snap, base_cutoff)
    mon = monthly(inv, cust)
    coh = cohorts(inv, cust, last_full)
    km, bands = second_purchase(inv, cust, snap, cfg["diagnose"]["second_purchase_horizon_days"])
    r = rfm(cust)
    seg = segment_summary(r)
    lorenz, conc = concentration(cust)
    acq = acquisition_like_for_like(inv)
    out = {"acquisition": acq, "snapshot": snap, "last_full_month": last_full, "customers": cust, "monthly": mon,
            "cohorts": coh, "second_purchase": km, "first_order_bands": bands, "rfm": r,
            "segments": seg, "lorenz": lorenz, "concentration": conc}
    out["headlines"] = headlines(out)
    return out


# ---------------------------------------------------------------- outputs

def write(d: dict, paths) -> dict:
    """Tables and charts for layer 1. Returns the figure paths."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from clv import viz

    t = paths.tables
    d["monthly"].assign(month=d["monthly"]["month"].astype(str)).to_csv(t / "diagnose_monthly.csv", index=False)
    coh = d["cohorts"].assign(cohort=d["cohorts"]["cohort"].astype(str))
    coh.to_csv(t / "diagnose_cohorts.csv", index=False)
    d["second_purchase"].to_csv(t / "diagnose_second_purchase.csv", index=False)
    d["first_order_bands"].to_csv(t / "diagnose_first_order_bands.csv", index=False)
    d["segments"].to_csv(t / "diagnose_rfm_segments.csv", index=False)
    d["acquisition"].assign(month=d["acquisition"]["month"].astype(str)).to_csv(t / "diagnose_acquisition.csv", index=False)
    figs = {}
    f = paths.figures

    # 1. monthly revenue by who it came from
    m = d["monthly"].copy()
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    parts = [("revenue_existing_base", "Existing base (bought in Dec 2009)", viz.SERIES[0]),
             ("revenue_returning", "First seen since Jan 2010, buying again", viz.SERIES[2]),
             ("revenue_new", "First purchase that month", viz.SERIES[1])]
    bottom = np.zeros(len(m))
    for col, label, colour in parts:
        ax.bar(x, m[col], bottom=bottom, color=colour, width=0.78, label=label, edgecolor=viz.SURFACE, linewidth=1)
        bottom += m[col].to_numpy()
    ax.set_xticks(x[::2], [p.strftime("%b %y") for p in m["month"]][::2])
    ax.yaxis.set_major_formatter(viz.gbp)
    ax.set_xlim(-0.6, len(m) - 0.4)
    ax.grid(axis="x", visible=False)
    ax.set_title("Revenue holds up because customers found in 2010 came back", pad=40)
    viz.legend_top(ax, 3)
    viz.subtitle(ax, "Monthly product revenue from known customers, split by when the customer first appeared",
                 above_legend=True)
    viz.footnote(fig, "December 2011 covers 9 days only.")
    figs["monthly"] = viz.save(fig, f / "01_monthly_revenue.png")

    # 2. cohort retention heatmap (new cohorts only)
    c = d["cohorts"]
    c = c[~c["is_existing_base"] & (c["age"] >= 1)]
    piv = c.pivot(index="cohort", columns="age", values="retention")
    piv = piv.loc[piv.notna().any(axis=1)]
    fig, ax = plt.subplots(figsize=(12, 7.2))
    vmax = float(np.nanquantile(piv.to_numpy(), 0.98))
    im = ax.imshow(piv.to_numpy(), cmap=viz.BLUES, vmin=0, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.2,
                    color="white" if v > vmax * 0.55 else viz.TEXT)
    sizes = d["cohorts"].drop_duplicates("cohort").set_index("cohort")["cohort_size"]
    ax.set_yticks(range(piv.shape[0]), [f"{p.strftime('%b %Y')}  ({sizes[p]:,})" for p in piv.index], fontsize=8.5)
    ax.set_xticks(range(piv.shape[1]), [str(a) for a in piv.columns], fontsize=8.5)
    ax.set_xlabel("Months since first purchase")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Share of each month's new customers who bought again, by month", pad=26)
    viz.subtitle(ax, "Numbers are percent of the cohort. Cohort size in brackets. Customers already buying in Dec 2009 are left out.")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, format=viz.pct)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=viz.TEXT_2, labelsize=8)
    figs["cohorts"] = viz.save(fig, f / "02_cohort_retention.png")

    # 3. time to second purchase
    km = d["second_purchase"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colours = {"Small first order": viz.SERIES[0], "Medium first order": viz.SERIES[2],
               "Large first order": viz.SERIES[1]}
    b = d["first_order_bands"].set_index("first_order_band")
    for g, colour in colours.items():
        k = km[km["group"] == g]
        label = f"{g} (£{b.loc[g, 'first_order_low']:,.0f} to £{b.loc[g, 'first_order_high']:,.0f})"
        if g == "Large first order":
            label = f"{g} (over £{b.loc[g, 'first_order_low']:,.0f})"
        if g == "Small first order":
            label = f"{g} (under £{b.loc[g, 'first_order_high']:,.0f})"
        ax.plot(k["day"], k["returned"], color=colour, label=label)
        last = k.dropna().iloc[-1]
        ax.annotate(f"{last['returned'] * 100:.0f}%", (last["day"], last["returned"]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=viz.TEXT_2, fontsize=9)
    ax.yaxis.set_major_formatter(viz.pct)
    ax.set_ylim(0, 0.9)
    ax.set_xlim(0, 390)
    ax.set_xticks([0, 30, 90, 180, 270, 365])
    ax.set_xlabel("Days since first purchase")
    ax.set_title("Bigger first orders come back sooner and more often", pad=26)
    viz.subtitle(ax, "Share of new customers who have placed a second order, Kaplan-Meier estimate")
    ax.legend(loc="upper left", ncols=1)
    figs["second_purchase"] = viz.save(fig, f / "03_second_purchase.png")

    # 4. RFM segments: share of customers vs share of revenue
    s = d["segments"].iloc[::-1]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(9.5, 6))
    h = 0.38
    ax.barh(y + h / 2, s["share_customers"], height=h, color=viz.SERIES[0], label="Share of customers")
    ax.barh(y - h / 2, s["share_revenue"], height=h, color=viz.SERIES[1], label="Share of revenue")
    for yi, (sc, sr) in enumerate(zip(s["share_customers"], s["share_revenue"])):
        ax.text(sr + 0.006, yi - h / 2, f"{sr * 100:.0f}%" if sr >= 0.01 else "<1%", va="center",
                fontsize=8.5, color=viz.TEXT_2)
    ax.set_yticks(y, s["segment"])
    ax.xaxis.set_major_formatter(viz.pct)
    ax.grid(axis="y", visible=False)
    ax.set_title("Champions: 15% of customers, half the revenue", pad=46)
    viz.legend_top(ax, 2)
    viz.subtitle(ax, "RFM segments as of 10 Dec 2011, full two years of purchases", above_legend=True)
    figs["segments"] = viz.save(fig, f / "04_rfm_segments.png")

    # 5. concentration
    lor = d["lorenz"]
    conc = d["concentration"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(lor["share_customers"], lor["share_revenue"], color=viz.SERIES[0])
    ax.plot([0, 1], [0, 1], color=viz.NEUTRAL, lw=1, ls=(0, (4, 3)))
    for xs, ys, txt in [(conc["customers_for_half_revenue"], 0.5,
                         f"{conc['customers_for_half_revenue'] * 100:.1f}% of customers\nbring in half the revenue"),
                        (0.2, conc["top_20pct"], f"top 20%: {conc['top_20pct'] * 100:.0f}% of revenue")]:
        ax.scatter([xs], [ys], s=40, color=viz.SERIES[0], zorder=3, edgecolor=viz.SURFACE, linewidth=2)
        ax.annotate(txt, (xs, ys), xytext=(14, -14), textcoords="offset points", fontsize=9, color=viz.TEXT_2,
                    va="top")
    ax.xaxis.set_major_formatter(viz.pct)
    ax.yaxis.set_major_formatter(viz.pct)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Customers, biggest spenders first")
    ax.set_ylabel("Share of revenue")
    n_half = int(round(conc["customers_for_half_revenue"] * len(d["customers"])))
    ax.set_title(f"{n_half} accounts bring in half the revenue", pad=26)
    viz.subtitle(ax, f"Cumulative share of two year revenue. Gini {conc['gini']:.2f}")
    figs["concentration"] = viz.save(fig, f / "05_concentration.png")

    # 6. acquisition like for like
    a = d["acquisition"].groupby("year")[["first_ever", "returning_after_break", "starters"]].sum()
    naive = d["monthly"].copy()
    naive["year"] = naive["month"].dt.year
    naive = naive[naive["month"].dt.month.between(4, 11)].groupby("year")["new_customers"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    years = list(a.index)
    xs = np.arange(len(years))
    known = np.array([yr > years[0] for yr in years])
    ax.bar(xs[~known], a["starters"][~known], width=0.5, color=viz.NEUTRAL, edgecolor=viz.SURFACE, linewidth=1)
    ax.bar(xs[known], a["first_ever"][known], width=0.5, color=viz.SERIES[1], edgecolor=viz.SURFACE, linewidth=1)
    ax.bar(xs[known], a["returning_after_break"][known], bottom=a["first_ever"][known], width=0.5,
           color=viz.SERIES[0], edgecolor=viz.SURFACE, linewidth=1)
    for xi, yr in zip(xs, years):
        ax.text(xi, a.loc[yr, "starters"] + 40, f"{a.loc[yr, 'starters']:,}", ha="center", fontsize=10,
                color=viz.TEXT)
    ax.text(0, a.iloc[0]["starters"] / 2, "no history before Dec 2009,\nso new and returning\ncannot be told apart",
            ha="center", va="center", color=viz.TEXT, fontsize=8.5)
    ax.set_xticks(xs, [f"Apr to Nov {y}" for y in years])
    ax.set_xlim(-0.7, len(years) - 0.3)
    ax.set_ylim(0, a["starters"].max() * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("New customers did not halve", pad=40)
    ax.legend(handles=[Patch(color=viz.SERIES[1], label="First purchase in the data"),
                       Patch(color=viz.SERIES[0], label="Back after a break"),
                       Patch(color=viz.NEUTRAL, label="Mixed")],
              loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncols=3, borderaxespad=0)
    viz.subtitle(ax, "Customers buying without having bought since the previous December, same lookback both years",
                 above_legend=True)
    viz.footnote(fig, f"A naive count of first purchases shows {naive.iloc[0]:,} vs {naive.iloc[-1]:,}, "
                      "which looks like a halving but compares a 4 to 11 month lookback with a 16 to 23 month one.")
    figs["acquisition"] = viz.save(fig, f / "06_acquisition.png")
    return figs
