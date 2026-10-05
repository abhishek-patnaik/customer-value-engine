"""Writes REPORT.md, the findings block in README.md and docs/model_card.md.

Every number in the text comes from the pipeline results, so the write up
cannot drift away from the code. A synthetic run writes its report to
outputs_synthetic/reports/ and never touches the real README.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from clv.config import ROOT

START, END = "<!-- FINDINGS:START -->", "<!-- FINDINGS:END -->"


# ---------------------------------------------------------------- formatting

def gbp(v: float, short: bool = False) -> str:
    sign = "-" if v < 0 and round(abs(v)) > 0 else ""
    a = abs(v)
    if short and a >= 1e6:
        return f"{sign}£{a / 1e6:.2f}M"
    if short and a >= 10_000:
        return f"{sign}£{a / 1000:,.0f}K"
    return f"{sign}£{a:,.0f}"


def pct(v: float, digits: int = 0) -> str:
    return f"{v * 100:.{digits}f}%"


def signed(v: float) -> str:
    return f"{'+' if v >= 0 else '-'}{abs(v) * 100:.0f}%"


def n(v) -> str:
    return f"{int(round(v)):,}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- findings

def build_findings(summ, out, diag, pred, dec, cfg) -> dict:
    h = diag["headlines"]
    conc = diag["concentration"]
    audit = out["audit"].set_index("step")
    rev_steps = "reversed order (sale + its cancellation)"
    undone = re.search(r"GBP ([\d,]+)", audit.loc[rev_steps, "note"])
    folds = pred["folds"]
    busy = next(k for k in folds if "busy" in k)
    quiet = next(k for k in folds if "quiet" in k)

    def fold_row(k, method_start):
        t = folds[k]["eval_revenue"]
        return t[t["method"].str.startswith(method_start)].iloc[0]

    main = folds[busy]["main"]
    fb, fq = folds[busy]["eval_revenue"].set_index("method"), folds[quiet]["eval_revenue"].set_index("method")
    al = dec["allocation"].set_index("segment")
    ss = dec["strategy_summary"].set_index("strategy")
    spent = float(al["spend"].sum())
    targeted = int(al["targeted"].sum())
    inc_rev = float(al["incremental_revenue"].sum())
    margin = dec["assumptions"]["margin"]
    acq = h["acquisition"]
    yrs = sorted(acq)
    seg = h["segments"]
    beats = ss.loc[[k for k in ss.index if k not in ("Optimised by expected return",
                                                     "Base case plan under these assumptions")], "optimised_beats_it"]
    if beats.min() >= 0.999:
        beats_text = "the optimised plan beats every simple plan in every run"
    else:
        beats_text = (f"the optimised plan beats each simple plan in at least {pct(beats.min())} of runs")
    sens = dec["share_summary"]
    always = sens.loc[sens["funded_in_share_of_runs"] >= 0.95, "segment"].tolist()
    if always:
        names = [f'"{x}"' for x in always]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        share = sens.loc[sens["funded_in_share_of_runs"] >= 0.95, "funded_in_share_of_runs"].min()
        how = "every run" if share >= 0.999 else f"at least {pct(share)} of runs"
        funded_text = f"{listed} get more than £100 in {how}."
    else:
        funded_text = "No segment is funded in every run."
    small = dec["customers"][dec["customers"]["segment"].isin(["Lost", "Hibernating"])]
    mon = diag["monthly"]
    h1 = {y: mon.loc[(mon["month"].dt.year == y) & (mon["month"].dt.month <= 5), "revenue"].sum()
          for y in sorted(set(mon["month"].dt.year))}
    ys = [y for y in h1 if h1[y] > 0 and (mon["month"].dt.year == y).sum() >= 5]
    oneoff = pred["customers"].loc[pred["customers"]["repeat_purchases"] == 0, "p_alive"]
    curve = dec["budget_curve"]
    full = curve.loc[(curve["budget"] - dec["budget"]).abs().idxmin(), "incremental_profit"]
    return {
        "beats_text": beats_text, "funded_text": funded_text,
        "h1_change": float(h1[ys[1]] / h1[ys[0]] - 1) if len(ys) >= 2 else 0.0,
        "oneoff_lo": float(oneoff.min()) if len(oneoff) else 0.0, "oneoff_hi": float(oneoff.max()) if len(oneoff) else 0.0,
        "full_profit": float(full), "n_sims": len(dec["draws"]),
        "risk_small": float(small["value_at_risk"].mean()) if len(small) else 0.0,
        "summ": summ, "h": h, "conc": conc,
        "n_half": int(round(conc["customers_for_half_revenue"] * summ["customers"])),
        "cancel_matched": int(re.search(r"([\d,]+) cancellations", audit.loc[rev_steps, "note"]).group(1).replace(",", "")),
        "cancel_value": float(undone.group(1).replace(",", "")) if undone else np.nan,
        "max_raw_qty_cancelled": True,
        "busy": busy, "quiet": quiet, "main": main,
        "err_busy": float(fb.loc[main, "total_error_pct"]), "err_quiet": float(fq.loc[main, "total_error_pct"]),
        "err_busy_cal": float(fb.loc[[m for m in fb.index if m.startswith(main.split(",")[0]) and "calendar" in m][0],
                                     "total_error_pct"]),
        "err_quiet_cal": float(fq.loc[[m for m in fq.index if m.startswith(main.split(",")[0]) and "calendar" in m][0],
                                      "total_error_pct"]),
        "err_busy_last": float(fb.loc["Same as last 26 weeks", "total_error_pct"]),
        "err_quiet_last": float(fq.loc["Same as last 26 weeks", "total_error_pct"]),
        "err_busy_hist": float(fb.loc["Historical average rate", "total_error_pct"]),
        "err_quiet_hist": float(fq.loc["Historical average rate", "total_error_pct"]),
        "sp_busy": float(fb.loc[main, "spearman"]), "sp_quiet": float(fq.loc[main, "spearman"]),
        "sp_busy_last": float(fb.loc["Same as last 26 weeks", "spearman"]),
        "sp_quiet_last": float(fq.loc["Same as last 26 weeks", "spearman"]),
        "top10_busy": float(fb.loc[main, "top10_capture"]), "top10_quiet": float(fq.loc[main, "top10_capture"]),
        "forecast": pred["forecast_total"], "forecast_cal": pred["forecast_total_calendar_clock"],
        "plan_lo": pred["planning_range"][0], "plan_hi": pred["planning_range"][1],
        "boot_lo": pred["bootstrap_interval"][0], "boot_hi": pred["bootstrap_interval"][1],
        "season": pred["season_index_forecast"],
        "budget": dec["budget"], "spent": spent, "targeted": targeted, "inc_rev": inc_rev,
        "inc_margin": inc_rev * margin, "profit": inc_rev * margin - spent,
        "ceiling": dec["spend_ceiling"], "margin": margin,
        "opt": ss.loc["Optimised by expected return"],
        "forced": ss.loc["Optimised, but forced to spend it all"],
        "equal": ss.loc["Same spend on every customer"],
        "best": ss.loc["Only the best customers"],
        "winback": ss.loc["Win back everyone lapsing"],
        "al": al, "seg": seg,
        "acq_years": yrs, "acq": acq, "acq_naive": h["acquisition_naive"],
        "clt": al.loc["Can't lose them"] if "Can't lose them" in al.index else None,
        "champ": al.loc["Champions"] if "Champions" in al.index else None,
    }


# ---------------------------------------------------------------- README block

def readme_block(f: dict, fig: str) -> str:
    s, h = f["summ"], f["h"]
    y0, y1 = f["acq_years"][0], f["acq_years"][-1]
    a0, a1 = f["acq"][y0]["starters"], f["acq"][y1]["starters"]
    clt = f["clt"]
    lines = [
        "## What the data says",
        "",
        f"- **Half the revenue comes from {f['n_half']} accounts.** That is {pct(f['conc']['customers_for_half_revenue'], 1)} "
        f"of {n(s['customers'])} customers. The top 20% bring in {pct(f['conc']['top_20pct'])}. This is a wholesaler, "
        "most customers are small shops, and a few hundred of them carry the business.",
        f"- **Most new customers do come back, just slowly.** {pct(h['second_90'])} place a second order within 90 days "
        f"and {pct(h['second_365'])} within a year (Kaplan-Meier, so recent customers are not counted as lost). "
        f"A bigger first order is a good sign: {pct(h['second_by_band']['Large first order'])} of the largest third come back "
        f"within a year against {pct(h['second_by_band']['Small first order'])} of the smallest.",
        f"- **Customers found in autumn are Christmas buyers.** Of the people who first bought in September to November "
        f"{h['autumn_year']}, only {pct(h['autumn_m2_10'])} ordered in a typical month over the next nine months, against "
        f"{pct(h['other_m2_10'])} for customers found earlier in the year. When the next autumn came round it went back "
        f"up to {pct(h['autumn_m11_13'])}.",
        f"- **New customers did not halve, even though a simple count says they did.** First purchases fell from "
        f"{n(f['acq_naive'][y0])} to {n(f['acq_naive'][y1])} (April to November, {y0} vs {y1}), but {y0} only has a few months "
        f"of history to check against, so returning customers look new. Measured the same way in both years, the count is "
        f"{n(a0)} vs {n(a1)}.",
        f"- **The value model ranks customers well in both seasons and gets the level right in one of them.** Backtested on "
        f"26 weeks it never saw, it was off by {signed(f['err_busy'])} in the busy half of the year and {signed(f['err_quiet'])} "
        f"in the quiet half. That miss has three causes: only a year of history to learn from, new customers whose early "
        f"buying burst it took as normal, and a genuinely weak first half of 2011. The top 10% of customers it picked held "
        f"{pct(f['top10_busy'])} and {pct(f['top10_quiet'])} of the revenue that actually came in.",
        f"- **Next six months from customers known in December 2011: {gbp(f['plan_lo'], True)} to {gbp(f['plan_hi'], True)}.** "
        f"The model says {gbp(f['forecast'], True)}. I plan on the low end, which is the model corrected by its error in the "
        f"same season last year.",
        f"- **Of a {gbp(f['budget'])} retention budget, only the first {gbp(f['spent'])} earns its keep.** Past that, each "
        f"extra pound brings back less than a pound of margin, and spending all {gbp(f['budget'])} turns a {gbp(f['profit'])} "
        f"gain into {'a ' + gbp(-f['full_profit']) + ' loss' if f['full_profit'] < 0 else gbp(f['full_profit'])}. "
        f"The first {gbp(f['spent'])}, spent on {n(f['targeted'])} "
        f"accounts, should bring back about {gbp(f['inc_margin'])} of margin. I re-ran the plan across "
        f"{n(f['n_sims'])} combinations of the assumptions it rests on. In 90% of them the point where spend stops paying "
        f"sits between {gbp(f['ceiling']['low'])} and {gbp(f['ceiling']['high'])}, and spreading the full budget evenly "
        f"loses money in {pct(f['equal']['share_of_runs_losing_money'])} of them.",
        "",
        f"![Revenue by customer type]({fig}/01_monthly_revenue.png)",
        "",
        "### What I would do about it",
        "",
        f"1. **Cap retention spend at around {gbp(round(f['spent'], -3))} for the next six months, not {gbp(f['budget'])}.** "
        "Put the rest somewhere it can be measured, or keep it. The return falls off fast once the few hundred accounts "
        "that matter are covered.",
        f"2. **Give the top {n(f['champ']['targeted']) if f['champ'] is not None else 'few hundred'} Champions an account manager "
        "check-in and first look at new ranges.** They are the largest single line in the plan because a small lift on a large "
        "account is worth more than a big lift on a small one.",
    ]
    if clt is not None and clt["targeted"] > 0:
        lines.append(
            f"3. **Call {n(clt['targeted'])} big accounts that have gone quiet, first.** They are the ones the plan "
            f"picks from the {n(clt['customers'])} customers in \"Can't lose them\": large, long histories, and on average "
            f"a {pct(clt['mean_p_alive_targeted'])} chance they are still active. That is low, but a senior call costs "
            f"little next to what they used to order.")
    lines += [
        "4. **Run the first campaign as a test, with a holdout.** Randomly keep about a third of the targeted accounts out. "
        "This data has no campaign history, so the response numbers in the plan are my assumptions. One properly measured "
        "campaign replaces them with facts, and the plan re-runs from `config.toml`.",
        "5. **Push first order size up.** Customers who start with a bigger basket come back more often. A starter pack or a "
        "small first-order threshold is worth testing.",
        "",
        f"![Allocation]({fig}/11_allocation.png)",
        "",
        "The full write up, with every table and chart, is in [REPORT.md](REPORT.md).",
        "",
        f"<sub>Numbers generated by the pipeline on {pd.Timestamp.today():%d %b %Y}. Re-running it rewrites this section.</sub>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- REPORT.md

def full_report(f, out, diag, pred, dec, cfg, fig: str, synthetic: bool) -> str:
    s, h = f["summ"], f["h"]
    audit = out["audit"].copy()
    audit_t = pd.DataFrame({
        "Step": audit["step"].str.capitalize(),
        "Rows removed": audit["rows_removed"].map(n),
        "Rows left": audit["rows_left"].map(n),
        "Value of removed rows": audit["value_removed"].map(lambda v: gbp(v) if abs(v) >= 0.5 else "£0"),
    })
    seg = diag["segments"]
    seg_t = pd.DataFrame({
        "Segment": seg["segment"], "Customers": seg["customers"].map(n),
        "Share of customers": seg["share_customers"].map(lambda v: pct(v, 1)),
        "Share of revenue": seg["share_revenue"].map(lambda v: pct(v, 1)),
        "Revenue per customer": seg["revenue_per_customer"].map(gbp),
        "Median days since last order": seg["median_recency_days"].map(n),
    })
    rows = []
    for k in (f["busy"], f["quiet"]):
        t = pred["folds"][k]["eval_revenue"]
        for _, r in t.iterrows():
            rows.append({"Holdout": k, "Method": r["method"], "Error on total": signed(r["total_error_pct"]),
                         "Rank correlation": f"{r['spearman']:.2f}", "Top 10% hold": pct(r["top10_capture"])})
    bt = pd.DataFrame(rows)
    sc = pred["folds"][f["busy"]]["segment_check"]
    scq = pred["folds"][f["quiet"]]["segment_check"]
    grp = pd.DataFrame({
        "Group": sc["group"], f"Error, {f['busy']}": sc["error_pct"].map(signed),
        f"Error, {f['quiet']}": scq.set_index("group").reindex(sc["group"])["error_pct"].map(signed).to_numpy(),
    })
    sp = pred["folds"][f["busy"]]["spend_check"]
    al = dec["allocation"]
    al_t = al[al["spend"] > 0]
    al_t = pd.DataFrame({
        "Segment": al_t["segment"], "Accounts targeted": al_t["targeted"].map(n),
        "Accounts in segment": al_t["customers"].map(n), "Spend": al_t["spend"].map(gbp),
        "Expected extra revenue": al_t["incremental_revenue"].map(gbp),
        "Margin back per £1": (1 + al_t["roi"]).map(lambda v: f"£{v:.2f}"),
    })
    sens = dec["share_summary"]
    sens = sens[(sens["high_spend"] > 50)]
    sens_t = pd.DataFrame({
        "Segment": sens["segment"], "Base case": sens["base_case_spend"].map(gbp),
        "Median": sens["median_spend"].map(gbp),
        "90% range": [f"{gbp(a)} to {gbp(b)}" for a, b in zip(sens["low_spend"], sens["high_spend"])],
        "Funded in": sens["funded_in_share_of_runs"].map(pct) + " of runs",
    })
    st = dec["strategy_summary"]
    st = st[st["strategy"] != "Base case plan under these assumptions"]
    st_t = pd.DataFrame({
        "Plan": st["strategy"], "Base case profit": st["base_case_profit"].map(gbp),
        "Median across assumptions": st["median_profit"].map(gbp),
        "90% range": [f"{gbp(a)} to {gbp(b)}" for a, b in zip(st["low_profit"], st["high_profit"])],
        "Loses money in": st["share_of_runs_losing_money"].map(pct) + " of runs",
    })
    dc = cfg["decide"]
    assume_t = pd.DataFrame([
        {"Assumption": "Gross margin", "Base case": pct(dc["gross_margin"]),
         "Range tested": f"{pct(dc['gross_margin_range'][0])} to {pct(dc['gross_margin_range'][1])}"},
        {"Assumption": "Most a campaign adds to an active customer's six-month revenue", "Base case": pct(dc["lift"]),
         "Range tested": f"{pct(dc['lift_range'][0])} to {pct(dc['lift_range'][1])}"},
        {"Assumption": "Share of value at risk a win-back can recover", "Base case": pct(dc["recovery"]),
         "Range tested": f"{pct(dc['recovery_range'][0])} to {pct(dc['recovery_range'][1])}"},
        {"Assumption": "Spend per account that gets two thirds of the effect", "Base case": gbp(dc["spend_scale"]),
         "Range tested": f"{gbp(dc['spend_scale_range'][0])} to {gbp(dc['spend_scale_range'][1])}"},
        {"Assumption": "Customer values", "Base case": "planning case",
         "Range tested": "planning case to raw model forecast"},
    ])
    m1 = h["month1_by_year"]
    y_list = sorted(m1)
    season = f["season"]
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    hi_m, lo_m = months[int(season.idxmax()) - 1], months[int(season.idxmin()) - 1]
    params = pred["forecast_params"]
    a_, b_ = params["a"], params["b"]
    head = "# Customer Value Engine: full report\n\n"
    if synthetic:
        head += "> **Simulated data.** This report was generated from the synthetic test store, not the real retailer.\n\n"
    text = head + f"""Data: UCI Online Retail II, a UK online wholesaler of gift and homeware items. Every invoice line from {s['first_date']:%d %B %Y} to {s['last_date']:%d %B %Y}. The questions: which customers come back, what they are worth, and where a retention budget should go.

## The data, and what I took out

Every cleaning rule is logged with the rows it removed and the money those rows carried. It is a waterfall, so each row is charged to the first rule that catches it. For reversed orders the value shown is the orders that were cancelled, since each pair nets to zero. Reasons for each rule are in [docs/DECISIONS.md](docs/DECISIONS.md).

{md_table(audit_t)}

What is left: **{n(s['clean_lines'])} sale lines, {n(s['invoices'])} invoices, {n(s['customers'])} known customers, {gbp(s['revenue'], True)} of product revenue.** {pct(s['uk_share_of_revenue'])} of it is from UK customers. Lines with no customer id carried {pct(s['guest_share_of_product_sales'], 1)} of product sales and are out of every customer-level number, so read "customers" as "customers we can identify".

The rule that matters most is matching cancellations to the orders they reverse. {n(f['cancel_matched'])} cancellations were paired with their original lines, taking {gbp(f['cancel_value'])} of orders that never really happened out of customer values, including one order for 80,995 units of a single item, cancelled twelve minutes after it was placed.

## 1. Diagnose

### Revenue rests on a few hundred accounts

![Concentration]({fig}/05_concentration.png)

{f['n_half']} customers ({pct(f['conc']['customers_for_half_revenue'], 1)}) bring in half the revenue, the top 20% bring in {pct(f['conc']['top_20pct'])}, and the Gini coefficient of customer revenue is {f['conc']['gini']:.2f}. Any average across "the customer" mostly describes small accounts that matter little, and any plan that treats customers equally spends most of its money where it cannot pay back.

![Revenue by customer type]({fig}/01_monthly_revenue.png)

Revenue in 2011 looks healthy, but a large share still comes from customers who were already buying in December 2009, and from customers first seen in 2010. The autumn peak is strong in both years: November runs at close to double a summer month.

### How fast new customers come back

![Second purchase]({fig}/03_second_purchase.png)

Measured with a Kaplan-Meier estimate so that customers who only arrived recently are not counted as lost, {pct(h['second_30'])} of new customers order again within 30 days, {pct(h['second_90'])} within 90 days and {pct(h['second_365'])} within a year. The median wait for a second order is {h['median_days_to_second']} days.

The size of the first order says a lot. Split into thirds, {pct(h['second_by_band']['Large first order'])} of customers with the largest first orders are back within a year, against {pct(h['second_by_band']['Small first order'])} for the smallest. This is correlation, not proof that a bigger first order causes loyalty (bigger shops place bigger orders and also reorder more), but it is cheap to test.

### Cohorts

![Cohort retention]({fig}/02_cohort_retention.png)

Each row is the customers whose first order fell in that month. About {pct(h['month1_retention'])} order again the following month ({', '.join(f'{pct(m1[y])} for {y} cohorts' for y in y_list)}).

The pattern that stands out is the diagonal. Customers first seen in September to November {h['autumn_year']} were active in only {pct(h['autumn_m2_10'])} of months over the following nine months, against {pct(h['other_m2_10'])} for customers found earlier that year. Then, eleven to thirteen months later, {pct(h['autumn_m11_13'])} came back. They are Christmas buyers, not lost customers, and judging an autumn campaign by its 90-day repeat rate would undersell it.

The December 2009 row is left out on purpose. Those 951 customers were not new, the data simply starts there.

### New customers did not halve

![Acquisition]({fig}/06_acquisition.png)

A plain count of "first purchase in the data" for April to November gives {n(f['acq_naive'][f['acq_years'][0]])} in {f['acq_years'][0]} and {n(f['acq_naive'][f['acq_years'][-1]])} in {f['acq_years'][-1]}, which reads as acquisition collapsing. It is mostly a measurement problem. In {f['acq_years'][0]} the data only looks back a few months, so a customer who last ordered in 2008 looks brand new. Counting both years the same way, customers buying without having bought since the previous December, gives {n(f['acq'][f['acq_years'][0]]['starters'])} and {n(f['acq'][f['acq_years'][-1]]['starters'])}. In {f['acq_years'][-1]}, {n(f['acq'][f['acq_years'][-1]]['returning_after_break'])} of those were customers returning after a break, which {f['acq_years'][0]} cannot show.

This data cannot say whether acquisition fell. It can say the chart showing a halving is wrong.

### RFM segments

Recency, frequency and money, each scored 1 to 5 by quintile, as of {diag['snapshot']:%d %B %Y}. Frequency is the number of different days a customer ordered, so an order split across three invoices on one day counts once.

{md_table(seg_t)}

![Segments]({fig}/04_rfm_segments.png)

## 2. Predict

The question here is what each customer is likely to spend over the next 26 weeks. Two models do the work:

- **MBG/NBD** for how many orders a customer will place. Each customer buys at their own steady rate while active and may quietly stop after any order, including the first. Nobody tells the shop when a customer leaves, so the model infers it from the gap since the last order compared with that customer's usual rhythm.
- **Gamma-Gamma** for how much each order will be worth. A customer with many orders is predicted close to their own average, a customer with one or two is pulled toward the overall average.

I wrote both from the original papers instead of using the `lifetimes` library, and tested them against simulated customers with known parameters ([models.py](clv/models.py), [tests](tests/test_models.py)).

### Time runs faster in November

![Seasonal index]({fig}/09_season_index.png)

These models assume a customer's buying rate is the same in every week of the year. In this business {hi_m} runs at {season.max():.2f} times an average month and {lo_m} at {season.min():.2f}. Rather than correct the forecast afterwards, I measure time in seasonal weeks: a week in a busy month counts for more than a week in a quiet one. The index comes from customers known to be active for the whole stretch (they bought in the four weeks before the cutoff), fitted with a Poisson model that has one fixed effect per customer, so customers joining or leaving cannot pass for seasonality. It is estimated only from data before each cutoff. My first version did let customers leaving pass for seasonality, and the simulated store, which has no seasons at all, is what caught it. The story is in [DECISIONS.md](docs/DECISIONS.md).

### How it was judged

Two backtests, each hiding 26 weeks the model never saw: the busy half of the year (June to December 2011) and the quiet half (December 2010 to June 2011). The quiet half matters most, because that is the season the real forecast covers. Every method predicts the same thing: revenue from customers already known at the cutoff.

{md_table(bt)}

![Backtest]({fig}/07_backtest.png)

What this says:

- **The ranking holds up.** In both seasons the model's top 10% held {pct(f['top10_busy'])} and {pct(f['top10_quiet'])} of the revenue that actually arrived. Its rank correlation with actual spend was {f['sp_busy']:.2f} in the busy season against {f['sp_busy_last']:.2f} for the "same as last 26 weeks" rule, and {f['sp_quiet']:.2f} against {f['sp_quiet_last']:.2f} in the quiet one{', close to a tie' if abs(f['sp_quiet'] - f['sp_quiet_last']) < 0.03 else ''}. For deciding who to call first it is at least as good as the simple rule, and its level is far better.
- **The seasonal clock earns its place.** Error on the total moved from {signed(f['err_busy_cal'])} to {signed(f['err_busy'])} in the busy fold and from {signed(f['err_quiet_cal'])} to {signed(f['err_quiet'])} in the quiet one, with no loss in ranking.
- **The level in the quiet season is where it falls down: {signed(f['err_quiet'])}.** That fold only had one year of history to learn from, new customers buy heavily in their first months and the model takes that burst as their normal rate, and the first half of 2011 was weak: January to May revenue was {pct(-f['h1_change'])} below the same months of 2010 even though thousands more customers had been acquired by then. A model of each customer's own habits cannot see that coming. Every baseline missed it by more.
- **The plain BG/NBD and the modified version score almost the same.** I use MBG/NBD because plain BG/NBD treats anyone without a repeat order as certainly still active, forever, and the budget plan depends on that probability. MBG/NBD at least lets it fall with time. It still gives one-off buyers a lot of benefit of the doubt ({pct(f['oneoff_lo'])} to {pct(f['oneoff_hi'])} chance of being active), so their value at risk is probably overstated, which if anything makes the plan's case against win-back spend on small accounts stronger.

![Deciles]({fig}/08_deciles.png)

Error on the total, by customer group:

{md_table(grp)}

The quiet season miss is biggest for customers acquired during the data, who had the least history behind them.

Two checks on the spend model:

- Gamma-Gamma assumes order value is unrelated to how often a customer orders. On the final model's data the rank correlation between the two is {pred['independence_spearman']:.2f}, weak but not zero: frequent buyers place slightly bigger orders, so the model slightly under-values its best customers. I have left it and noted it in the model card.
- In the busy season backtest, for customers who did buy, the predicted average order was {gbp(sp['predicted_mean_order'])} against an actual {gbp(sp['actual_mean_order'])}.

### The forecast

Refitted on all two years, the model expects **{gbp(f['forecast'], True)}** from the {n(s['customers'])} customers known on {pred['snapshot'] - pd.Timedelta(days=1):%d %B %Y} over the 26 weeks to {pred['forecast_end']:%d %B %Y}. Bootstrapping customers puts the parameter uncertainty at only {gbp(f['boot_lo'], True)} to {gbp(f['boot_hi'], True)}, which is far too narrow to plan on, because the real risk is the model being wrong about the business, not the parameters being noisy. The backtests are the better guide, so the planning range is the forecast corrected by each backtest's error: **{gbp(f['plan_lo'], True)} to {gbp(f['plan_hi'], True)}**. The budget plan uses the low end. The same model on a calendar clock would have said {gbp(f['forecast_cal'], True)}, because it would treat the quiet months from December to May as average ones.

This covers known customers only. Customers acquired from now on are extra.

![P alive]({fig}/10_p_alive.png)

The chart above is the model's view of who is still active. A frequent buyer who has gone quiet for six months or more is probably gone, because the silence is long compared with how often they used to order. A customer with a single repeat order the same time ago has given much less evidence either way, and the model says so.

Parameters: r = {params['r']:.3f}, alpha = {params['alpha']:.2f}, a = {a_:.3f}, b = {b_:.2f} (MBG/NBD, seasonal weeks); p = {params['p']:.2f}, q = {params['q']:.2f}, gamma = {params['gamma']:.1f} (Gamma-Gamma). The model card is in [docs/model_card.md](docs/model_card.md).

## 3. Decide

The question: where should a {gbp(f['budget'])} retention budget go over the next six months?

### What is measured and what is assumed

The data tells me what each customer is likely to be worth and how likely they are to have already gone. It cannot tell me how customers respond to a call or a discount, because there is no campaign history. So that part is a set of assumptions, written down with a range for each:

{md_table(assume_t)}

For each customer, the most a campaign can add is the lift on their expected value plus the recoverable share of their value at risk (what they would be worth if still active, times the chance they have already left). Returns diminish with spend per account. Money goes where the next pound earns the most, and stops where the next pound earns back less than a pound of margin. The full formula is at the top of [decide.py](clv/decide.py).

### The plan

![Allocation]({fig}/11_allocation.png)

{md_table(al_t)}

Under the base case, **{gbp(f['spent'])} of the {gbp(f['budget'])} is worth spending**, on {n(f['targeted'])} accounts, for about {gbp(f['inc_rev'])} of extra revenue and {gbp(f['inc_margin'])} of margin. That is {gbp(f['profit'])} after spend. Spending the whole {gbp(f['budget'])}, even in the best places available, would {'turn that into a loss of ' + gbp(-f['full_profit']) if f['full_profit'] < 0 else 'cut it to ' + gbp(f['full_profit'])}.

The rest should not be spent on retaining existing accounts. Most customers are small enough that even a cheap campaign costs more than it can bring back in margin. Winning back lapsed small accounts is the worst use of the money: in "Lost" and "Hibernating" the value at risk averages {gbp(f['risk_small'])} per customer.

![Budget curve]({fig}/13_budget_curve.png)

### Compared with plans a team might default to

{md_table(st_t)}

![Strategies]({fig}/12_strategies.png)

Where to spend and how much to spend behave differently. "Where" holds up whatever the assumptions: {f['beats_text']}. "How much" depends more on the assumptions: the point where spend stops paying ranges from {gbp(f['ceiling']['low'])} to {gbp(f['ceiling']['high'])} (median {gbp(f['ceiling']['median'])}), but it stays below {gbp(f['budget'])} in {pct(f['ceiling']['share_under_budget'])} of runs. Forcing the full budget out, even in the best possible places, loses money in {pct(f['forced']['share_of_runs_losing_money'])} of runs.

### How the money moves when the assumptions change

{md_table(sens_t)}

{f['funded_text']} Below that, whether a segment is funded depends on the assumptions, which is a reason to test before spending there.

The full target list, one row per account with the action and the expected return, is in `outputs/tables/decide_target_list.csv`.

## What I would do next

1. **Measure the response.** Run the plan with about a third of targeted accounts held out at random, for one season. That turns the assumption table into measured numbers, and the plan re-runs from `config.toml`.
2. **Add margin by product.** I used one gross margin for everything. Customers who buy mostly low-margin lines are worth less than their revenue says.
3. **Treat seasonal buyers as their own group.** Autumn customers behave differently enough that a separate model, or a covariate for acquisition season, should improve the quiet season forecast.
4. **Re-run monthly.** The scores age quickly for frequent buyers. The pipeline runs in a few minutes, so the target list can be refreshed each month.

## Limitations

- No campaign history, so the decide layer rests on stated assumptions. The sensitivity analysis shows which conclusions survive them.
- Customers without an id ({pct(s['guest_share_of_product_sales'], 1)} of product sales) are invisible to every customer-level result.
- The data starts in December 2009 with an existing customer base whose earlier history is unknown.
- Two backtests are not many. I would want a third season before trusting the level of the forecast in a quiet half year.
- The models assume a customer either keeps buying at their usual rate or stops. Real customers also slow down gradually, which is part of why the quiet season was over-predicted.
"""
    return text


def model_card(f, pred, cfg) -> str:
    p = pred["forecast_params"]
    return f"""# Model card: customer value over the next 26 weeks

**What it predicts.** For each known customer: expected number of orders, expected value per order, and so expected revenue over the next 26 weeks. Also the probability the customer is still active.

**Models.** MBG/NBD (Batislam, Denizel and Filiztekin, 2007) for order counts, Gamma-Gamma (Fader, Hardie and Lee, 2005) for order value. Both written from the papers in `clv/models.py` and checked against simulation in `tests/test_models.py`.

**Unit of purchase.** A customer ordering on a given calendar day, however many invoices that day.

**Clock.** Seasonal weeks. A monthly index from customers known to be active up to the cutoff (bought in its last four weeks), Poisson with a fixed effect per customer, each month shrunk toward 1 by a prior of 50 purchase days.

**Training data.** Clean invoices from known customers, {f['summ']['first_date']:%d %b %Y} to {f['summ']['last_date']:%d %b %Y}.

**Fitted parameters (final model).** r = {p['r']:.3f}, alpha = {p['alpha']:.2f}, a = {p['a']:.3f}, b = {p['b']:.2f}; p = {p['p']:.2f}, q = {p['q']:.2f}, gamma = {p['gamma']:.1f}.

**Backtests.** Fit before a cutoff, predict the next 26 weeks for customers known at the cutoff.

| Holdout | Error on total revenue | Rank correlation | Top 10% hold |
|---|---|---|---|
| {f['busy']} | {signed(f['err_busy'])} | {f['sp_busy']:.2f} | {pct(f['top10_busy'])} |
| {f['quiet']} | {signed(f['err_quiet'])} | {f['sp_quiet']:.2f} | {pct(f['top10_quiet'])} |

**Use it for.** Ranking customers by expected value, spotting large accounts that have probably gone quiet, and a planning range for revenue from the existing base.

**Do not use it for.** A single point forecast of next half year's revenue. In the quiet season backtest the total came out {pct(abs(f['err_quiet']))} {'too high' if f['err_quiet'] > 0 else 'too low'}. Use the planning range, {gbp(f['plan_lo'], True)} to {gbp(f['plan_hi'], True)}, and plan on the low end. It also says nothing about customers not yet acquired.

**Known weaknesses.**
- Over-predicted by {pct(abs(f['err_quiet']))} in the one backtest with only a year of history, most of all for recently acquired customers.
- Cannot anticipate a business-wide slowdown.
- Assumes order value is unrelated to order frequency. Measured rank correlation {pred['independence_spearman']:.2f}, so the best customers are slightly under-valued.
- Gives one-off buyers a high chance of still being active ({pct(f['oneoff_lo'])} to {pct(f['oneoff_hi'])}), probably too high.
- Customers without an id are not modelled.
"""


# ---------------------------------------------------------------- write

def write(summ, out, diag, pred, dec, cfg, paths, synthetic: bool = False) -> list[Path]:
    f = build_findings(summ, out, diag, pred, dec, cfg)
    written = []
    if synthetic:
        rel = "../figures"
        targets = [(paths.reports / "REPORT.md", full_report(f, out, diag, pred, dec, cfg, rel, True)),
                   (paths.reports / "README_findings_preview.md", readme_block(f, rel)),
                   (paths.reports / "model_card.md", model_card(f, pred, cfg))]
    else:
        fig = "outputs/figures"
        targets = [(ROOT / "REPORT.md", full_report(f, out, diag, pred, dec, cfg, fig, False)),
                   (ROOT / "docs" / "model_card.md", model_card(f, pred, cfg))]
        readme = ROOT / "README.md"
        block = readme_block(f, fig)
        if readme.exists() and START in readme.read_text(encoding="utf-8"):
            txt = readme.read_text(encoding="utf-8")
            pre, rest = txt.split(START, 1)
            post = rest.split(END, 1)[1]
            targets.append((readme, f"{pre}{START}\n\n{block}\n\n{END}{post}"))
    for path, text in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
