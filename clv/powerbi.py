"""Tables for the Power BI report, shaped as a small star schema.

Facts and dimensions are written as plain CSV so the report needs no Python
at refresh time. Column names are written for people, because they show up
as field names in Power BI. The build steps and every measure are in
docs/POWERBI.md.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from clv.diagnose import SEGMENT_ORDER
from clv.viz import NEUTRAL, SERIES

SEGMENT_NOTES = {
    "Champions": "Bought recently, often and a lot",
    "Loyal": "Strong history, recent enough",
    "Potential loyalists": "Recent, building a habit",
    "New": "Recent first order, nothing else yet",
    "Promising": "Fairly recent, one small order",
    "Need attention": "Middling on everything, drifting",
    "About to sleep": "One small order a few months ago",
    "At risk": "Used to buy well, quiet for months",
    "Can't lose them": "Among the biggest accounts, gone quiet",
    "Hibernating": "Small and quiet for a long time",
    "Lost": "One small order, long ago",
}


def build(clean_out: dict, diag: dict, pred: dict, dec: dict) -> dict[str, pd.DataFrame]:
    inv = clean_out["invoices"]
    cust = diag["customers"]
    rfm = diag["rfm"]
    pc = pred["customers"]
    target = dec["target_list"]

    dim_customer = pd.DataFrame({
        "Customer ID": cust.index,
        "Country": cust["country"].to_numpy(),
        "First Purchase": cust["first_purchase"].dt.date.to_numpy(),
        "Last Purchase": cust["last_purchase"].dt.date.to_numpy(),
        "Cohort Month": cust["cohort"].dt.to_timestamp().dt.date.to_numpy(),
        "Customer Type": np.where(cust["is_existing_base"], "Existing base", "Acquired in the data"),
        "Purchase Days": cust["purchase_days"].to_numpy(),
        "Revenue 2 Years": cust["revenue"].round(2).to_numpy(),
        "Returns 2 Years": cust["returns"].round(2).to_numpy(),
        "Days Since Last Purchase": cust["recency_days"].to_numpy(),
        "R Score": rfm["R"].reindex(cust.index).to_numpy(),
        "F Score": rfm["F"].reindex(cust.index).to_numpy(),
        "M Score": rfm["M"].reindex(cust.index).to_numpy(),
        "RFM Code": rfm["rfm_code"].reindex(cust.index).to_numpy(),
        "Segment": rfm["segment"].reindex(cust.index).to_numpy(),
    }).set_index("Customer ID")
    dim_customer["P Alive"] = pc["p_alive"].reindex(dim_customer.index).round(4)
    dim_customer["Expected Purchases 26w"] = pc["expected_purchases"].reindex(dim_customer.index).round(3)
    dim_customer["Expected Order Value"] = pc["expected_spend"].reindex(dim_customer.index).round(2)
    dim_customer["Model Value 26w"] = pc["expected_value"].reindex(dim_customer.index).round(2)
    dim_customer["Planning Value 26w"] = pc["planning_value"].reindex(dim_customer.index).round(2)
    dim_customer["Value At Risk 26w"] = dec["customers"]["value_at_risk"].reindex(dim_customer.index).round(2)
    dim_customer["Value Rank"] = pc["value_rank"].reindex(dim_customer.index)
    dim_customer["Value Decile"] = np.ceil(dim_customer["Value Rank"] / len(dim_customer) * 10).astype(int)
    dim_customer["Recommended Spend"] = target["recommended_spend"].reindex(dim_customer.index).fillna(0).round(2)
    dim_customer["Expected Extra Revenue"] = (target["expected_incremental_revenue"].reindex(dim_customer.index)
                                              .fillna(0).round(2))
    dim_customer["Recommended Action"] = target["action"].reindex(dim_customer.index).fillna("No paid action")
    dim_customer["In Target List"] = np.where(dim_customer["Recommended Spend"] > 0, "Yes", "No")
    dim_customer = dim_customer.reset_index()

    fact_invoice = pd.DataFrame({
        "Invoice": inv["invoice"], "Customer ID": inv["customer_id"],
        "Date": inv["invoice_date"].dt.date, "Invoice Time": inv["invoice_date"],
        "Revenue": inv["revenue"].round(2), "Lines": inv["n_lines"], "Units": inv["n_units"],
        "Country": inv["country"],
    })

    ret = clean_out["returns"]
    fact_return = pd.DataFrame({
        "Invoice": ret["invoice"], "Customer ID": ret["customer_id"], "Date": ret["invoice_date"].dt.date,
        "Stock Code": ret["stock_code"], "Return Value": (-ret["revenue"]).round(2),
    })

    start = inv["invoice_date"].min().normalize()
    end = pred["forecast_end"]
    days = pd.date_range(start, end, freq="D")
    dim_date = pd.DataFrame({
        "Date": days.date, "Year": days.year, "Quarter": "Q" + days.quarter.astype(str),
        "Month Number": days.month, "Month": days.strftime("%b"), "Month Start": days.to_period("M").to_timestamp().date,
        "Year Month": days.strftime("%Y-%m"), "Weekday": days.strftime("%a"), "Weekday Number": days.dayofweek + 1,
        "Is Forecast Period": np.where(days >= pred["snapshot"], "Forecast", "Actual"),
    })

    coh = diag["cohorts"]
    fact_cohort = pd.DataFrame({
        "Cohort Month": coh["cohort"].dt.to_timestamp().dt.date, "Months Since First Purchase": coh["age"],
        "Cohort Size": coh["cohort_size"], "Active Customers": coh["active"], "Revenue": coh["revenue"].round(2),
        "Retention": coh["retention"].round(4), "Revenue Retention": coh["revenue_retention"].round(4),
        "Cumulative Revenue Per Customer": coh["cum_revenue_per_customer"].round(2),
        "Cohort Type": np.where(coh["is_existing_base"], "Existing base", "New customers"),
    })

    seg = diag["segments"].set_index("segment")
    al = dec["allocation"].set_index("segment")
    sens = dec["share_summary"].set_index("segment")
    dim_segment = pd.DataFrame({"Segment": [s for s in SEGMENT_ORDER if s in seg.index]})
    dim_segment["Segment Order"] = np.arange(1, len(dim_segment) + 1)
    dim_segment["Description"] = dim_segment["Segment"].map(SEGMENT_NOTES)
    dim_segment["Action"] = dim_segment["Segment"].map(lambda s: dec_action(dec, s))
    dim_segment["Planned Spend"] = dim_segment["Segment"].map(al["spend"]).fillna(0).round(2)
    dim_segment["Accounts Targeted"] = dim_segment["Segment"].map(al["targeted"]).fillna(0).astype(int)
    dim_segment["Expected Extra Revenue"] = dim_segment["Segment"].map(al["incremental_revenue"]).fillna(0).round(2)
    dim_segment["Spend Low"] = dim_segment["Segment"].map(sens["low_spend"]).fillna(0).round(2)
    dim_segment["Spend High"] = dim_segment["Segment"].map(sens["high_spend"]).fillna(0).round(2)

    mf = pred["monthly_forecast"]
    # a stub of a few days at the end reads like a collapse on a monthly chart
    mf = mf[(pd.to_datetime(mf["period_end"]) - pd.to_datetime(mf["period_start"])).dt.days >= 14]
    hist = inv.assign(m=inv["invoice_date"].dt.to_period("M")).groupby("m")["revenue"].sum()
    hist = hist[hist.index < pred["snapshot"].to_period("M")]
    fact_month = pd.concat([
        pd.DataFrame({"Month Start": hist.index.to_timestamp().date, "Actual Revenue": hist.round(2).to_numpy()}),
        pd.DataFrame({"Month Start": pd.to_datetime(mf["period_start"]).dt.to_period("M").dt.to_timestamp().dt.date,
                      "Model Forecast": mf["forecast_revenue"].round(2).to_numpy(),
                      "Planning Forecast": mf["planning_revenue"].round(2).to_numpy()}),
    ], ignore_index=True).groupby("Month Start", as_index=False).sum(min_count=1)
    fact_month["Note"] = np.where(fact_month["Model Forecast"].notna(),
                                  "Forecast from customers known on 9 Dec 2011. December 2011 is part actual, part forecast", "")

    curve = dec["budget_curve"].rename(columns={"budget": "Budget", "incremental_profit": "Incremental Profit",
                                                "incremental_revenue": "Incremental Revenue"})
    curve = curve[["Budget", "Incremental Revenue", "Incremental Profit"]].round(2)
    strat = dec["strategy_summary"].rename(columns={
        "strategy": "Strategy", "base_case_profit": "Base Case Profit", "median_profit": "Median Profit",
        "low_profit": "Profit Low", "high_profit": "Profit High",
        "share_of_runs_losing_money": "Share Of Runs Losing Money"})
    strat = strat[strat["Strategy"] != "Base case plan under these assumptions"]
    strat = strat[["Strategy", "Base Case Profit", "Median Profit", "Profit Low", "Profit High",
                   "Share Of Runs Losing Money"]].round(2)

    return {"dim_customer": dim_customer, "fact_invoice": fact_invoice, "fact_return": fact_return,
            "dim_date": dim_date, "fact_cohort": fact_cohort, "dim_segment": dim_segment,
            "fact_month": fact_month, "plan_budget_curve": curve, "plan_strategies": strat}


def dec_action(dec: dict, segment: str) -> str:
    t = dec["target_list"]
    hit = t.loc[t["segment"] == segment, "action"]
    return hit.iloc[0] if len(hit) else "No paid action, newsletter only"


def theme() -> dict:
    """Power BI theme with the same colours as the charts in the repo."""
    return {
        "name": "Customer Value Engine",
        "dataColors": SERIES + ["#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "background": "#FCFCFB", "foreground": "#0B0B0B", "tableAccent": SERIES[0],
        "good": "#1BAF7A", "neutral": NEUTRAL, "bad": "#E34948",
        "maximum": "#0D366B", "center": "#6DA7EC", "minimum": "#F4F8FD",
        "textClasses": {
            "title": {"fontFace": "Segoe UI Semibold", "fontSize": 13, "color": "#0B0B0B"},
            "label": {"fontFace": "Segoe UI", "fontSize": 10, "color": "#52514E"},
            "callout": {"fontFace": "Segoe UI Semibold", "fontSize": 26, "color": "#0B0B0B"},
        },
    }


def write(tables: dict[str, pd.DataFrame], folder) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, df in tables.items():
        p = folder / f"{name}.csv"
        df.to_csv(p, index=False)
        out.append(p)
    p = folder / "cve_theme.json"
    p.write_text(json.dumps(theme(), indent=2))
    out.append(p)
    return out
