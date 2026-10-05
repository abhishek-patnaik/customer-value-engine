"""Turns the raw lines into tables a customer analysis can trust.

The rules run in a fixed order and every one of them is logged in an audit
table: how many rows it removed and how much money those rows carried. The
audit is a waterfall, so a row is charged to the first rule that removes it.

Outputs
    lines     clean sale lines, one per product per invoice, known customers only
    returns   cancellation lines that could not be matched to the order they undo
    invoices  one row per clean invoice (the unit the value models work on)
    audit     the waterfall
    checks    integrity checks that should all come back clean
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

RAW_COLS = ["invoice", "stock_code", "description", "quantity", "invoice_date", "price", "customer_id", "country"]


class Waterfall:
    def __init__(self, df: pd.DataFrame):
        self.rows = [{"step": "raw rows", "rows_removed": 0, "rows_left": len(df),
                      "value_removed": 0.0, "note": "both sheets stacked"}]

    def log(self, step: str, removed: pd.DataFrame, left: int, note: str, value: float | None = None) -> None:
        if value is None:
            value = float((removed["quantity"] * removed["price"]).sum()) if len(removed) else 0.0
        self.rows.append({"step": step, "rows_removed": len(removed), "rows_left": left,
                          "value_removed": round(value, 2), "note": note})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def is_non_product(codes: pd.Series, cfg_clean: dict) -> pd.Series:
    upper = codes.str.upper()
    exact = upper.isin([c.upper() for c in cfg_clean["non_product_codes"]])
    prefix = pd.Series(False, index=codes.index)
    for p in cfg_clean["non_product_prefixes"]:
        prefix |= codes.str.lower().str.startswith(p.lower())
    return (exact | prefix).fillna(False).astype(bool)


def match_cancellations(sales: pd.DataFrame, cancels: pd.DataFrame, window_days: int):
    """Pair each cancellation line with the order line it reverses.

    A cancellation reverses a sale when the customer, product, unit price and
    quantity all agree and the sale happened before it (within the window).
    When several sales qualify, the most recent unused one is taken. Returns
    the index labels of matched sales and matched cancellations.
    """
    key = ["customer_id", "stock_code", "price"]
    pool: dict[tuple, list] = defaultdict(list)
    s = sales[key + ["quantity", "invoice_date"]].sort_values("invoice_date")
    for idx, cust, code, price, qty, when in s.itertuples(name=None):
        pool[(cust, code, round(price, 4), int(qty))].append((when, idx))

    window = pd.Timedelta(days=window_days)
    used_sales, used_cancels = set(), []
    c = cancels[key + ["quantity", "invoice_date"]].sort_values("invoice_date")
    for idx, cust, code, price, qty, when in c.itertuples(name=None):
        candidates = pool.get((cust, code, round(price, 4), -int(qty)))
        if not candidates:
            continue
        for j in range(len(candidates) - 1, -1, -1):
            sale_when, sale_idx = candidates[j]
            if sale_when > when or sale_idx in used_sales:
                continue
            if when - sale_when > window:
                break
            used_sales.add(sale_idx)
            used_cancels.append(idx)
            break
    return list(used_sales), used_cancels


def clean(raw: pd.DataFrame, cfg: dict) -> dict:
    cc = cfg["clean"]
    df = raw.copy()
    wf = Waterfall(df)

    # 1. The two sheets overlap. An invoice that appears in an earlier sheet
    #    is the same invoice, so later copies go.
    first_sheet = df.groupby("invoice", sort=False)["sheet"].transform("first")
    drop = df["sheet"] != first_sheet
    wf.log("sheet overlap", df[drop], int((~drop).sum()), "invoice already present in the earlier sheet")
    df = df[~drop]

    # 2. Exact duplicate lines: same invoice, product, quantity, price,
    #    customer and second. Most likely recorded twice.
    drop = df.duplicated(subset=RAW_COLS, keep="first")
    wf.log("exact duplicate line", df[drop], int((~drop).sum()), "identical in every column")
    df = df[~drop]

    inv = df["invoice"].str.upper()

    # 3. Bad debt write offs are accounting entries, not purchases.
    drop = inv.str.startswith("A")
    wf.log("bad debt adjustment", df[drop], int((~drop).sum()), "invoice starts with A")
    df, inv = df[~drop], inv[~drop]

    # 4. Postage, fees, discounts, manual and test lines are not products.
    drop = is_non_product(df["stock_code"], cc)
    wf.log("non-product code", df[drop], int((~drop).sum()), "postage, fees, discounts, manual, samples, tests")
    df, inv = df[~drop], inv[~drop]

    is_cancel = inv.str.startswith("C")

    # 5. Lines with no price (free samples, internal moves).
    drop = (df["price"] <= 0) & ~is_cancel
    wf.log("zero or negative price", df[drop], int((~drop).sum()), "sale lines with price <= 0")
    df, inv, is_cancel = df[~drop], inv[~drop], is_cancel[~drop]

    # 6. Negative quantity on a normal invoice is a stock correction
    #    (damaged, lost, found), not something a customer did.
    drop = (df["quantity"] <= 0) & ~is_cancel
    wf.log("stock adjustment", df[drop], int((~drop).sum()), "quantity <= 0 on a non cancellation invoice")
    df, inv, is_cancel = df[~drop], inv[~drop], is_cancel[~drop]

    # 7. No customer id: the money is real, but it cannot be tied to anyone,
    #    so it is out of every customer level analysis. Its share is reported.
    drop = df["customer_id"].isna()
    sale_value = df["quantity"] * df["price"] * ~is_cancel
    meta = {"guest_sales_value": float(sale_value[drop].sum()),
            "known_sales_value": float(sale_value[~drop].sum())}
    wf.log("missing customer id", df[drop], int((~drop).sum()), "guest or unrecorded customer")
    df, is_cancel = df[~drop], is_cancel[~drop]

    # 8. A cancellation that exactly reverses an earlier line removes both.
    sales, cancels = df[~is_cancel], df[is_cancel]
    matched_sales, matched_cancels = match_cancellations(sales, cancels, cc["match_window_days"])
    gone = df.loc[matched_sales + matched_cancels]
    left = len(df) - len(gone)
    undone = df.loc[matched_sales]
    undone_value = float((undone["quantity"] * undone["price"]).sum())
    # The pair nets to zero, so the value shown is the orders that were undone.
    wf.log("reversed order (sale + its cancellation)", gone, left,
           f"{len(matched_cancels):,} cancellations matched, GBP {undone_value:,.0f} of orders undone",
           value=undone_value)
    df = df.drop(index=gone.index)

    # 9. What is left of the cancellations are partial returns or returns of
    #    orders placed before the data starts. Kept aside, not in invoice values.
    is_cancel = df["invoice"].str.upper().str.startswith("C")
    returns = df[is_cancel].copy()
    wf.log("unmatched cancellation (kept as return)", returns, int((~is_cancel).sum()),
           "partial returns or orders from before Dec 2009")
    lines = df[~is_cancel].copy()

    lines["revenue"] = lines["quantity"] * lines["price"]
    returns["revenue"] = returns["quantity"] * returns["price"]
    lines = lines.sort_values(["invoice_date", "invoice"]).reset_index(drop=True)
    returns = returns.sort_values("invoice_date").reset_index(drop=True)

    invoices = (
        lines.groupby("invoice", as_index=False)
        .agg(customer_id=("customer_id", "first"), country=("country", "first"),
             invoice_date=("invoice_date", "min"), revenue=("revenue", "sum"),
             n_lines=("stock_code", "size"), n_units=("quantity", "sum"))
        .sort_values(["invoice_date", "invoice"]).reset_index(drop=True)
    )

    per_invoice = lines.groupby("invoice").agg(n_cust=("customer_id", "nunique"),
                                               n_country=("country", "nunique"),
                                               span=("invoice_date", lambda s: (s.max() - s.min()).total_seconds()))
    checks = {
        "invoices_with_several_customers": int((per_invoice["n_cust"] > 1).sum()),
        "invoices_with_several_countries": int((per_invoice["n_country"] > 1).sum()),
        "invoices_spanning_over_an_hour": int((per_invoice["span"] > 3600).sum()),
        "customers_in_several_countries": int((lines.groupby("customer_id")["country"].nunique() > 1).sum()),
        "non_positive_lines_left": int(((lines["quantity"] <= 0) | (lines["price"] <= 0)).sum()),
    }
    return {"lines": lines, "returns": returns, "invoices": invoices, "audit": wf.frame(),
            "checks": checks, "meta": meta}


def summary(raw: pd.DataFrame, out: dict) -> dict:
    """Headline numbers about the data the rest of the project stands on."""
    lines, inv, ret, meta = out["lines"], out["invoices"], out["returns"], out["meta"]
    sale_value = float(lines["revenue"].sum())
    guest, known = meta["guest_sales_value"], meta["known_sales_value"]
    return {
        "raw_rows": int(len(raw)),
        "clean_lines": int(len(lines)),
        "customers": int(lines["customer_id"].nunique()),
        "invoices": int(len(inv)),
        "products": int(lines["stock_code"].nunique()),
        "countries": int(lines["country"].nunique()),
        "first_date": lines["invoice_date"].min(),
        "last_date": lines["invoice_date"].max(),
        "revenue": sale_value,
        "returns_value": float(-ret["revenue"].sum()),
        "returns_share": float(-ret["revenue"].sum() / sale_value) if sale_value else np.nan,
        "guest_share_of_product_sales": guest / (guest + known) if guest + known else np.nan,
        "uk_share_of_revenue": float(lines.loc[lines["country"] == "United Kingdom", "revenue"].sum() / sale_value)
        if sale_value else np.nan,
    }
