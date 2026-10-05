"""The cleaning has to remove exactly what was planted in the simulated data,
and the small rules have to behave on hand made cases."""

import pandas as pd
import pytest

from clv import clean
from clv.data import _as_code


def removed(out, step):
    a = out["audit"].set_index("step")
    return int(a.loc[step, "rows_removed"])


def test_every_planted_problem_is_caught_by_its_own_rule(planted, cleaned):
    _, t = planted
    assert removed(cleaned, "sheet overlap") == t["overlap_rows"]
    assert removed(cleaned, "exact duplicate line") == t["duplicates"]
    assert removed(cleaned, "bad debt adjustment") == t["bad_debt"]
    assert removed(cleaned, "non-product code") == t["postage_lines"]
    assert removed(cleaned, "zero or negative price") == t["zero_price_lines"]
    assert removed(cleaned, "stock adjustment") == t["stock_adjustments"]
    assert removed(cleaned, "missing customer id") == t["guest_lines"]
    assert removed(cleaned, "reversed order (sale + its cancellation)") == 2 * t["full_cancellations"]
    assert removed(cleaned, "unmatched cancellation (kept as return)") == t["partial_returns"]


def test_clean_lines_are_the_true_orders_minus_the_cancelled_ones(planted, cleaned):
    _, t = planted
    assert len(cleaned["lines"]) == t["clean_known_lines"] - t["full_cancellations"]
    assert cleaned["lines"]["revenue"].sum() == pytest.approx(t["clean_value"] - t["full_cancel_value"])
    assert -cleaned["returns"]["revenue"].sum() == pytest.approx(t["partial_returns_value"])


def test_waterfall_adds_up(cleaned):
    a = cleaned["audit"]
    assert (a["rows_left"].shift() - a["rows_removed"]).iloc[1:].tolist() == a["rows_left"].iloc[1:].tolist()
    assert a["rows_left"].iloc[-1] == len(cleaned["lines"])


def test_integrity_checks_are_clean(cleaned):
    assert all(v == 0 for v in cleaned["checks"].values()), cleaned["checks"]


def test_invoices_add_up_to_lines(cleaned):
    assert cleaned["invoices"]["revenue"].sum() == pytest.approx(cleaned["lines"]["revenue"].sum())
    assert cleaned["invoices"]["invoice"].is_unique
    assert (cleaned["invoices"]["revenue"] > 0).all()


def test_no_customer_is_invented(planted, cleaned):
    _, t = planted
    assert set(cleaned["lines"]["customer_id"]) <= set(t["customers"]["customer_id"])


# hand made cases for the matching rule

def _lines(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "stock_code", "price", "quantity", "invoice_date"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


def test_cancellation_takes_the_most_recent_earlier_sale():
    sales = _lines([("1", "A", 2.0, 5, "2010-01-01"), ("1", "A", 2.0, 5, "2010-03-01"),
                    ("1", "A", 2.0, 5, "2010-06-01")])
    cancels = _lines([("1", "A", 2.0, -5, "2010-04-01")]).set_axis([10])
    s, c = clean.match_cancellations(sales, cancels, 365)
    assert s == [1] and c == [10]


def test_cancellation_never_matches_a_later_sale_or_another_customer_or_price():
    sales = _lines([("1", "A", 2.0, 5, "2010-06-01"), ("2", "A", 2.0, 5, "2010-01-01"),
                    ("1", "A", 2.5, 5, "2010-01-01")])
    cancels = _lines([("1", "A", 2.0, -5, "2010-04-01")]).set_axis([10])
    assert clean.match_cancellations(sales, cancels, 365) == ([], [])


def test_a_sale_is_only_reversed_once():
    sales = _lines([("1", "A", 2.0, 5, "2010-01-01")])
    cancels = _lines([("1", "A", 2.0, -5, "2010-02-01"), ("1", "A", 2.0, -5, "2010-02-02")]).set_axis([10, 11])
    s, c = clean.match_cancellations(sales, cancels, 365)
    assert s == [0] and c == [10]


def test_window_is_respected():
    sales = _lines([("1", "A", 2.0, 5, "2010-01-01")])
    cancels = _lines([("1", "A", 2.0, -5, "2011-06-01")]).set_axis([10])
    assert clean.match_cancellations(sales, cancels, 365) == ([], [])
    assert clean.match_cancellations(sales, cancels, 600) == ([0], [10])


def test_partial_return_is_not_matched():
    sales = _lines([("1", "A", 2.0, 10, "2010-01-01")])
    cancels = _lines([("1", "A", 2.0, -4, "2010-02-01")]).set_axis([10])
    assert clean.match_cancellations(sales, cancels, 365) == ([], [])


def test_non_product_codes(cfg):
    codes = pd.Series(["POST", "post", "85123A", "M", "gift_0001_20", "BANK CHARGES", "22423", None], dtype="string")
    assert clean.is_non_product(codes, cfg["clean"]).tolist() == [True, True, False, True, True, True, False, False]


def test_codes_from_excel_are_normalised():
    s = pd.Series([85123, 85123.0, "85123A", " 22423 ", None, 13085.0], dtype="object")
    assert _as_code(s).tolist() == ["85123", "85123", "85123A", "22423", pd.NA, "13085"]
