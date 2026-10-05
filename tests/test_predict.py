"""The predict layer on the simulated store, which has no seasonality and
follows the model's assumptions, so the right answers are known."""

import numpy as np
import pandas as pd
import pytest

from clv import predict
from clv.diagnose import purchase_days


def test_seasonal_index_is_flat_when_there_are_no_seasons(store, pred):
    days = purchase_days(store["out"]["invoices"])
    end = days["day"].max() + pd.Timedelta(days=1)
    for back in (0, 26, 52):
        idx = predict.seasonal_index(days, end - pd.Timedelta(weeks=back))
        assert idx.between(0.75, 1.3).all(), idx.round(2).to_dict()


def test_seasonal_index_finds_a_planted_season():
    rng = np.random.default_rng(0)
    rows = []
    start = pd.Timestamp("2010-01-01")
    for c in range(800):
        rate = rng.gamma(2, 0.02)  # purchases per day
        for d in pd.date_range(start, "2011-12-31"):
            boost = 2.0 if d.month == 11 else 1.0
            if rng.random() < rate * boost:
                rows.append((str(c), d))
    days = pd.DataFrame(rows, columns=["customer_id", "day"])
    idx = predict.seasonal_index(days, pd.Timestamp("2012-01-01"))
    assert idx[11] / idx.drop(11).mean() == pytest.approx(2.0, rel=0.15)


def test_clock_runs_faster_in_busy_months():
    idx = pd.Series([0.5] * 10 + [2.0, 1.0], index=range(1, 13))
    clock = predict.Clock(pd.Timestamp("2010-01-01"), pd.Timestamp("2011-01-01"), idx)
    jan = clock([pd.Timestamp("2010-02-01")])[0] - clock([pd.Timestamp("2010-01-01")])[0]
    nov = clock([pd.Timestamp("2010-12-01")])[0] - clock([pd.Timestamp("2010-11-01")])[0]
    assert nov / jan == pytest.approx(4 * 30 / 31, rel=1e-6)
    flat = predict.Clock(pd.Timestamp("2010-01-01"), pd.Timestamp("2011-01-01"))
    assert flat([pd.Timestamp("2010-01-08")])[0] == pytest.approx(1.0)


def test_model_is_close_when_its_assumptions_hold(pred):
    for fold in pred["folds"].values():
        main = fold["eval_revenue"].iloc[0]
        assert abs(main["total_error_pct"]) < 0.12, (fold["main"], main["total_error_pct"])


def test_model_beats_the_simple_baselines(pred):
    for fold in pred["folds"].values():
        t = fold["eval_revenue"].set_index("method")
        main = fold["main"]
        for base in ("Same as last 26 weeks", "Historical average rate"):
            assert abs(t.loc[main, "total_error_pct"]) < abs(t.loc[base, "total_error_pct"])
            assert t.loc[main, "spearman"] > t.loc[base, "spearman"]


def test_deciles_are_ordered_and_complete(pred):
    for fold in pred["folds"].values():
        d = fold["deciles"]
        assert d["predicted_per_customer"].is_monotonic_decreasing
        assert d["actual_per_customer"].iloc[0] > d["actual_per_customer"].iloc[-1]
        assert d["share_of_actual"].sum() == pytest.approx(1.0)


def test_forecast_outputs_are_sane(pred):
    c = pred["customers"]
    assert c["p_alive"].between(0, 1).all()
    assert (c["expected_value"] >= 0).all()
    assert c["value_rank"].is_monotonic_increasing
    lo, hi = pred["planning_range"]
    assert lo <= pred["forecast_total"] / (1 + max(pred["backtest_errors"].values())) + 1
    assert lo < hi
    m = pred["monthly_forecast"]
    assert m["forecast_revenue"].sum() == pytest.approx(pred["forecast_total"], rel=1e-6)


def test_holdout_never_leaks_into_the_fit(store, cfg, diag):
    """Changing what happens after the cutoff must not change the prediction."""
    days = purchase_days(store["out"]["invoices"])
    snap = diag["snapshot"]
    cal_end = snap - pd.Timedelta(weeks=26)
    start, far = days["day"].min(), snap + pd.Timedelta(weeks=40)
    base = diag["customers"].index[diag["customers"]["is_existing_base"]]
    a = predict.backtest(days, base, start, far, cal_end, snap, 26, "mbgnbd")
    doubled = pd.concat([days, days[days["day"] >= cal_end].assign(day=lambda x: x["day"] + pd.Timedelta(hours=1))])
    b = predict.backtest(doubled, base, start, far, cal_end, snap, 26, "mbgnbd")
    assert np.allclose(a["scores"]["expected_value"], b["scores"]["expected_value"])
