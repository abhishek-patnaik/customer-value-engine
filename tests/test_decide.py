"""The budget optimiser on cases where the right answer is obvious."""

import numpy as np
import pytest

from clv import decide


def test_never_spends_past_the_point_of_no_return():
    M = np.array([10_000.0, 1_000.0, 50.0])
    x = decide.optimal_spend(M, k=25, margin=0.35, budget=None)
    # the last pound on every funded account returns exactly one pound of margin
    funded = x > 0
    marginal = 0.35 * M[funded] / 25 * np.exp(-x[funded] / 25)
    assert np.allclose(marginal, 1.0)
    # an account whose whole potential is worth less than the first pound gets nothing
    assert x[2] == 0


def test_respects_the_budget_and_equalises_returns():
    M = np.array([10_000.0, 5_000.0, 2_000.0])
    x = decide.optimal_spend(M, k=25, margin=0.35, budget=100)
    assert x.sum() == pytest.approx(100, rel=1e-6)
    marginal = 0.35 * M / 25 * np.exp(-x / 25)
    assert np.allclose(marginal[x > 0], marginal[x > 0][0], rtol=1e-4)
    assert x[0] > x[1] > x[2]


def test_forced_spending_uses_the_whole_budget():
    M = np.array([100.0, 100.0])
    x = decide.optimal_spend(M, k=25, margin=0.35, budget=1_000, force=True)
    assert x.sum() == pytest.approx(1_000, rel=1e-6)
    assert x[0] == pytest.approx(x[1])


def test_optimised_plan_beats_every_alternative(dec):
    s = dec["strategy_summary"].set_index("strategy")
    for name, beats in s["optimised_beats_it"].dropna().items():
        assert beats >= 0.95, name
    assert s.loc["Optimised by expected return", "share_of_runs_losing_money"] == 0


def test_allocation_adds_up(dec):
    al = dec["allocation"]
    assert al["spend"].sum() == pytest.approx(dec["target_list"]["recommended_spend"].sum())
    assert al["spend"].sum() <= dec["budget"] + 1e-6
    assert al["targeted"].sum() == len(dec["target_list"])
    assert (dec["target_list"]["return_per_pound"] >= 1 - 1e-6).all()


def test_value_at_risk_is_zero_for_customers_surely_active(dec):
    c = dec["customers"]
    sure = c[c["p_alive"] > 0.999]
    assert (sure["value_at_risk"] <= sure["planning_value"] * 0.0011).all()
