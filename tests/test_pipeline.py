"""Layer 1 checks, the Power BI tables, and the written text."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clv import diagnose, powerbi, report
from clv.config import ROOT

# Words and patterns that make writing read as generated. The project text
# should sound like a person wrote it, because a person is answering for it.
BANNED = ["delve", "leverag", "robust", "seamless", "comprehensive", "crucial", "furthermore", "moreover",
          "in conclusion", "worth noting", "not just", "showcase", "cutting-edge", "game-changer", "unlock",
          "harness", "tapestry", "pivotal", "meticulous", "realm", "testament", "elevate", "utiliz", "empower",
          "holistic", "streamline", "—", "–"]


def test_every_rfm_cell_has_a_segment():
    segs = {diagnose.segment_for(r, fm) for r in range(1, 6) for fm in range(1, 6)}
    assert segs == set(diagnose.SEGMENT_ORDER)


def test_rfm_covers_every_customer_once(diag):
    r = diag["rfm"]
    assert r.index.is_unique and len(r) == len(diag["customers"])
    assert r[["R", "F", "M"]].isin([1, 2, 3, 4, 5]).all().all()


def test_cohorts_are_consistent(diag):
    c = diag["cohorts"]
    assert c["retention"].between(0, 1).all()
    assert (c.loc[c["age"] == 0, "retention"] == 1).all()
    assert c["cum_revenue_per_customer"].groupby(c["cohort"]).apply(lambda s: s.is_monotonic_increasing).all()


def test_second_purchase_curve_only_goes_up(diag):
    for _, g in diag["second_purchase"].groupby("group"):
        r = g["returned"].dropna()
        assert r.is_monotonic_increasing and r.between(0, 1).all()


def test_kaplan_meier_matches_a_hand_count():
    # 4 customers: return on day 2, day 5, censored at day 3, return on day 5
    km = diagnose.kaplan_meier(np.array([2, 5, 3, 5]), np.array([True, True, False, True]), np.arange(7))
    s = km.set_index("day")["not_returned"]
    assert s[1] == 1.0 and s[2] == pytest.approx(0.75) and s[5] == pytest.approx(0.0)


def test_concentration_numbers(diag):
    c = diag["concentration"]
    assert 0 < c["top_1pct"] < c["top_10pct"] < c["top_20pct"] < 1
    assert 0 < c["gini"] < 1


def test_powerbi_tables(store, diag, pred, dec, tmp_path):
    t = powerbi.build(store["out"], diag, pred, dec)
    files = powerbi.write(t, tmp_path)
    assert (tmp_path / "cve_theme.json").exists() and len(files) == len(t) + 1
    cust = t["dim_customer"]
    assert cust["Customer ID"].is_unique
    assert set(t["fact_invoice"]["Customer ID"]) <= set(cust["Customer ID"])
    assert set(cust["Segment"]) <= set(t["dim_segment"]["Segment"])
    assert t["dim_date"]["Date"].is_unique
    assert set(t["fact_invoice"]["Date"]) <= set(t["dim_date"]["Date"])
    assert cust["Recommended Spend"].sum() == pytest.approx(dec["allocation"]["spend"].sum(), abs=1)


def _check_text(text: str, where: str) -> None:
    low = text.lower()
    for w in BANNED:
        assert w not in low, f"{w!r} in {where}"
    assert not re.search(r"\bnan\b", low), f"nan in {where}"
    assert "{" not in text.replace("{{", "") or "```" in text, f"unformatted placeholder in {where}"


def test_generated_text_reads_like_a_person_wrote_it(store, diag, pred, dec, cfg, paths):
    from clv import clean

    summ = clean.summary(store["raw"], store["out"])
    for p in report.write(summ, store["out"], diag, pred, dec, cfg, paths, synthetic=True):
        _check_text(Path(p).read_text(encoding="utf-8"), p.name)


@pytest.mark.parametrize("name", ["README.md", "REPORT.md", "docs/DECISIONS.md", "docs/POWERBI.md",
                                  "docs/model_card.md", "config.toml"])
def test_committed_text_has_no_dashes_or_filler(name):
    p = ROOT / name
    if not p.exists():
        pytest.skip(f"{name} is written by the pipeline")
    text = p.read_text(encoding="utf-8")
    low = text.lower()
    for w in BANNED:
        assert w not in low, f"{w!r} in {name}"


def test_code_comments_have_no_long_dashes():
    for p in (ROOT / "clv").glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert "—" not in text and "–" not in text, p.name
