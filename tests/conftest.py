"""Shared fixtures. Everything runs on the simulated store, so the tests need
no download and know the right answers in advance."""

import pytest

from clv import clean, decide, diagnose, predict
from clv.config import Paths, load_config
from clv.synthetic import build


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["predict"]["bootstrap_rounds"] = 5
    c["decide"]["simulations"] = 100
    return c


@pytest.fixture(scope="session")
def planted():
    raw, truth = build(seed=11, n_customers=1500, n_guest_orders=400)
    return raw, truth


@pytest.fixture(scope="session")
def cleaned(planted, cfg):
    raw, _ = planted
    return clean.clean(raw, cfg)


@pytest.fixture(scope="session")
def store(cfg):
    """A full size simulated store, run through the whole pipeline once."""
    raw, truth = build()
    out = clean.clean(raw, cfg)
    return {"raw": raw, "truth": truth, "out": out}


@pytest.fixture(scope="session")
def paths(tmp_path_factory):
    return Paths(tmp_path_factory.mktemp("run")).ensure()


@pytest.fixture(scope="session")
def diag(store, cfg):
    return diagnose.run(store["out"], cfg)


@pytest.fixture(scope="session")
def pred(store, diag, cfg):
    return predict.run(store["out"], diag, cfg)


@pytest.fixture(scope="session")
def dec(pred, diag, cfg):
    return decide.run(pred, diag, cfg)
