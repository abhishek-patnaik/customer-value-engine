"""The value models against simulated customers with known parameters.

There is no closed form to check BG/NBD or MBG/NBD predictions against, so
they are checked against a Monte Carlo of the process the model describes.
"""

import numpy as np
import pytest

from clv.models import BGNBD, MBGNBD, GammaGamma

R, ALPHA, A, B = 0.75, 6.0, 0.9, 3.5


def simulate(n, rng, modified, horizon=26.0):
    """x, t_x, T for n customers, plus their purchases in the next `horizon` weeks."""
    T = rng.uniform(5, 100, n)
    lam = rng.gamma(R, 1 / ALPHA, n)
    p = rng.beta(A, B, n)
    X, TX, H = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        if modified and rng.random() < p[i]:
            continue
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T[i] + horizon:
                break
            if t <= T[i]:
                X[i] += 1
                TX[i] = t
            else:
                H[i] += 1
            if rng.random() < p[i]:
                break
    return X, TX, T, H


@pytest.fixture(scope="module", params=[(BGNBD, False), (MBGNBD, True)], ids=["bgnbd", "mbgnbd"])
def fitted(request):
    model_cls, modified = request.param
    rng = np.random.default_rng(3)
    X, TX, T, H = simulate(6000, rng, modified)
    return model_cls().fit(X, TX, T), X, TX, T, H


def test_recovers_parameters(fitted):
    m = fitted[0]
    assert m.converged
    assert m.r == pytest.approx(R, rel=0.15)
    assert m.alpha == pytest.approx(ALPHA, rel=0.2)
    assert m.a == pytest.approx(A, rel=0.3)
    assert m.b == pytest.approx(B, rel=0.3)


def test_conditional_forecast_matches_simulation(fitted):
    m, X, TX, T, H = fitted
    pred = m.expected_purchases(26, X, TX, T)
    assert pred.sum() == pytest.approx(H.sum(), rel=0.05)
    for lo, hi in [(0, 0), (1, 2), (3, 6), (7, 1000)]:
        k = (X >= lo) & (X <= hi)
        assert pred[k].mean() == pytest.approx(H[k].mean(), rel=0.15, abs=0.05)


def test_p_alive_behaves(fitted):
    m = fitted[0]
    # same history, longer silence -> less likely to be active
    p_recent = m.p_alive([5], [50], [52])[0]
    p_quiet = m.p_alive([5], [10], [52])[0]
    assert 0 < p_quiet < p_recent <= 1
    # more purchases with the same silence -> stronger evidence of leaving
    assert m.p_alive([20], [10], [52])[0] < p_quiet


def test_plain_bgnbd_thinks_one_off_buyers_are_always_active():
    m = BGNBD(r=R, alpha=ALPHA, a=A, b=B)
    assert m.p_alive([0, 0], [0, 0], [1, 200]).tolist() == [1.0, 1.0]
    mm = MBGNBD(r=R, alpha=ALPHA, a=A, b=B)
    p = mm.p_alive([0, 0], [0, 0], [1, 200])
    assert p[1] < p[0] < 1


@pytest.mark.parametrize("model_cls,modified", [(BGNBD, False), (MBGNBD, True)])
def test_new_customer_expectation_matches_simulation(model_cls, modified):
    rng = np.random.default_rng(5)
    n = 100_000
    lam, p = rng.gamma(R, 1 / ALPHA, n), rng.beta(A, B, n)
    alive = np.ones(n, bool) if not modified else rng.random(n) >= p
    t, count = np.zeros(n), np.zeros(n)
    for _ in range(300):
        t = t + rng.exponential(1 / lam)
        ok = alive & (t <= 26)
        count += ok
        alive = ok & (rng.random(n) >= p)
    expected = model_cls(r=R, alpha=ALPHA, a=A, b=B).expected_purchases_new(26)
    assert expected == pytest.approx(count.mean(), rel=0.03)


def test_gamma_gamma_recovers_mean_spend():
    rng = np.random.default_rng(9)
    p, q, g, n = 4.0, 3.5, 250.0, 6000
    nu = rng.gamma(q, 1 / g, n)
    x = rng.integers(1, 15, n)
    m = np.array([rng.gamma(p, 1 / nu[i], x[i]).mean() for i in range(n)])
    gg = GammaGamma().fit(x, m)
    assert gg.converged
    assert gg.population_mean == pytest.approx(p * g / (q - 1), rel=0.05)
    future = np.array([rng.gamma(p, 1 / nu[i]) for i in range(n)])
    assert gg.expected_spend(x, m).mean() == pytest.approx(future.mean(), rel=0.05)


def test_gamma_gamma_shrinks_less_with_more_orders():
    gg = GammaGamma(p=4.0, q=3.5, gamma=250.0)
    w = gg.shrinkage([1, 5, 50])
    assert w[0] < w[1] < w[2] < 1
    # a customer with no repeat order gets the population mean
    assert gg.expected_spend([0], [0.0])[0] == pytest.approx(gg.population_mean)
