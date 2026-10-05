"""BG/NBD and Gamma-Gamma, written out from the papers.

I wrote these myself instead of using the `lifetimes` package. It is no
longer maintained and breaks on current pandas, and the likelihoods are short
enough that owning them is easier than patching around someone else's code.
Every formula below is checked in the tests, against simulation where there
is no closed form to compare with.

References
    Fader, Hardie and Lee (2005), "Counting Your Customers" the Easy Way:
        An Alternative to the Pareto/NBD Model. Marketing Science 24(2).
    Fader, Hardie and Lee (2005), RFM and CLV: Using Iso-Value Curves for
        Customer Base Analysis. Journal of Marketing Research 42(4). The
        Gamma-Gamma spend model is in the appendix.

Notation, per customer, all time in weeks
    x    number of repeat purchases (purchase days after the first one)
    t_x  time of the last purchase, measured from the first
    T    time from the first purchase to the end of the observation window
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special


def _group(*cols: np.ndarray):
    """Collapse identical rows so the likelihood is evaluated once per pattern."""
    stacked = np.column_stack(cols)
    uniq, inv, counts = np.unique(stacked, axis=0, return_inverse=True, return_counts=True)
    return uniq, counts


@dataclass
class BGNBD:
    r: float = np.nan
    alpha: float = np.nan
    a: float = np.nan
    b: float = np.nan
    converged: bool = False
    loglik: float = np.nan

    @property
    def params(self) -> dict:
        return {"r": self.r, "alpha": self.alpha, "a": self.a, "b": self.b}

    @staticmethod
    def _ll(params, x, tx, T):
        r, alpha, a, b = params
        A1 = special.gammaln(r + x) - special.gammaln(r) + r * np.log(alpha)
        A2 = special.gammaln(a + b) + special.gammaln(b + x) - special.gammaln(b) - special.gammaln(a + b + x)
        A3 = -(r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            A4 = np.where(x > 0, np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + tx), -np.inf)
        return A1 + A2 + np.logaddexp(A3, A4)

    def fit(self, x, tx, T, start=(1.0, 1.0, 1.0, 1.0)) -> "BGNBD":
        (rows, w) = _group(np.asarray(x, float), np.asarray(tx, float), np.asarray(T, float))
        xs, txs, Ts = rows.T
        scale = max(Ts.max() / 10, 1.0)

        def nll(logp):
            p = np.exp(logp)
            p = (p[0], p[1] * scale, p[2], p[3])
            return -(w * self._ll(p, xs, txs, Ts)).sum() / w.sum()

        best = None
        for s in (start, (0.5, 0.5, 0.5, 2.0), (2.0, 2.0, 1.0, 1.0)):
            init = np.log([s[0], s[1], s[2], s[3]])
            res = optimize.minimize(nll, init, method="L-BFGS-B", bounds=[(-8, 6)] * 4)
            res = optimize.minimize(nll, res.x, method="Nelder-Mead",
                                    options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 4000})
            if best is None or res.fun < best.fun:
                best = res
        r, alpha, a, b = np.exp(best.x)
        self.r, self.alpha, self.a, self.b = float(r), float(alpha * scale), float(a), float(b)
        self.converged = bool(best.success) and np.all(np.abs(best.x) < 5.9)
        self.loglik = float(-best.fun * w.sum())
        return self

    def p_alive(self, x, tx, T) -> np.ndarray:
        x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(a) - np.log(b + x - 1) + (r + x) * (np.log(alpha + T) - np.log(alpha + tx))
        odds = np.where(x > 0, np.exp(log_ratio), 0.0)
        return 1.0 / (1.0 + odds)

    def expected_purchases(self, t, x, tx, T) -> np.ndarray:
        """Expected purchases in the next t weeks, given each customer's history."""
        x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        z = t / (alpha + T + t)
        hyp = special.hyp2f1(r + x, b + x, a + b + x - 1, z)
        first = (a + b + x - 1) / (a - 1)
        second = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp
        return first * second * self.p_alive(x, tx, T)

    def expected_purchases_new(self, t) -> np.ndarray:
        """Expected repeat purchases in the first t weeks for a brand new customer."""
        t = np.asarray(t, float)
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        hyp = special.hyp2f1(r, b, a + b - 1, t / (alpha + t))
        return (a + b - 1) / (a - 1) * (1 - (alpha / (alpha + t)) ** r * hyp)


@dataclass
class MBGNBD(BGNBD):
    """Modified BG/NBD (Batislam, Denizel and Filiztekin, 2007).

    Same as BG/NBD except a customer can also drop out straight after the
    first purchase. In plain BG/NBD someone who bought once two years ago and
    never came back is still "alive" with certainty, and in this data that
    is thousands of customers.
    """

    @staticmethod
    def _ll(params, x, tx, T):
        r, alpha, a, b = params
        A1 = special.gammaln(r + x) - special.gammaln(r) + r * np.log(alpha)
        A2 = (special.gammaln(a + b) + special.gammaln(b + x + 1) - special.gammaln(b)
              - special.gammaln(a + b + x + 1))
        A3 = -(r + x) * np.log(alpha + T)
        A4 = np.log(a) - np.log(b + x) + (r + x) * (np.log(alpha + T) - np.log(alpha + tx))
        return A1 + A2 + A3 + np.logaddexp(0.0, A4)

    def p_alive(self, x, tx, T) -> np.ndarray:
        x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        log_odds = np.log(a) - np.log(b + x) + (r + x) * (np.log(alpha + T) - np.log(alpha + tx))
        return 1.0 / (1.0 + np.exp(log_odds))

    def expected_purchases(self, t, x, tx, T) -> np.ndarray:
        x, tx, T = (np.asarray(v, float) for v in (x, tx, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        hyp = special.hyp2f1(r + x, b + x + 1, a + b + x, t / (alpha + T + t))
        first = (a + b + x) / (a - 1)
        second = 1 - hyp * ((alpha + T) / (alpha + T + t)) ** (r + x)
        return first * second * self.p_alive(x, tx, T)

    def expected_purchases_new(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        hyp = special.hyp2f1(r, b + 1, a + b, t / (alpha + t))
        return b / (a - 1) * (1 - hyp * (alpha / (alpha + t)) ** r)


@dataclass
class GammaGamma:
    p: float = np.nan
    q: float = np.nan
    gamma: float = np.nan
    converged: bool = False

    @property
    def params(self) -> dict:
        return {"p": self.p, "q": self.q, "gamma": self.gamma}

    @staticmethod
    def _ll(params, x, m):
        p, q, g = params
        return (special.gammaln(p * x + q) - special.gammaln(p * x) - special.gammaln(q) + q * np.log(g)
                + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(g + x * m))

    def fit(self, x, m) -> "GammaGamma":
        """x = number of repeat purchases (> 0), m = mean value of those purchases."""
        x, m = np.asarray(x, float), np.asarray(m, float)
        keep = (x > 0) & (m > 0)
        x, m = x[keep], m[keep]
        scale = float(np.median(m))

        def nll(logp):
            p, q, g = np.exp(logp)
            return -self._ll((p, q, g * scale), x, m).mean()

        best = None
        for s in ((1.0, 2.0, 1.0), (4.0, 4.0, 1.0), (0.5, 1.5, 0.5)):
            res = optimize.minimize(nll, np.log(s), method="L-BFGS-B", bounds=[(-6, 8)] * 3)
            res = optimize.minimize(nll, res.x, method="Nelder-Mead",
                                    options={"xatol": 1e-8, "fatol": 1e-11, "maxiter": 4000})
            if best is None or res.fun < best.fun:
                best = res
        p, q, g = np.exp(best.x)
        self.p, self.q, self.gamma = float(p), float(q), float(g * scale)
        self.converged = bool(best.success) and self.q > 1
        return self

    @property
    def population_mean(self) -> float:
        return self.p * self.gamma / (self.q - 1)

    def expected_spend(self, x, m) -> np.ndarray:
        """Expected value of a future purchase. Customers with no repeat
        purchase get the population mean, everyone else a blend of the
        population mean and their own average, weighted by how much
        evidence they have."""
        x, m = np.asarray(x, float), np.asarray(m, float)
        p, q, g = self.p, self.q, self.gamma
        with np.errstate(divide="ignore", invalid="ignore"):
            own = (g + x * m) * p / (p * x + q - 1)
        return np.where(x > 0, own, self.population_mean)

    def shrinkage(self, x) -> np.ndarray:
        """Weight on the customer's own average (0 = pure population mean)."""
        x = np.asarray(x, float)
        return self.p * x / (self.p * x + self.q - 1)
