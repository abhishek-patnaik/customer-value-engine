# Model card: customer value over the next 26 weeks

**What it predicts.** For each known customer: expected number of orders, expected value per order, and so expected revenue over the next 26 weeks. Also the probability the customer is still active.

**Models.** MBG/NBD (Batislam, Denizel and Filiztekin, 2007) for order counts, Gamma-Gamma (Fader, Hardie and Lee, 2005) for order value. Both written from the papers in `clv/models.py` and checked against simulation in `tests/test_models.py`.

**Unit of purchase.** A customer ordering on a given calendar day, however many invoices that day.

**Clock.** Seasonal weeks. A monthly index from customers known to be active up to the cutoff (bought in its last four weeks), Poisson with a fixed effect per customer, each month shrunk toward 1 by a prior of 50 purchase days.

**Training data.** Clean invoices from known customers, 01 Dec 2009 to 09 Dec 2011.

**Fitted parameters (final model).** r = 0.740, alpha = 10.01, a = 0.178, b = 3.56; p = 2.20, q = 3.60, gamma = 462.9.

**Backtests.** Fit before a cutoff, predict the next 26 weeks for customers known at the cutoff.

| Holdout | Error on total revenue | Rank correlation | Top 10% hold |
|---|---|---|---|
| Jun to Dec 2011 (busy season) | -5% | 0.63 | 63% |
| Dec 2010 to Jun 2011 (quiet season) | +51% | 0.55 | 61% |

**Use it for.** Ranking customers by expected value, spotting large accounts that have probably gone quiet, and a planning range for revenue from the existing base.

**Do not use it for.** A single point forecast of next half year's revenue. In the quiet season backtest the total came out 51% too high. Use the planning range, £2.49M to £3.96M, and plan on the low end. It also says nothing about customers not yet acquired.

**Known weaknesses.**
- Over-predicted by 51% in the one backtest with only a year of history, most of all for recently acquired customers.
- Cannot anticipate a business-wide slowdown.
- Assumes order value is unrelated to order frequency. Measured rank correlation 0.25, so the best customers are slightly under-valued.
- Gives one-off buyers a high chance of still being active (77% to 95%), probably too high.
- Customers without an id are not modelled.
