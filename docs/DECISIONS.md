# Decisions log

The choices that shaped the project, why I made them, and what I gave up. Written as I went, cleaned up at the end.

## Data

**UCI Online Retail II, both sheets, nothing sampled.**
It is a real UK online wholesaler (mostly gift and homeware items, many customers are small shops buying in bulk), with every invoice line from December 2009 to December 2011. Two full years matter here: a cohort chart needs time to show decay, and a value model needs a calibration period plus a holdout long enough to judge it.

**The raw file is downloaded, not committed.**
`python -m clv fetch` pulls the zip from UCI. The loader checks the row count of each sheet against the published numbers and refuses to run on a file that does not match, so a broken download fails loudly instead of quietly producing smaller numbers.

**Codes are normalised as text.**
Excel stores a stock code like `85123` as a number and `85123A` as text, and customer ids come through as `13085.0`. Everything is turned into a clean string once, in the loader, so `85123` and `85123.0` can never become two products.

## Cleaning

Every rule is logged in `outputs/tables/clean_audit.csv`: rows removed and the money those rows carried. It is a waterfall, so a row is charged to the first rule that catches it, and the order of the rules is part of the decision.

**The two sheets overlap, and the overlap goes first.**
The 2009-2010 sheet actually runs to 9 December 2010, and the 2010-2011 sheet starts on 1 December 2010. The 1,088 invoices from those nine days appear in both, 22,523 rows worth £377K. Any invoice already seen in the earlier sheet is dropped from the later one. If this ran after the duplicate check it would look like a duplicate problem, which it is not. Without this step, December 2010 would show roughly double its real sales, right in the middle of every cohort chart.

**Exact duplicate lines are dropped.**
Same invoice, product, quantity, price, customer and timestamp. A till can record a quantity, so the same product appearing twice on one invoice with identical everything is far more likely a double entry than two separate decisions. It is a judgement call, and the audit shows how much money it moves so anyone can disagree with it.

**Non-product lines are out.**
Postage, carriage, Amazon fees, bank charges, manual entries, discounts, samples, test lines and gift vouchers. They are not things a customer chose to buy, and postage especially would inflate the value of overseas customers who pay more to ship. The list lives in `config.toml`.

**Free lines and stock corrections are out.**
A sale line with price zero, or a negative quantity on a normal invoice, is the warehouse fixing its stock (damaged, lost, found), not a customer action. In the real file all 3,457 negative quantity lines on normal invoices have price zero and no customer, so the price rule catches every one of them first and the stock rule removes nothing. I kept it anyway as a guard: a correction with a price on it would otherwise be read as a sale.

**Lines without a customer id are out of every customer analysis, and their share is reported.**
The money is real but it cannot be tied to anyone, so it cannot show up in a cohort or a lifetime value. Pretending it belongs to known customers would overstate them. In this data that is 227,089 lines and 13.1% of product sales. Every customer number in the project is about known customers, not the whole business.

**A cancellation that exactly reverses an order removes both lines.**
This is the one rule that changes the answer most. The dataset has an order for 80,995 units of one item (£168K), cancelled twelve minutes later, and another for 74,215 units (£77K) from a customer whose other orders came to under £200. Dropping cancellations alone would keep that order and hand one customer a six-figure lifetime value. So each cancellation line is paired with the sale it undoes: same customer, product, unit price and quantity, placed before it and within a year. If several sales qualify, the most recent unused one is taken, and a sale can only be undone once. On the real data this pairs 5,956 cancellations with their orders and takes £546K of sales that never really happened out of customer values. Both bulk orders above are among them.

**Cancellations that match nothing are kept as returns, not subtracted from invoices.**
These are partial returns (fewer units than were bought) or returns of orders placed before December 2009. Pushing them into some earlier invoice would mean guessing which one. They sit in their own table so their value is known, and so customer level net revenue can subtract them where that matters. That is 11,630 lines worth £164K, about 1% of clean revenue, so leaving them out of invoice values moves nothing that matters.

**Twelve customer ids appear under two countries.**
All twelve are non-UK ids, each split across two countries (Denmark and Belgium, Austria and Cyprus and so on). It looks like the same id reused for different shops, but twelve customers out of 5,839 cannot move any result, so they are kept as one customer each. For anything split by country, a customer takes the country where they spent the most.

**Big customers are kept.**
Many customers are businesses that buy in bulk, so very large invoices are real, not errors. After cleaning, the median invoice is £301, the 99th percentile £3,406 and the largest £43,628. The top 1% of customers bring in 31% of revenue and the top 20% bring in 77%. Nothing is capped at the cleaning stage. How the value models cope with them is a modelling decision, logged with the models.

## What is left after cleaning

| | |
|---|---|
| Raw rows | 1,067,371 |
| Clean sale lines | 770,623 |
| Known customers | 5,839 |
| Invoices | 36,337 |
| Period | 1 Dec 2009 to 9 Dec 2011 |
| Product revenue, known customers | £16.52M |
| UK share of that revenue | 83.6% |
| Customers with a single purchase day | 28.5% |

The full waterfall, with the money each rule moved, is in `outputs/tables/clean_audit.csv`.

## Diagnose

**The December 2009 cohort is not a cohort.**
951 customers "first" bought in December 2009 because that is when the data starts. Most of them were customers long before. They are labelled as the existing base and kept out of every new customer number: cohort averages, time to second order, the acquisition check.

**Time to second order uses Kaplan-Meier, not a plain share.**
A customer who arrived in November 2011 has had a few weeks to come back. Counting them as "did not return" drags the repeat rate down for no reason. Kaplan-Meier treats them as still waiting, which is the standard fix for this. The curve is reported up to one year after the first order.

**Frequency counts days, not invoices.**
2,837 times a customer placed more than one invoice on the same day. That is usually one order split up, not a second decision, so RFM frequency and every model below count purchase days.

**The cohort chart stops at the last full month.**
December 2011 has nine days. Left in, every cohort would appear to drop off a cliff in its final column.

**I checked the acquisition trend before reporting it.**
A plain count said first-time buyers fell from 2,094 to 1,132 between April to November 2010 and the same months of 2011. That number compares a 4 to 11 month lookback with a 16 to 23 month one. Giving both years the same lookback (no purchase since the previous December) gives 2,094 against 2,155. I report that the data cannot say whether acquisition fell.

**RFM uses quintiles on rank, and a fixed map from scores to segments.**
Frequency has huge ties (thousands of customers with one purchase day), so scores are taken on rank to keep five equal groups. The segment names follow the common R by FM grid, written out in `diagnose.segment_for` so every one of the 25 cells is assigned on purpose.

## Predict

**I wrote the models instead of using `lifetimes`.**
It has not been maintained for years and fails on current pandas. The likelihoods are short and well documented, so owning them was less work than patching around a dead dependency. Each one is tested against simulated customers with known parameters, including the case a < 1, where the closed form is easy to get wrong.

**MBG/NBD over BG/NBD, even though they score the same.**
I started with plain BG/NBD. Its probability of being active is exactly 1 for anyone with no repeat purchase, however long ago they bought. In this base that is 1,667 customers, and the budget plan uses that probability to work out value at risk. MBG/NBD lets a customer drop out straight after the first order, so the probability at least falls with time. It does not fall far: one-off buyers still get 0.77 to 0.95, which is probably generous, so their value at risk is if anything overstated. That only strengthens the plan's case against win-back spend on small accounts. Both backtests came out within a percent of each other, so I chose on behaviour, not on score.

**Time is measured in seasonal weeks.**
These models assume a steady buying rate through the year. In this business November runs at about 1.6 times an average month and January at about 0.8. I estimate a monthly index and let each day count for its month's index, so a week in November is "longer" than a week in January. The models fit and forecast on that clock. In the busy season backtest this took the error on the total from -16% to -5%. The ranking did not change, which is what you would expect: the clock changes when time passes, not who buys more.

**How the index is estimated, after two mistakes.**
The final version uses only customers known to be active for the whole stretch, meaning they bought at least once in the four weeks before the cutoff. While a customer is active the model says they buy at a steady personal rate, so any month to month pattern in their purchases is seasonality. Purchase days are fitted with a Poisson model with one fixed effect per customer, so a heavy buyer joining the panel cannot pass for a busy month. A month seen only a few times is pulled toward 1. The index is always estimated from data before the cutoff.

I got there in three steps:

1. My first index came from the existing base (customers who bought in December 2009), regressing their monthly purchase days on month of year plus a trend. It made December look about a third busier than it is, because the base is defined as "bought in December 2009", so in that month every one of them is active by construction. Worth remembering any time a group is defined by an action in a period and then measured in the same period.
2. Excluding that month fixed December, but the simulated store caught a second problem. It has no seasonality at all, yet in the run before the fix the seasonal clock came out 11% low in one backtest and 53% high in the other, while the plain calendar clock was within 3% on both. A group of customers buys less over time simply because some of them leave, and with a year or less of history that decline cannot be told apart from seasonality, so it leaked into the index. Restricting the index to customers known to still be active removes that decline by construction.
3. With that change the simulated store comes out within 3% on both backtests, as it should, and the real index is stable whichever cutoff it is estimated from.

On the real data the second fix moved the quiet season backtest from +56% to +51%. Most of that miss is real, not my code.

**Two backtests, one per half of the year.**
My first backtest held out June to December 2011 and came out at a few percent off, which looked great. The forecast covers December to June, so I added a backtest on December 2010 to June 2011. It over-predicted by 51%. Three things drive it: that fold only has a year of history, new customers buy heavily in their first months and the model takes that as their normal rate, and the first half of 2011 was simply weak for the business. I kept the result in the report rather than tuning until it went away.

**The forecast is a range, and the plan uses the low end.**
Bootstrapping customers gives a parameter interval of about ±1%, which describes noise in the fitted numbers and says nothing about the model being wrong. The two backtests are a better guide to that. The planning range is the forecast divided by (1 + error) for each backtest. The quiet season one matches the forecast window, so its correction is the planning case used by the budget layer. I would rather the plan be pleasantly surprised.

**Baselines a team would actually use.**
"Same as last 26 weeks" and "historical average rate". The model beats both on the total in both seasons and on ranking in the busy season. In the quiet season the ranking is close to a tie with "same as last 26 weeks". I say that rather than claim a win.

**Big customers are handled by Gamma-Gamma, not capped.**
Gamma-Gamma blends each customer's own average order with the population average, weighted by how many orders they have. A customer with 60 orders is predicted almost exactly at their own average, so a large account keeps its large orders. Capping them would make the forecast wrong for exactly the customers that matter most.

**Gamma-Gamma's independence assumption is a little off.**
It assumes order value is unrelated to order frequency. The rank correlation is 0.25. Frequent buyers place somewhat larger orders, so the model slightly under-values its best customers. I left it, noted it in the model card, and checked that predicted and actual order values for buyers in the holdout are close (£420 against £439).

## Decide

**The response to marketing is assumed, and the plan is tested across those assumptions.**
There is no campaign data, so lift, recovery, cost per account and margin are judgement calls. Each has a base case and a range in `config.toml`. 2,000 runs draw every assumption from its range, along with customer values anywhere between the planning case and the raw model forecast, and the whole plan is re-optimised each time. The report only leans on conclusions that survive that.

**Diminishing returns per account.**
Extra revenue from spending x on a customer is M × (1 - e^(-x/k)): the first call matters more than the fifth. With a concave response the best plan has a closed form. Spend on each account until the last pound returns the same everywhere, and stop entirely where it returns less than a pound of margin. That is simple enough to check by hand, which matters for a plan someone will be asked to defend.

**Optimised per customer, reported per segment.**
Segments are how marketing teams talk, but averaging within a segment hides the fact that the top Champion is worth a hundred times the median one. The optimiser works per account and the report sums up by segment.

**The plan does not have to spend the budget.**
If the next pound returns less than a pound, the plan stops. I also show what happens if the whole £50,000 has to go out anyway, placed as well as possible, because "use it or lose it" budgets are real. It loses money in about half the runs.

**The comparison plans are the ones people actually propose.**
Equal spend per customer, only the best customers, and win back everyone lapsing. All three spend the full budget, so the comparison mixes "where" and "how much". The forced optimised plan separates the two.

**Value at risk comes from the model.**
Expected value is P(active) × value if active, so value if active is expected value / P(active), and the part at risk is that times (1 - P(active)). No new parameters, and it is zero for someone the model is sure is still buying.

## Power BI

**CSV star schema, no Python at refresh time.**
Whoever opens the report should not need a Python environment. The pipeline writes the tables, Power BI reads them. Customers are the dimension everything hangs off, invoices and returns are facts, and the already aggregated tables (cohorts, monthly forecast, plan) stay unrelated so nobody double counts them by accident.

**Margin is a slider in the report.**
It is the assumption people argue about most, so the report lets them move it. The note on the page says that the slider rescales the plan but only a re-run re-optimises it.

## Testing

**The cleaning is tested against simulated data with planted problems.**
The simulator generates customers with the same processes the value models assume, then plants known amounts of every kind of mess the real file has: overlap between the sheets, duplicate lines, bad debt entries, postage, free lines, stock corrections, guest orders, full cancellations and partial returns. The tests check that each rule removes exactly what was planted and that the clean revenue equals the true orders minus the cancelled ones. Small hand made cases cover the matching rule's edges: most recent sale wins, never a later sale, never another customer or price, a sale is undone once, the window is respected, partial returns stay unmatched.

**The models are tested against simulation.**
Simulated customers from known BG/NBD, MBG/NBD and Gamma-Gamma parameters: the fit has to recover the parameters, and the conditional predictions have to match a Monte Carlo of the same process.

**The optimiser is tested on cases with known answers.**
It never spends past the point of no return, never goes over budget, spends nothing on an account worth nothing, and gives more to the account with more at stake.

**The write up is tested too.**
No em or en dashes, no placeholders, no "nan" anywhere in the generated text.
