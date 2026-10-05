# Building the Power BI report

The finished report is in `powerbi/`. I generate it from code with `python -m clv powerbi-project` ([clv/pbip.py](../clv/pbip.py)), which writes the data model, measures, theme and visuals as a Power BI project. Power BI only reads the CSV files in `outputs/powerbi/`, so nobody needs Python to open or refresh it.

This page describes the same build as manual steps: what the model looks like, every measure, and what is on each page. It is useful for reading the report's logic, or for rebuilding it by hand.

## 1. The files

| File | Grain | What it is for |
|---|---|---|
| `dim_customer.csv` | one row per customer | RFM scores, segment, model outputs, the recommended action |
| `fact_invoice.csv` | one row per clean invoice | revenue over time |
| `fact_return.csv` | one row per unmatched return line | returns |
| `dim_date.csv` | one row per day, Dec 2009 to Jun 2012 | the date table, forecast days included |
| `dim_segment.csv` | one row per RFM segment | order, description, planned spend and its range |
| `fact_cohort.csv` | one row per cohort per month of age | the retention matrix |
| `fact_month.csv` | one row per month | actual revenue, then the model and planning forecasts |
| `plan_budget_curve.csv` | one row per budget level | profit at each budget |
| `plan_strategies.csv` | one row per strategy | the strategy comparison |
| `cve_theme.json` | | colours matching the charts in the repo |

## 2. Load the data

1. Open Power BI Desktop, **Get data > Text/CSV**, and load each CSV above. Click **Transform data** rather than Load on the first one so you can fix the types.
2. In Power Query, check these types. Power BI guesses most of them right, but not all of these:
   - `Customer ID` is **Text** in every table. If it is left as a number, the relationships still work, but Power BI starts summing customer ids in visuals.
   - `Date`, `First Purchase`, `Last Purchase`, `Cohort Month`, `Month Start` are **Date**.
   - `Invoice` is **Text** (some start with letters in the raw data).
   - `Months Since First Purchase`, `Purchase Days`, `Value Rank`, `Value Decile` are **Whole number**.
3. **Close & Apply**.

## 3. Model

In Model view, create these relationships (one to many, single direction):

| From (one) | To (many) |
|---|---|
| `dim_customer[Customer ID]` | `fact_invoice[Customer ID]` |
| `dim_customer[Customer ID]` | `fact_return[Customer ID]` |
| `dim_date[Date]` | `fact_invoice[Date]` |
| `dim_date[Date]` | `fact_return[Date]` |
| `dim_segment[Segment]` | `dim_customer[Segment]` |

Leave `fact_cohort`, `fact_month` and the two `plan_` tables unrelated. They are already aggregated, and joining them to the date table only invites double counting.

Then:

- Select `dim_date`, **Table tools > Mark as date table**, using `Date`.
- Sort columns: `dim_segment[Segment]` by `Segment Order`, `dim_date[Month]` by `Month Number`, `dim_date[Weekday]` by `Weekday Number`. Do the same for `dim_customer[Segment]` by adding `Segment Order` to it with a lookup column:
  ```DAX
  Segment Order = RELATED(dim_segment[Segment Order])
  ```
- **View > Themes > Browse for themes** and pick `cve_theme.json`.

Three calculated columns used below:

```DAX
// dim_customer
P Alive Band =
SWITCH(TRUE(),
    dim_customer[P Alive] >= 0.9, "1. 90% and over",
    dim_customer[P Alive] >= 0.7, "2. 70 to 90%",
    dim_customer[P Alive] >= 0.4, "3. 40 to 70%",
    "4. under 40%")

// fact_cohort
Cohort Year = YEAR(fact_cohort[Cohort Month])

// fact_cohort
Cohort Label = FORMAT(fact_cohort[Cohort Month], "MMM yyyy")
```

## 4. Measures

Put them all in one empty table so they are easy to find: **Home > Enter data**, name it `_Measures`, load it, then add these.

```DAX
Revenue = SUM(fact_invoice[Revenue])
Orders = COUNTROWS(fact_invoice)
Active Customers = DISTINCTCOUNT(fact_invoice[Customer ID])
Avg Order Value = DIVIDE([Revenue], [Orders])
Returns = SUM(fact_return[Return Value])
Return Rate = DIVIDE([Returns], [Revenue])

Revenue PY = CALCULATE([Revenue], SAMEPERIODLASTYEAR(dim_date[Date]))
Revenue YoY % = DIVIDE([Revenue] - [Revenue PY], [Revenue PY])

Customers = COUNTROWS(dim_customer)
Repeat Customer Rate =
    DIVIDE(CALCULATE([Customers], dim_customer[Purchase Days] > 1), [Customers])

Revenue Share of Total =
    DIVIDE([Revenue], CALCULATE([Revenue], ALL(dim_customer), ALL(dim_segment)))

Top 20% Revenue Share =
VAR n = ROUNDUP([Customers] * 0.2, 0)
VAR top = TOPN(n, dim_customer, dim_customer[Revenue 2 Years], DESC)
RETURN DIVIDE(SUMX(top, dim_customer[Revenue 2 Years]), SUM(dim_customer[Revenue 2 Years]))

// cohort matrix: weighted, so totals and averages stay correct
Retention % = DIVIDE(SUM(fact_cohort[Active Customers]), SUM(fact_cohort[Cohort Size]))
Revenue Retention % =
    DIVIDE(
        SUM(fact_cohort[Revenue]),
        CALCULATE(SUM(fact_cohort[Revenue]), fact_cohort[Months Since First Purchase] = 0,
                  REMOVEFILTERS(fact_cohort[Months Since First Purchase])))

// model outputs
Model Value 26w = SUM(dim_customer[Model Value 26w])
Planning Value 26w = SUM(dim_customer[Planning Value 26w])
Value At Risk 26w = SUM(dim_customer[Value At Risk 26w])
Avg P Alive = AVERAGE(dim_customer[P Alive])

// the plan
Planned Spend = SUM(dim_customer[Recommended Spend])
Accounts Targeted = CALCULATE([Customers], dim_customer[In Target List] = "Yes")
Expected Extra Revenue = SUM(dim_customer[Expected Extra Revenue])
```

The margin is an assumption, so make it a slider instead of a constant: **Modeling > New parameter > Numeric range**, name `Margin`, minimum 0.25, maximum 0.45, increment 0.01, default 0.35. Power BI creates `Margin Value` for you. Then:

```DAX
Expected Extra Margin = [Expected Extra Revenue] * [Margin Value]
Margin Back per £1 = DIVIDE([Expected Extra Margin], [Planned Spend])
Profit After Spend = [Expected Extra Margin] - [Planned Spend]
```

The slider only rescales the plan that was optimised at 35%. Re-running the pipeline with a different `gross_margin` in `config.toml` is what actually re-optimises it. The page note below says so.

Format: currency measures as `£#,0`, rates as percentage with one decimal.

## 5. Pages

Canvas 1280 x 720. Every page gets a text box at the top with one sentence saying what the page answers, and the slicers in one row under it.

### Page 1. Overview

*"Where the revenue comes from and what the next six months look like."*

- Slicers: `dim_customer[Customer Type]`, `dim_customer[Country]` (dropdown).
- Four cards: `Revenue`, `Customers`, `Repeat Customer Rate`, `Planning Value 26w` (title it "Next 6 months, planning case").
- Line chart: X `fact_month[Month Start]`, Y `Actual Revenue`, `Planning Forecast`, `Model Forecast`. Set the forecast lines dashed. Add a subtitle: "Forecast covers customers known on 9 Dec 2011, December 2011 is partial."
- Bar chart: Y `dim_segment[Segment]`, X `Revenue Share of Total`, sorted by `Segment Order`.
- Card: `Top 20% Revenue Share`, titled "Revenue from the top 20% of customers".

### Page 2. Retention

*"Who comes back, and how fast."*

- Matrix: rows `fact_cohort[Cohort Month]` (format MMM yyyy), columns `Months Since First Purchase`, values `Retention %`. Filter the visual to `Cohort Type = New customers` and `Months Since First Purchase >= 1`. Conditional formatting > Background colour > Gradient, lowest `#F4F8FD`, highest `#0D366B`, font colour rule white above 20%.
- Line chart: X `Months Since First Purchase` (1 to 12), Y `Retention %`, legend `Cohort Year`. 2010 vs 2011 cohorts is the comparison worth seeing.
- Line chart: same, with `Revenue Retention %`.
- Text box: "Customers who were already buying in December 2009 are kept out of the matrix. They are not new customers, the data just starts there."

### Page 3. Customer value

*"Who is worth what over the next six months, and who might already be gone."*

- Slicers: `dim_customer[Segment]`, `dim_customer[P Alive Band]`.
- Column chart: X `Value Decile`, Y `Planning Value 26w`. It shows how steep the drop is after decile 1.
- Matrix: rows `Segment`, columns `P Alive Band`, values `Customers`. Conditional formatting on values.
- Table: top 25 accounts. Columns `Customer ID`, `Country`, `Segment`, `Revenue 2 Years`, `Planning Value 26w`, `P Alive` (data bars), `Days Since Last Purchase`. Visual level filter: Top N 25 by `Planning Value 26w`.
- Card: `Value At Risk 26w`, titled "Value at risk if nothing changes".

### Page 4. Retention budget plan

*"Where the retention money should go, and where it should not."*

- Slicer: the `Margin` parameter slider.
- Cards: `Planned Spend`, `Accounts Targeted`, `Expected Extra Margin`, `Margin Back per £1`.
- Bar chart: Y `dim_segment[Segment]`, X `dim_segment[Planned Spend]`. In the format pane add error bars from `Spend Low` to `Spend High` (Analytics > Error bars, by field). Title "Planned spend, with the range across assumptions".
- Line chart: X `plan_budget_curve[Budget]`, Y `Incremental Profit`. Add a constant X line at 50,000 labelled "budget on the table".
- Bar chart: Y `plan_strategies[Strategy]`, X `Median Profit`, error bars `Profit Low` to `Profit High`.
- Table: the target list. `Customer ID`, `Segment`, `Recommended Action`, `Recommended Spend`, `Expected Extra Revenue`, `P Alive`. Filter `In Target List = Yes`, sort by `Expected Extra Revenue`.
- Text box: "The response to marketing is assumed, not measured. The ranges come from config.toml. The first campaign should be run as a test with a holdout group, see the README."

## 6. Refresh

After a new pipeline run, **Home > Refresh** picks up the new CSVs. If the project folder moved, **Transform data > Data source settings > Change source** on each file, or switch every query to a single **Folder** source pointing at `outputs/powerbi`.

Save the report as `powerbi/customer_value_engine.pbix` and export a PDF of the four pages to `powerbi/` for people without Power BI.
