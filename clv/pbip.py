"""Builds the Power BI report as a Power BI Project (.pbip).

    python -m clv powerbi-project [--data-folder PATH]

A .pbip is a folder of plain text that Power BI Desktop opens like a .pbix:
the semantic model (tables, relationships, measures) as model.bim and the
report pages as report.json. Writing it from code means the report is
rebuilt from the same tables as everything else, and a change to a measure
shows up as a readable diff instead of a binary file.

The CSVs are read from one folder, held in the `DataFolder` parameter. If
the project moves, re-run this command, or change the parameter in Power BI
under Transform data > Manage parameters.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path, PureWindowsPath

from clv.config import ROOT
from clv.powerbi import theme

NAME = "CustomerValueEngine"
M = "_Measures"

# colours, the same as the charts in the repo
BLUE, ORANGE, AQUA, YELLOW = "#2A78D6", "#EB6834", "#1BAF7A", "#EDA100"
INK, INK_2, GRID, SURFACE = "#0B0B0B", "#52514E", "#E6E5E1", "#FCFCFB"
RAMP_LO, RAMP_HI = "#F4F8FD", "#0D366B"

# ---------------------------------------------------------------- model

TEXT, INT, DBL, DATE = "string", "int64", "double", "dateTime"

TABLES = {
    "dim_customer": [
        ("Customer ID", TEXT), ("Country", TEXT), ("First Purchase", DATE), ("Last Purchase", DATE),
        ("Cohort Month", DATE), ("Customer Type", TEXT), ("Purchase Days", INT), ("Revenue 2 Years", DBL),
        ("Returns 2 Years", DBL), ("Days Since Last Purchase", INT), ("R Score", INT), ("F Score", INT),
        ("M Score", INT), ("RFM Code", TEXT), ("Segment", TEXT), ("P Alive", DBL), ("Expected Purchases 26w", DBL),
        ("Expected Order Value", DBL), ("Model Value 26w", DBL), ("Planning Value 26w", DBL),
        ("Value At Risk 26w", DBL), ("Value Rank", INT), ("Value Decile", INT), ("Recommended Spend", DBL),
        ("Expected Extra Revenue", DBL), ("Recommended Action", TEXT), ("In Target List", TEXT),
    ],
    "fact_invoice": [
        ("Invoice", TEXT), ("Customer ID", TEXT), ("Date", DATE), ("Invoice Time", DATE), ("Revenue", DBL),
        ("Lines", INT), ("Units", INT), ("Country", TEXT),
    ],
    "fact_return": [
        ("Invoice", TEXT), ("Customer ID", TEXT), ("Date", DATE), ("Stock Code", TEXT), ("Return Value", DBL),
    ],
    "dim_date": [
        ("Date", DATE), ("Year", INT), ("Quarter", TEXT), ("Month Number", INT), ("Month", TEXT),
        ("Month Start", DATE), ("Year Month", TEXT), ("Weekday", TEXT), ("Weekday Number", INT),
        ("Is Forecast Period", TEXT),
    ],
    "dim_segment": [
        ("Segment", TEXT), ("Segment Order", INT), ("Description", TEXT), ("Action", TEXT), ("Planned Spend", DBL),
        ("Accounts Targeted", INT), ("Expected Extra Revenue", DBL), ("Spend Low", DBL), ("Spend High", DBL),
    ],
    "fact_cohort": [
        ("Cohort Month", DATE), ("Months Since First Purchase", INT), ("Cohort Size", INT),
        ("Active Customers", INT), ("Revenue", DBL), ("Retention", DBL), ("Revenue Retention", DBL),
        ("Cumulative Revenue Per Customer", DBL), ("Cohort Type", TEXT),
    ],
    "fact_month": [
        ("Month Start", DATE), ("Actual Revenue", DBL), ("Model Forecast", DBL), ("Planning Forecast", DBL),
        ("Note", TEXT),
    ],
    "plan_budget_curve": [("Budget", DBL), ("Incremental Revenue", DBL), ("Incremental Profit", DBL)],
    "plan_strategies": [
        ("Strategy", TEXT), ("Base Case Profit", DBL), ("Median Profit", DBL), ("Profit Low", DBL),
        ("Profit High", DBL), ("Share Of Runs Losing Money", DBL),
    ],
}

# columns that are labels or ids, never something to add up
NO_SUM = {"Customer ID", "Invoice", "RFM Code", "R Score", "F Score", "M Score", "Value Rank", "Value Decile",
          "Year", "Month Number", "Weekday Number", "Segment Order", "Months Since First Purchase", "P Alive",
          "Days Since Last Purchase", "Budget"}

SORT_BY = {("dim_date", "Month"): "Month Number", ("dim_date", "Weekday"): "Weekday Number",
           ("dim_segment", "Segment"): "Segment Order"}

CALC_COLUMNS = {
    "dim_customer": [
        ("Segment Order", "RELATED(dim_segment[Segment Order])", INT, None),
        ("P Alive Band", 'SWITCH(TRUE(), dim_customer[P Alive] >= 0.9, "90% and over", '
                         'dim_customer[P Alive] >= 0.7, "70 to 90%", dim_customer[P Alive] >= 0.4, "40 to 70%", '
                         '"under 40%")', TEXT, None),
        ("P Alive Band Order", 'SWITCH(TRUE(), dim_customer[P Alive] >= 0.9, 1, dim_customer[P Alive] >= 0.7, 2, '
                               'dim_customer[P Alive] >= 0.4, 3, 4)', INT, None),
    ],
    "fact_cohort": [
        ("Cohort Year", "FORMAT(fact_cohort[Cohort Month], \"yyyy\") & \" cohorts\"", TEXT, None),
        ("Cohort Label", "FORMAT(fact_cohort[Cohort Month], \"mmm yyyy\")", TEXT, None),
    ],
}
CALC_SORT = {("dim_customer", "Segment"): "Segment Order", ("dim_customer", "P Alive Band"): "P Alive Band Order",
             ("fact_cohort", "Cohort Label"): "Cohort Month"}

GBP, PCT, INT_FMT, GBP2 = "£#,0", "0.0%", "#,0", "£#,0.00"

MEASURES = [
    # name, expression, format
    ("Revenue", "SUM(fact_invoice[Revenue])", GBP),
    ("Orders", "COUNTROWS(fact_invoice)", INT_FMT),
    ("Active Customers", "DISTINCTCOUNT(fact_invoice[Customer ID])", INT_FMT),
    ("Avg Order Value", "DIVIDE([Revenue], [Orders])", GBP),
    ("Returns", "SUM(fact_return[Return Value])", GBP),
    ("Return Rate", "DIVIDE([Returns], [Revenue])", PCT),
    ("Customers", "COUNTROWS(dim_customer)", INT_FMT),
    ("Repeat Customer Rate", "DIVIDE(CALCULATE([Customers], dim_customer[Purchase Days] > 1), [Customers])", PCT),
    ("Revenue Share of Total", "DIVIDE([Revenue], CALCULATE([Revenue], ALL(dim_customer), ALL(dim_segment)))", PCT),
    ("Top 20% Revenue Share",
     "VAR n = ROUNDUP([Customers] * 0.2, 0)\nVAR biggest = TOPN(n, dim_customer, dim_customer[Revenue 2 Years], DESC)\n"
     "RETURN DIVIDE(SUMX(biggest, dim_customer[Revenue 2 Years]), SUM(dim_customer[Revenue 2 Years]))", PCT),
    ("Actual Revenue", "SUM(fact_month[Actual Revenue])", GBP),
    ("Model Forecast", "SUM(fact_month[Model Forecast])", GBP),
    ("Planning Forecast", "SUM(fact_month[Planning Forecast])", GBP),
    # cohorts: weighted so totals stay right; new customers only, from month 1
    ("Retention %", "DIVIDE(SUM(fact_cohort[Active Customers]), SUM(fact_cohort[Cohort Size]))", PCT),
    ("New Customer Retention",
     "IF(SELECTEDVALUE(fact_cohort[Months Since First Purchase]) = 0, BLANK(),\n"
     "    CALCULATE([Retention %], fact_cohort[Cohort Type] = \"New customers\"))", "0%"),
    ("Retention First Year",
     "VAR age = SELECTEDVALUE(fact_cohort[Months Since First Purchase])\n"
     "RETURN IF(age >= 1 && age <= 12, CALCULATE([Retention %], fact_cohort[Cohort Type] = \"New customers\"))", PCT),
    ("Month 1 Retention",
     "CALCULATE([Retention %], fact_cohort[Cohort Type] = \"New customers\", "
     "fact_cohort[Months Since First Purchase] = 1)", PCT),
    # model outputs
    ("Model Value 26w", "SUM(dim_customer[Model Value 26w])", GBP),
    ("Planning Value 26w", "SUM(dim_customer[Planning Value 26w])", GBP),
    ("Value At Risk 26w", "SUM(dim_customer[Value At Risk 26w])", GBP),
    ("Avg P Alive", "AVERAGE(dim_customer[P Alive])", PCT),
    ("Top 25 Planning Value", "IF(MIN(dim_customer[Value Rank]) <= 25, [Planning Value 26w])", GBP),
    # the plan
    ("Planned Spend", "SUM(dim_customer[Recommended Spend])", GBP),
    ("Accounts Targeted", "CALCULATE([Customers], dim_customer[In Target List] = \"Yes\")", INT_FMT),
    ("Expected Extra Revenue", "SUM(dim_customer[Expected Extra Revenue])", GBP),
    ("Margin Value", "SELECTEDVALUE('Margin'[Margin], 0.35)", "0%"),
    ("Expected Extra Margin", "[Expected Extra Revenue] * [Margin Value]", GBP),
    ("Margin Back per Pound", "DIVIDE([Expected Extra Margin], [Planned Spend])", GBP2),
    ("Profit After Spend", "[Expected Extra Margin] - [Planned Spend]", GBP),
    ("Target Spend", "VAR s = SUM(dim_customer[Recommended Spend]) RETURN IF(s > 0, s)", GBP),
    ("Target Extra Revenue",
     "IF(SUM(dim_customer[Recommended Spend]) > 0, SUM(dim_customer[Expected Extra Revenue]))", GBP),
    ("Budget Profit", "SUM(plan_budget_curve[Incremental Profit])", GBP),
    ("Strategy Median Profit", "SUM(plan_strategies[Median Profit])", GBP),
    ("Strategy Base Case Profit", "SUM(plan_strategies[Base Case Profit])", GBP),
    ("Segment Spend Low", "SUM(dim_segment[Spend Low])", GBP),
    ("Segment Spend High", "SUM(dim_segment[Spend High])", GBP),
]

RELATIONSHIPS = [
    ("fact_invoice", "Customer ID", "dim_customer", "Customer ID"),
    ("fact_return", "Customer ID", "dim_customer", "Customer ID"),
    ("fact_invoice", "Date", "dim_date", "Date"),
    ("fact_return", "Date", "dim_date", "Date"),
    ("dim_customer", "Segment", "dim_segment", "Segment"),
]


def _m_query(table: str, cols) -> list[str]:
    m_type = {TEXT: "type text", INT: "Int64.Type", DBL: "type number", DATE: "type date"}
    types = ", ".join(
        f'{{"{c}", {"type datetime" if (table, c) == ("fact_invoice", "Invoice Time") else m_type[t]}}}'
        for c, t in cols)
    return [
        "let",
        f'    Source = Csv.Document(File.Contents(DataFolder & "{table}.csv"), '
        '[Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),',
        "    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),",
        f'    Typed = Table.TransformColumnTypes(Promoted, {{{types}}}, "en-US")',
        "in",
        "    Typed",
    ]


def model(data_folder: str) -> dict:
    tables = []
    for name, cols in TABLES.items():
        columns = []
        for c, t in cols:
            col = {"name": c, "dataType": t, "sourceColumn": c,
                   "summarizeBy": "none" if (c in NO_SUM or t in (TEXT, DATE)) else "sum"}
            if t == DATE:
                col["formatString"] = "dd mmm yyyy" if c != "Invoice Time" else "dd mmm yyyy hh:nn"
            if (name, c) in SORT_BY:
                col["sortByColumn"] = SORT_BY[(name, c)]
            if (name, c) in CALC_SORT:
                col["sortByColumn"] = CALC_SORT[(name, c)]
            if name == "dim_date" and c == "Date":
                col["isKey"] = True
            if c in ("Revenue 2 Years", "Returns 2 Years", "Revenue", "Model Value 26w", "Planning Value 26w",
                     "Value At Risk 26w", "Recommended Spend", "Expected Extra Revenue", "Planned Spend",
                     "Return Value", "Actual Revenue", "Model Forecast", "Planning Forecast", "Spend Low",
                     "Spend High", "Expected Order Value", "Budget", "Incremental Revenue", "Incremental Profit",
                     "Base Case Profit", "Median Profit", "Profit Low", "Profit High"):
                col["formatString"] = GBP
            if c in ("P Alive", "Retention", "Revenue Retention", "Share Of Runs Losing Money"):
                col["formatString"] = "0%"
            columns.append(col)
        for c, expr, t, fmt in CALC_COLUMNS.get(name, []):
            col = {"type": "calculated", "name": c, "dataType": t, "isDataTypeInferred": False, "expression": expr,
                   "summarizeBy": "none"}
            if (name, c) in CALC_SORT:
                col["sortByColumn"] = CALC_SORT[(name, c)]
            if c.endswith("Order"):
                col["isHidden"] = True
            columns.append(col)
        t = {"name": name, "columns": columns,
             "partitions": [{"name": name, "mode": "import", "source": {"type": "m", "expression": _m_query(name, cols)}}]}
        if name == "dim_date":
            t["dataCategory"] = "Time"
        tables.append(t)

    tables.append({
        "name": "Margin",
        "columns": [{"type": "calculatedTableColumn", "name": "Margin", "dataType": DBL, "isNameInferred": True,
                     "isDataTypeInferred": True, "sourceColumn": "[Margin]", "formatString": "0%",
                     "summarizeBy": "none"}],
        "partitions": [{"name": "Margin", "mode": "import",
                        "source": {"type": "calculated",
                                   "expression": "SELECTCOLUMNS(GENERATESERIES(0.25, 0.45, 0.01), \"Margin\", [Value])"}}],
    })
    tables.append({
        "name": M,
        "columns": [{"type": "calculatedTableColumn", "name": "Column", "dataType": TEXT, "isHidden": True,
                     "isNameInferred": True, "isDataTypeInferred": True, "sourceColumn": "[Column]"}],
        "partitions": [{"name": M, "mode": "import", "source": {"type": "calculated", "expression": "ROW(\"Column\", \"\")"}}],
        "measures": [{"name": n, "expression": e.split("\n") if "\n" in e else e, "formatString": f}
                     for n, e, f in MEASURES],
    })
    rels = [{"name": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{a}.{b}.{c}.{d}")), "fromTable": a, "fromColumn": b,
             "toTable": c, "toColumn": d} for a, b, c, d in RELATIONSHIPS]
    return {
        "name": NAME,
        "compatibilityLevel": 1567,
        "model": {
            "culture": "en-GB",
            "dataAccessOptions": {"legacyRedirects": True, "returnErrorValuesAsNull": True},
            "defaultPowerBIDataSourceVersion": "powerBI_V3",
            "sourceQueryCulture": "en-GB",
            "tables": tables,
            "relationships": rels,
            "expressions": [{
                "name": "DataFolder", "kind": "m",
                "expression": f'"{data_folder}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]',
            }],
            "annotations": [
                {"name": "__PBI_TimeIntelligenceEnabled", "value": "0"},
                {"name": "PBI_QueryOrder", "value": json.dumps(["DataFolder"] + list(TABLES))},
                {"name": "PBIDesktopVersion", "value": "2.140"},
            ],
        },
    }


# ---------------------------------------------------------------- report helpers

def lit(v):
    if isinstance(v, bool):
        return {"expr": {"Literal": {"Value": "true" if v else "false"}}}
    if isinstance(v, (int, float)):
        return {"expr": {"Literal": {"Value": f"{v}D"}}}
    return {"expr": {"Literal": {"Value": f"'{v}'"}}}


def colour(hex_):
    return {"solid": {"color": lit(hex_)}}


def col(table, name):
    return ("col", table, name)


def mea(name):
    return ("mea", M, name)


class Query:
    """Builds the prototypeQuery and projections for one visual."""

    def __init__(self):
        self.sources = {}
        self.select = []
        self.order = []

    def _src(self, table):
        if table not in self.sources:
            self.sources[table] = f"t{len(self.sources)}"
        return self.sources[table]

    def field(self, f):
        kind, table, name = f
        ref = {"SourceRef": {"Source": self._src(table)}}
        if kind == "col":
            return {"Column": {"Expression": ref, "Property": name}}, f"{table}.{name}"
        return {"Measure": {"Expression": ref, "Property": name}}, f"{M}.{name}"

    def add(self, f):
        expr, qref = self.field(f)
        if qref not in [s["Name"] for s in self.select]:
            self.select.append({**expr, "Name": qref, "NativeReferenceName": f[2]})
        return qref

    def sort(self, f, desc=False):
        expr, _ = self.field(f)
        self.order.append({"Direction": 2 if desc else 1, "Expression": expr})

    def proto(self):
        q = {"Version": 2, "From": [{"Name": s, "Entity": t, "Type": 0} for t, s in self.sources.items()],
             "Select": self.select}
        if self.order:
            q["OrderBy"] = self.order
        return q


def title_obj(text, size=11, show=True):
    return {"title": [{"properties": {"show": lit(show), "text": lit(text), "fontSize": lit(size),
                                      "fontColor": colour(INK), "fontFamily": lit("Segoe UI Semibold")}}],
            "background": [{"properties": {"show": lit(False)}}],
            "border": [{"properties": {"show": lit(False)}}],
            "dropShadow": [{"properties": {"show": lit(False)}}]}


class Page:
    def __init__(self, name, display):
        self.name, self.display = name, display
        self.visuals = []

    def _add(self, x, y, w, h, single):
        vid = uuid.uuid5(uuid.NAMESPACE_URL, f"{self.name}-{len(self.visuals)}").hex[:20]
        cfg = {"name": vid, "layouts": [{"id": 0, "position": {"x": x, "y": y, "z": len(self.visuals) * 1000,
                                                               "width": w, "height": h, "tabOrder": len(self.visuals)}}],
               "singleVisual": single}
        self.visuals.append({"x": x, "y": y, "z": len(self.visuals) * 1000, "width": w, "height": h,
                             "config": json.dumps(cfg), "filters": "[]"})

    def text(self, x, y, w, h, runs):
        """runs: list of paragraphs, each a list of (text, size, bold, colour)."""
        paras = []
        for para in runs:
            paras.append({"textRuns": [{"value": t, "textStyle": {
                "fontSize": f"{s}pt", "fontWeight": "bold" if b else "normal", "color": c,
                "fontFamily": "Segoe UI Semibold" if b else "Segoe UI"}} for t, s, b, c in para]})
        self._add(x, y, w, h, {"visualType": "textbox", "drillFilterOtherVisuals": True,
                               "objects": {"general": [{"properties": {"paragraphs": paras}}]},
                               "vcObjects": {"background": [{"properties": {"show": lit(False)}}]}})

    def card(self, x, y, w, h, measure, label):
        q = Query()
        ref = q.add(mea(measure))
        self._add(x, y, w, h, {
            "visualType": "card", "projections": {"Values": [{"queryRef": ref}]}, "prototypeQuery": q.proto(),
            "objects": {"labels": [{"properties": {"color": colour(INK), "fontSize": lit(24),
                                                   "labelDisplayUnits": lit(1)}}],
                        "categoryLabels": [{"properties": {"show": lit(True), "color": colour(INK_2),
                                                           "fontSize": lit(10)}}]},
            "vcObjects": {**title_obj(label, show=False),
                          "background": [{"properties": {"show": lit(True), "color": colour("#FFFFFF")}}],
                          "border": [{"properties": {"show": lit(True), "color": colour(GRID), "radius": lit(6)}}]},
            "columnProperties": {ref: {"displayName": label}},
        })

    def slicer(self, x, y, w, h, field, label, dropdown=True, single=False):
        q = Query()
        ref = q.add(field)
        objects = {"data": [{"properties": {"mode": lit("Dropdown" if dropdown else "Basic")}}],
                   "header": [{"properties": {"show": lit(True), "text": lit(label), "fontColor": colour(INK_2)}}]}
        if single:
            objects["selection"] = [{"properties": {"singleSelect": lit(True)}}]
        self._add(x, y, w, h, {"visualType": "slicer", "projections": {"Values": [{"queryRef": ref, "active": True}]},
                               "prototypeQuery": q.proto(), "objects": objects,
                               "vcObjects": title_obj(label, show=False)})

    def chart(self, x, y, w, h, vtype, category, values, title, series=None, sort=None, colours=None,
              labels=False, legend=True, y_fmt=None, dashed=None, categorical=False, wide_labels=False,
              whole_labels=False):
        q = Query()
        proj = {"Category": [{"queryRef": q.add(category), "active": True}],
                "Y": [{"queryRef": q.add(v)} for v in values]}
        if series:
            proj["Series"] = [{"queryRef": q.add(series)}]
        if sort:
            q.sort(*sort) if isinstance(sort, tuple) and isinstance(sort[0], tuple) else q.sort(sort)
        objects = {
            "legend": [{"properties": {"show": lit(legend), "position": lit("Top"), "labelColor": colour(INK_2)}}],
            "categoryAxis": [{"properties": {"labelColor": colour(INK_2), "showAxisTitle": lit(False),
                                             "gridlineShow": lit(False)}}],
            "valueAxis": [{"properties": {"labelColor": colour(INK_2), "showAxisTitle": lit(False),
                                          "gridlineColor": colour(GRID)}}],
            "labels": [{"properties": {"show": lit(labels), "color": colour(INK_2)}}],
        }
        if colours:
            objects["dataPoint"] = [{"properties": {"fill": colour(colours[0])}}] + [
                {"properties": {"fill": colour(c)}, "selector": {"metadata": q.select[i + 1]["Name"]}}
                for i, c in enumerate(colours)]
        if categorical:
            objects["categoryAxis"][0]["properties"]["axisType"] = lit("Categorical")
        if wide_labels:
            objects["categoryAxis"][0]["properties"]["maxMarginFactor"] = lit(45)
        if whole_labels:
            objects["labels"][0]["properties"]["labelDisplayUnits"] = lit(1)
            objects["labels"][0]["properties"]["labelPrecision"] = lit(0)
        if dashed:
            objects["lineStyles"] = [{"properties": {"strokeLineJoin": lit("round"), "lineStyle": lit("dashed")},
                                      "selector": {"metadata": f"{M}.{m}"}} for m in dashed]
            objects["lineStyles"].append({"properties": {"strokeWidth": lit(2)}})
        self._add(x, y, w, h, {"visualType": vtype, "projections": proj, "prototypeQuery": q.proto(),
                               "objects": objects, "vcObjects": title_obj(title)})

    def table(self, x, y, w, h, fields, title, sort=None, names=None):
        q = Query()
        refs = [q.add(f) for f in fields]
        if sort:
            q.sort(*sort)
        cp = {r: {"displayName": n} for r, n in zip(refs, names or [])}
        self._add(x, y, w, h, {
            "visualType": "tableEx", "projections": {"Values": [{"queryRef": r} for r in refs]},
            "prototypeQuery": q.proto(), "columnProperties": cp,
            "objects": {"total": [{"properties": {"totals": lit(False)}}],
                        "columnHeaders": [{"properties": {"fontColor": colour(INK_2), "backColor": colour(SURFACE)}}],
                        "grid": [{"properties": {"gridHorizontalColor": colour(GRID), "rowPadding": lit(3)}}]},
            "vcObjects": title_obj(title)})

    def matrix(self, x, y, w, h, rows, cols_, value, title, gradient=False, fmt_pct=True):
        q = Query()
        r, c, v = q.add(rows), q.add(cols_), q.add(value)
        objects = {"subTotals": [{"properties": {"rowSubtotals": lit(False), "columnSubtotals": lit(False)}}],
                   "columnHeaders": [{"properties": {"fontColor": colour(INK_2), "backColor": colour(SURFACE)}}],
                   "rowHeaders": [{"properties": {"fontColor": colour(INK_2)}}],
                   "values": [{"properties": {"fontSize": lit(8)}}],
                   "grid": [{"properties": {"rowPadding": lit(1), "gridVerticalColor": colour(SURFACE),
                                            "gridHorizontalColor": colour(SURFACE)}}]}
        if gradient:
            measure_expr = {"Measure": {"Expression": {"SourceRef": {"Entity": value[1]}}, "Property": value[2]}}
            objects["values"].append({
                "properties": {"backColor": {"solid": {"color": {"expr": {"FillRule": {
                    "Input": measure_expr,
                    "FillRule": {"linearGradient2": {
                        "min": {"color": {"Literal": {"Value": f"'{RAMP_LO}'"}}},
                        "max": {"color": {"Literal": {"Value": f"'{RAMP_HI}'"}}},
                        "nullColoringStrategy": {"strategy": {"Literal": {"Value": "'noColor'"}}}}}}}}}}},
                "selector": {"data": [{"dataViewWildcard": {"matchingOption": 1}}], "metadata": v}})
            objects["values"].append({
                "properties": {"fontColor": {"solid": {"color": {"expr": {"FillRule": {
                    "Input": measure_expr,
                    "FillRule": {"linearGradient2": {
                        "min": {"color": {"Literal": {"Value": f"'{INK}'"}}, "value": {"Literal": {"Value": "0.2499D"}}},
                        "max": {"color": {"Literal": {"Value": "'#FFFFFF'"}}, "value": {"Literal": {"Value": "0.25D"}}},
                        "nullColoringStrategy": {"strategy": {"Literal": {"Value": "'noColor'"}}}}}}}}}}},
                "selector": {"data": [{"dataViewWildcard": {"matchingOption": 1}}], "metadata": v}})
        self._add(x, y, w, h, {"visualType": "pivotTable",
                               "projections": {"Rows": [{"queryRef": r, "active": True}], "Columns": [{"queryRef": c}],
                                               "Values": [{"queryRef": v}]},
                               "prototypeQuery": q.proto(), "objects": objects, "vcObjects": title_obj(title)})

    def section(self, ordinal):
        return {"name": self.name, "displayName": self.display, "displayOption": 1, "width": 1280, "height": 720,
                "ordinal": ordinal, "config": json.dumps({"objects": {"background": [{"properties": {
                    "color": colour(SURFACE), "transparency": lit(0)}}]}}),
                "filters": "[]", "visualContainers": self.visuals}


def header(p: Page, title: str, line: str):
    p.text(24, 12, 1232, 64, [[(title, 20, True, INK)], [(line, 11, False, INK_2)]])


# ---------------------------------------------------------------- pages

def pages() -> list[Page]:
    C, S, F, MO = "dim_customer", "dim_segment", "fact_cohort", "fact_month"

    p1 = Page("overview", "Overview")
    header(p1, "Customer Value Engine",
           "UK online wholesaler, Dec 2009 to Dec 2011. Where the revenue comes from and what the next six months "
           "look like. Known customers only.")
    p1.slicer(24, 84, 220, 56, col(C, "Customer Type"), "Customer type")
    p1.slicer(256, 84, 220, 56, col(C, "Country"), "Country")
    for i, (m, lab) in enumerate([("Revenue", "Revenue, two years"), ("Customers", "Customers"),
                                  ("Repeat Customer Rate", "Bought on more than one day"),
                                  ("Planning Value 26w", "Next 6 months, planning case"),
                                  ("Top 20% Revenue Share", "Revenue from the top 20%")]):
        p1.card(24 + i * 248, 152, 232, 96, m, lab)
    p1.chart(24, 264, 760, 432, "lineChart", col(MO, "Month Start"),
             [mea("Actual Revenue"), mea("Planning Forecast"), mea("Model Forecast")],
             "Monthly revenue, then the forecast for customers known on 9 Dec 2011",
             colours=[BLUE, ORANGE, AQUA], dashed=["Planning Forecast", "Model Forecast"])
    p1.chart(800, 264, 456, 432, "clusteredBarChart", col(S, "Segment"), [mea("Revenue Share of Total")],
             "Share of revenue by RFM segment", sort=col(S, "Segment"), colours=[BLUE], labels=True, legend=False,
             wide_labels=True)

    p2 = Page("retention", "Retention")
    header(p2, "Retention", "Who comes back, and how fast. New customers only: people already buying in "
                            "December 2009 were not new, the data just starts there.")
    p2.card(24, 84, 232, 96, "Month 1 Retention", "Buy again the next month")
    p2.matrix(24, 196, 760, 500, col(F, "Cohort Label"), col(F, "Months Since First Purchase"),
              mea("New Customer Retention"), "Share of each month's new customers buying again, by month since "
                                             "their first order", gradient=True)
    p2.chart(800, 84, 456, 300, "lineChart", col(F, "Months Since First Purchase"), [mea("Retention First Year")],
             "First year retention, 2010 vs 2011 cohorts", series=col(F, "Cohort Year"), colours=None)
    p2.text(800, 400, 456, 296, [
        [("Read it like this", 12, True, INK)],
        [("Each row is the customers whose first order fell in that month. Customers first seen in September to "
          "November are Christmas buyers: quiet through spring and summer, back the next autumn, which shows as a "
          "diagonal in the matrix.", 10, False, INK_2)],
        [("The last cohorts are short because the data ends in December 2011. December 2011 itself is left out "
          "because it only has nine days.", 10, False, INK_2)]])

    p3 = Page("customer_value", "Customer value")
    header(p3, "Customer value", "Who is worth what over the next six months, and who might already be gone. "
                                 "Values come from the MBG/NBD and Gamma-Gamma models.")
    p3.slicer(24, 84, 220, 56, col(C, "Segment"), "Segment")
    p3.slicer(256, 84, 220, 56, col(C, "P Alive Band"), "Chance still active")
    p3.card(488, 84, 232, 96, "Planning Value 26w", "Next 6 months, planning case")
    p3.card(736, 84, 232, 96, "Value At Risk 26w", "Value at risk if nothing changes")
    p3.card(984, 84, 272, 96, "Avg P Alive", "Average chance still active")
    p3.chart(24, 196, 500, 260, "clusteredColumnChart", col(C, "Value Decile"), [mea("Planning Value 26w")],
             "Planning value by decile, 1 = top 10% of customers", sort=col(C, "Value Decile"), colours=[BLUE],
             labels=True, legend=False, categorical=True)
    p3.matrix(24, 472, 500, 224, col(C, "Segment"), col(C, "P Alive Band"), mea("Customers"),
              "Customers by segment and chance of still being active", gradient=False)
    p3.table(540, 196, 716, 500,
             [col(C, "Customer ID"), col(C, "Country"), col(C, "Segment"), mea("Top 25 Planning Value"),
              col(C, "P Alive"), col(C, "Days Since Last Purchase")],
             "Top 25 accounts by expected value", sort=(mea("Top 25 Planning Value"), True),
             names=["Customer", "Country", "Segment", "Next 6 months", "Chance active", "Days since last order"])

    p4 = Page("budget_plan", "Budget plan")
    header(p4, "Retention budget plan", "Where the retention money should go, and where it should not. "
                                        "Campaign response is assumed, not measured: see config.toml.")
    p4.slicer(24, 84, 220, 56, col("Margin", "Margin"), "Gross margin (default 35%)", single=True)
    for i, (m, lab) in enumerate([("Planned Spend", "Spend that pays back"), ("Accounts Targeted", "Accounts targeted"),
                                  ("Expected Extra Margin", "Expected extra margin"),
                                  ("Margin Back per Pound", "Margin back per £1")]):
        p4.card(256 + i * 252, 84, 236, 96, m, lab)
    p4.chart(24, 196, 400, 250, "clusteredBarChart", col(S, "Segment"), [mea("Target Spend")],
             "Planned spend by segment", sort=col(S, "Segment"), colours=[BLUE], labels=True, legend=False,
             wide_labels=True, whole_labels=True)
    p4.chart(440, 196, 400, 250, "lineChart", col("plan_budget_curve", "Budget"), [mea("Budget Profit")],
             "Profit after spend at each budget (best placement)", colours=[BLUE], legend=False)
    p4.chart(856, 196, 400, 250, "clusteredBarChart", col("plan_strategies", "Strategy"),
             [mea("Strategy Median Profit")], "Profit by plan, median across 2,000 assumption runs",
             sort=(mea("Strategy Median Profit"), True), colours=[BLUE], labels=True, legend=False,
             wide_labels=True)
    p4.table(24, 462, 1232, 234,
             [col(C, "Customer ID"), col(C, "Segment"), col(C, "Recommended Action"), mea("Target Spend"),
              mea("Target Extra Revenue"), col(C, "P Alive")],
             "Target list", sort=(mea("Target Extra Revenue"), True),
             names=["Customer", "Segment", "Action", "Spend", "Expected extra revenue", "Chance active"])
    return [p1, p2, p3, p4]


THEME_FILE = "cve_theme.json"


def report() -> dict:
    ps = pages()
    return {
        "config": json.dumps({"version": "5.66", "themeCollection": {"customTheme": {
                                  "name": THEME_FILE, "version": {"visual": "2.13.0", "report": "3.4.0", "page": "2.3.1"},
                                  "type": 1}},
                              "activeSectionIndex": 0,
                              "defaultDrillFilterOtherVisuals": True,
                              "settings": {"useNewFilterPaneExperience": True, "allowChangeFilterTypes": True,
                                           "useStylableVisualContainerHeader": True, "exportDataMode": 1}}),
        "layoutOptimization": 0,
        "resourcePackages": [{"resourcePackage": {"disabled": False, "name": "RegisteredResources", "type": 1,
                                                  "items": [{"name": THEME_FILE, "path": THEME_FILE, "type": 201}]}}],
        "theme": THEME_FILE,
        "sections": [p.section(i) for i, p in enumerate(ps)],
    }


# ---------------------------------------------------------------- write

def _platform(kind: str) -> dict:
    return {"$schema": "https://developer.microsoft.com/json-schemas/fabric/gitIntegration/platformProperties/2.0.0/schema.json",
            "metadata": {"type": kind, "displayName": NAME},
            "config": {"version": "2.0", "logicalId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{NAME}-{kind}"))}}


def write(out: Path, data_folder: str) -> list[Path]:
    if not data_folder.endswith(("\\", "/")):
        data_folder += "\\" if "\\" in data_folder else "/"
    sm, rp = out / f"{NAME}.SemanticModel", out / f"{NAME}.Report"
    files = {
        out / f"{NAME}.pbip": {"version": "1.0", "artifacts": [{"report": {"path": f"{NAME}.Report"}}],
                               "settings": {"enableAutoRecovery": True}},
        sm / "definition.pbism": {"version": "1.0", "settings": {}},
        sm / "model.bim": model(data_folder),
        sm / ".platform": _platform("SemanticModel"),
        rp / "definition.pbir": {"version": "1.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"},
                                                                         "byConnection": None}},
        rp / "report.json": report(),
        rp / ".platform": _platform("Report"),
        rp / "StaticResources" / "RegisteredResources" / THEME_FILE: theme(),
    }
    written = []
    for p, content in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(p)
    (out / ".gitignore").write_text("**/.pbi/localSettings.json\n**/.pbi/cache.abf\n", encoding="utf-8")
    return written


def default_folder() -> str:
    p = ROOT / "outputs" / "powerbi"
    return str(PureWindowsPath(p)) if ":" in str(p) else str(p)
