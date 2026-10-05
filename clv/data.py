"""Getting the raw file and reading it into one tidy table.

The workbook has two sheets (Dec 2009 to Dec 2010 and Dec 2010 to Dec 2011).
Reading 1M rows from Excel is slow, so the result is cached as parquet next to
the raw file and only rebuilt when the workbook changes.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from clv.config import ROOT

COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def raw_path(cfg: dict) -> Path:
    return ROOT / cfg["data"]["raw_file"]


def fetch(cfg: dict, force: bool = False) -> Path:
    """Download the zip from UCI and extract the workbook into data/raw/."""
    target = raw_path(cfg)
    if target.exists() and not force:
        print(f"already have {target.relative_to(ROOT)}, use --force to download again")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["data"]["source_url"]
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        target.write_bytes(zf.read(name))
    print(f"saved {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB), sha256 {sha256(target)[:16]}")
    return target


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_code(s: pd.Series) -> pd.Series:
    """Excel stores numeric looking codes as numbers (85123 or 85123.0) and the
    rest as text (85123A). Turn both into the same clean string."""

    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v).strip()

    return s.map(one).astype("string")


def tidy(frame: pd.DataFrame, sheet: str) -> pd.DataFrame:
    """Rename columns and fix types. No rows are removed here."""
    df = frame.rename(columns=COLUMNS)[list(COLUMNS.values())].copy()
    df["invoice"] = _as_code(df["invoice"])
    df["stock_code"] = _as_code(df["stock_code"])
    df["customer_id"] = _as_code(df["customer_id"])
    df["description"] = df["description"].astype("string").str.strip()
    df["country"] = df["country"].astype("string").str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"]).astype("int64")
    df["price"] = pd.to_numeric(df["price"]).astype("float64")
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df.insert(0, "sheet", sheet)
    df.insert(1, "row_in_sheet", np.arange(len(df), dtype="int64"))
    return df


def _read_excel(path: Path) -> dict[str, pd.DataFrame]:
    try:
        return pd.read_excel(path, sheet_name=None, engine="calamine")
    except (ImportError, ValueError):
        print("python-calamine not installed, falling back to openpyxl (a few minutes)")
        return pd.read_excel(path, sheet_name=None, engine="openpyxl")


def load_raw(cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """Both sheets stacked, typed, with nothing removed."""
    path = raw_path(cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(ROOT)} is missing. Run `python -m clv fetch` first, "
            "or download the zip from UCI and put online_retail_II.xlsx in data/raw/."
        )
    stamp = f"{path.stat().st_size}-{int(path.stat().st_mtime)}"
    cache = path.parent.parent / "interim" / "raw.parquet"
    stamp_file = cache.with_suffix(".stamp")
    if cache.exists() and not refresh and stamp_file.exists() and stamp_file.read_text() == stamp:
        return pd.read_parquet(cache)

    print("reading the workbook (only happens once)")
    sheets = _read_excel(path)
    expected = cfg["data"]["expected_rows"]
    for name, n in expected.items():
        if name not in sheets:
            raise ValueError(f"sheet {name!r} not found, got {list(sheets)}")
        if len(sheets[name]) != n:
            raise ValueError(f"sheet {name!r} has {len(sheets[name]):,} rows, expected {n:,}. Re-download the file.")
    df = pd.concat([tidy(sheets[name], name) for name in expected], ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    stamp_file.write_text(stamp)
    return df
