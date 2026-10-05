"""Loads config.toml and resolves the folders every stage writes to."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Paths:
    """Output folders. A synthetic run writes somewhere else on purpose,
    so simulated numbers can never end up in the real report."""

    base: Path

    @property
    def data(self) -> Path:
        return self.base / "data"

    @property
    def tables(self) -> Path:
        return self.base / "tables"

    @property
    def figures(self) -> Path:
        return self.base / "figures"

    @property
    def reports(self) -> Path:
        return self.base / "reports"

    def ensure(self) -> "Paths":
        for p in (self.data, self.tables, self.figures, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def paths_for(kind: str) -> Paths:
    base = ROOT / ("outputs" if kind == "real" else "outputs_synthetic")
    return Paths(base).ensure()
