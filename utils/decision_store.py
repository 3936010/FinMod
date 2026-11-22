"""
utils/decision_store.py — append-only-ish JSONL store for DecisionInput records.

One JSON object per line, keyed by (ticker, as_of_date). Re-running the logger
on the same day replaces that day's record instead of duplicating it. Writes go
through a temp file + os.replace so a crash (or a Jetson power cut) never leaves
a half-written file.

At ~10 tickers × 252 days/year the file stays a few MB, so rewriting it on each
upsert is fine. Move to Parquet/SQLite if the watchlist grows a lot.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from utils.data_models import DecisionInput, DECISION_SCHEMA_VERSION

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "decision_inputs"
DEFAULT_PATH = DATA_DIR / f"{DECISION_SCHEMA_VERSION.replace('.', '_')}.jsonl"


class DecisionStore:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)

    def load(self) -> list[DecisionInput]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(DecisionInput.model_validate_json(line))
                except Exception as e:
                    print(f"Warning: skipping bad record on line {n} of {self.path}: {e}")
        return records

    def _write(self, records: Iterable[DecisionInput]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for r in sorted(records, key=lambda r: (r.as_of_date, r.ticker)):
                f.write(r.model_dump_json() + "\n")
        os.replace(tmp, self.path)

    def upsert(self, new: Iterable[DecisionInput]) -> int:
        by_key = {r.key: r for r in self.load()}
        new = list(new)
        for r in new:
            old = by_key.get(r.key)
            if old is not None and old.label.resolved_on and not r.label.resolved_on:
                r.label = old.label          # never lose a resolved label on re-run
            by_key[r.key] = r
        self._write(by_key.values())
        return len(new)

    def replace_all(self, records: Iterable[DecisionInput]):
        self._write(records)

    def to_frame(self) -> pd.DataFrame:
        """Training view: one row per record — key columns, flat features
        (feature.*), label columns (label.*) and provenance flags."""
        rows = []
        for r in self.load():
            row = {"ticker": r.ticker, "as_of_date": r.as_of_date}
            row.update({f"feature.{k}": v for k, v in r.feature_vector().items()})
            row.update({f"label.{k}": v for k, v in r.label.model_dump().items()})
            row.update({f"prov.{k}": v for k, v in r.provenance.model_dump().items() if k != "errors"})
            rows.append(row)
        return pd.DataFrame(rows)
