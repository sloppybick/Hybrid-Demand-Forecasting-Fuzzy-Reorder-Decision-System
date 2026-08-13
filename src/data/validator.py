"""Schema and sanity validation for canonical sales data."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationReport:
    n_rows: int
    n_skus: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    n_missing_quantity: int
    n_negative_quantity: int
    n_duplicate_rows: int
    warnings: list[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        return not self.warnings

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            f"rows={self.n_rows} skus={self.n_skus} "
            f"range=[{self.date_min.date()} -> {self.date_max.date()}]",
            f"missing_quantity={self.n_missing_quantity} "
            f"negative_quantity={self.n_negative_quantity} "
            f"duplicate_rows={self.n_duplicate_rows}",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def validate_sales(df: pd.DataFrame) -> ValidationReport:
    required = {"date", "sku", "quantity"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"validate_sales: missing columns {missing_cols}")

    warnings: list[str] = []

    n_missing = int(df["quantity"].isna().sum())
    if n_missing:
        warnings.append(f"{n_missing} rows have missing quantity")

    n_negative = int((df["quantity"] < 0).sum())
    if n_negative:
        warnings.append(f"{n_negative} rows have negative quantity")

    n_dupes = int(df.duplicated(subset=["date", "sku"]).sum())
    if n_dupes:
        warnings.append(f"{n_dupes} duplicate (date, sku) rows")

    if df.empty:
        warnings.append("dataframe is empty")
        date_min = date_max = pd.NaT
    else:
        date_min, date_max = df["date"].min(), df["date"].max()

    return ValidationReport(
        n_rows=len(df),
        n_skus=df["sku"].nunique() if not df.empty else 0,
        date_min=date_min,
        date_max=date_max,
        n_missing_quantity=n_missing,
        n_negative_quantity=n_negative,
        n_duplicate_rows=n_dupes,
        warnings=warnings,
    )
