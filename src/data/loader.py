"""
Loads raw sales/inventory data and converts it into the project's canonical
internal schema, regardless of the source dataset's original column names.

Canonical sales schema:
    date (datetime64) | sku (str) | quantity (float)

Canonical inventory schema:
    sku | current_stock | lead_time | minimum_order_quantity | unit_cost
"""
from __future__ import annotations

import pandas as pd


CANONICAL_SALES_COLUMNS = ["date", "sku", "quantity"]
CANONICAL_INVENTORY_COLUMNS = [
    "sku",
    "current_stock",
    "lead_time",
    "minimum_order_quantity",
    "unit_cost",
]


def load_sales_csv(
    path: str,
    date_col: str = "date",
    sku_col: str = "sku",
    quantity_col: str = "quantity_sold",
) -> pd.DataFrame:
    """Load a raw sales CSV and map it onto the canonical schema.

    Parameters let the caller adapt to whatever column names the source
    dataset happens to use, so external schemas never leak past this layer.
    """
    df = pd.read_csv(path)
    missing = {date_col, sku_col, quantity_col} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing expected columns: {missing}")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]),
            "sku": df[sku_col].astype(str),
            "quantity": pd.to_numeric(df[quantity_col], errors="coerce"),
        }
    )
    return out[CANONICAL_SALES_COLUMNS]


def load_inventory_csv(path: str) -> pd.DataFrame:
    """Load inventory master data (current stock, lead time, MOQ, unit cost)."""
    df = pd.read_csv(path)
    missing = set(CANONICAL_INVENTORY_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Inventory CSV missing expected columns: {missing}")
    df = df.copy()
    df["sku"] = df["sku"].astype(str)
    for col in ["current_stock", "lead_time", "minimum_order_quantity", "unit_cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[CANONICAL_INVENTORY_COLUMNS]


def dataframe_to_canonical_sales(
    df: pd.DataFrame, date_col: str, sku_col: str, quantity_col: str
) -> pd.DataFrame:
    """Same mapping as load_sales_csv but from an in-memory DataFrame."""
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col]),
            "sku": df[sku_col].astype(str),
            "quantity": pd.to_numeric(df[quantity_col], errors="coerce"),
        }
    )
    return out[CANONICAL_SALES_COLUMNS]
