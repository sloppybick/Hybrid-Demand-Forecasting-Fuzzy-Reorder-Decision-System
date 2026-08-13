"""Generate the sample dataset used by the app and tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.synthetic_data import generate_sales, generate_inventory_master

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sales = generate_sales()
    inventory = generate_inventory_master()

    sales_path = OUT_DIR / "sales.csv"
    inv_path = OUT_DIR / "inventory.csv"
    sales.to_csv(sales_path, index=False)
    inventory.to_csv(inv_path, index=False)
    print(f"Wrote {len(sales)} sales rows -> {sales_path}")
    print(f"Wrote {len(inventory)} inventory rows -> {inv_path}")


if __name__ == "__main__":
    main()
