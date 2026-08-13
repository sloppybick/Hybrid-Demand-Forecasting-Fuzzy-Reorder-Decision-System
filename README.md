# Demand Forecasting + Fuzzy Reorder Hybrid

**"How much inventory will we need, and when should we reorder?"**

Prophet predicts demand. Fuzzy logic decides what to do about that demand.
This is the second-tier decision-support layer that sits upstream of
[OptiRoute](../optiroute) (how goods move) and downstream feeds the same
kind of business context as the Dynamic Pricing Agent — together forming a
small "intelligent commerce" portfolio: **forecasting → decision-making →
optimization**.

```
Historical Sales -> Data Cleaning -> Demand Forecast -> Uncertainty Analysis
    -> Fuzzy Reorder Engine -> Inventory Recommendation -> Dashboard
```

## What it does

For every SKU, the system produces:

| SKU     | Current Stock | Forecast Demand | Lead Time | Safety Stock | Reorder Point | Fuzzy Urgency | Decision      |
|---------|---------------|------------------|-----------|---------------|----------------|----------------|---------------|
| SKU-004 | 15            | 133              | 14 days   | 41            | 174            | 92.1           | Urgent Reorder|
| SKU-005 | 500           | 380              | 6 days    | 55            | 435            | 8.0            | Do Not Reorder|

Two policies are computed side by side so they can be compared:

- **Classical (s, Q) reorder point** — `ROP = lead_time_demand + z·σ√L`, the
  textbook baseline.
- **Fuzzy reorder engine** — a Mamdani fuzzy inference system that combines
  stock coverage, forecast demand, forecast *uncertainty*, and supplier lead
  time into a 0–100 urgency score and a plain-language explanation, instead
  of a single hard threshold.

## Architecture

```
src/
├── data/          loader, validator, cleaner, aggregator (EDA stats)
├── forecasting/    baselines (naive/MA/seasonal-naive), Prophet engine, backtesting/metrics
├── inventory/     safety stock, classical reorder point, order quantity sizing
├── fuzzy/         membership functions, rule base (21 rules), Mamdani inference + explanation
├── simulation/    scenario stress-testing (demand spike, supplier disruption, promo)
└── pipeline.py    the hybrid glue: forecast -> features -> fuzzy -> recommendation

app/               Streamlit dashboard (Home + 6 pages)
tests/             48 pytest tests across data/forecasting/inventory/fuzzy/pipeline
data/sample/       synthetic 5-SKU, 540-day dataset (generated, not hand-written)
configs/           forecasting.yaml, fuzzy.yaml (reference/tuning docs)
```

The forecasting engine and fuzzy engine are intentionally decoupled — the
fuzzy system only ever sees normalized inputs (`stock_ratio`, `demand_ratio`,
`uncertainty`, `lead_time_days`), never Prophet internals, so either half
could be swapped independently.

### Why normalized fuzzy inputs

Raw stock/demand units differ by orders of magnitude across SKUs (15 units
vs. 500 units). The fuzzy membership functions are defined over
dimensionless ratios instead:

- `stock_ratio = current_stock / lead_time_demand` — 1.0 means "exactly
  enough stock to cover the lead time."
- `demand_ratio = forecast_demand / historical_average_demand` — 1.0 means
  "business as usual."
- `uncertainty = (forecast_upper - forecast_lower) / forecast` — relative
  forecast spread.
- `lead_time_days` — kept in real units; it's a supplier property, not a
  demand-scale property.

This lets one rule base apply to every SKU without retuning.

## Running it

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# (optional) regenerate the synthetic sample dataset
python scripts/generate_sample_data.py

# run the test suite
pytest -q

# launch the dashboard
streamlit run app/Home.py
```

## Dashboard pages

1. **Home** — inventory health summary + color-coded reorder queue across all SKUs.
2. **Data Explorer** — historical demand, rolling volatility, weekly seasonality index, trend.
3. **Forecasting** — Prophet forecast with confidence interval vs. the three baselines.
4. **Inventory Intelligence** — the classical (s, Q) policy, worked step by step.
5. **Fuzzy Decision Engine** — drive the fuzzy system from a SKU's live data or manual sliders; shows the urgency gauge and a plain-language "Why?" explanation.
6. **Scenarios** — Normal / Demand Spike (+30%) / Supplier Disruption (lead time ×2) / Promotional Period (+50%), same SKU, same policy, side by side.
7. **Model Evaluation** — rolling-origin backtest (MAE/RMSE/MAPE/sMAPE) comparing Naive, Moving Average, Seasonal Naive, and Prophet.

## Using your own data

Swap `data/sample/sales.csv` / `inventory.csv` for real data, or point
`src/data/loader.load_sales_csv()` / `load_inventory_csv()` at your files —
the loader maps arbitrary source column names onto the canonical schema
(`date, sku, quantity` / `sku, current_stock, lead_time,
minimum_order_quantity, unit_cost`) so nothing downstream needs to change.

## Tests

```bash
pytest -q
# 48 passed
```

Covers: data cleaning (dedupe, gap-filling, outlier capping), forecasting
baselines and metrics, safety-stock/ROP/order-quantity math, the fuzzy
engine's rule coverage and monotonicity properties (e.g. increasing
uncertainty never *decreases* urgency), and full hybrid-pipeline
integration (including a scenario-correctness fix — see below).

## Known limitation / honest note

An early version capped `lead_time_demand` at the forecast horizon, which
meant a "supplier disruption (lead time ×2)" scenario could show *lower*
urgency than normal once the (unchanged) horizon fell short of the new,
longer lead time. Fixed in `src/pipeline.py` by making the effective
forecast horizon always cover at least the current lead time
(`test_supplier_disruption_scenario_increases_or_maintains_urgency` guards
against a regression). Worth knowing if you extend the lead-time range in
the Scenarios or Inventory pages.

## Deployment

Streamlit Community Cloud or Docker both work unchanged — the app makes no
disk writes outside `data/`, and `@st.cache_data` / `@st.cache_resource`
keep Prophet refits and fuzzy-engine construction cheap across reruns.
Point `app/Home.py` at the entrypoint.

## Not yet built (natural next steps, per the original spec)

- SQLite/Postgres persistence layer (currently reads CSVs each run)
- Model selection per SKU (SARIMA/XGBoost), intermittent-demand models (Croston/SBA/TSB)
- Promotion/holiday regressors in Prophet
- Supplier lead-time *distributions* instead of a point estimate
- "Decision Replay" backtest of historical fuzzy decisions vs. actual outcomes
- Multi-echelon (supplier → warehouse → DC → store) inventory modeling

These are exactly the V2–V6 / Decision-Replay extensions from the original
project brief — the current build is the complete MVP (stages 1–10) plus
Scenarios, which was the fastest path to something genuinely demonstrable.
