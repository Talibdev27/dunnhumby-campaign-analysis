"""Load the dunnhumby Complete Journey tables, caching each CSV as parquet on first read."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT  # the Kaggle CSVs sit in the project root
CACHE_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

TABLES = {
    "transactions": "transaction_data.csv",
    "demographics": "hh_demographic.csv",
    "products": "product.csv",
    "campaigns": "campaign_desc.csv",
    "campaign_households": "campaign_table.csv",
    "coupons": "coupon.csv",
    "redemptions": "coupon_redempt.csv",
    "causal": "causal_data.csv",
}

DTYPES = {
    "transactions": {
        "household_key": "int32", "BASKET_ID": "int64", "DAY": "int16",
        "PRODUCT_ID": "int32", "QUANTITY": "int32", "SALES_VALUE": "float32",
        "STORE_ID": "int32", "RETAIL_DISC": "float32", "TRANS_TIME": "int16",
        "WEEK_NO": "int8", "COUPON_DISC": "float32", "COUPON_MATCH_DISC": "float32",
    },
    "causal": {
        "PRODUCT_ID": "int32", "STORE_ID": "int32", "WEEK_NO": "int8",
        "display": "category", "mailer": "category",
    },
}


def load(name: str, raw_dir: Path = RAW_DIR, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Return table `name` with lower-case columns. Builds the parquet cache if missing."""
    if name not in TABLES:
        raise KeyError(f"unknown table {name!r}; choose from {sorted(TABLES)}")
    cache = cache_dir / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(raw_dir / TABLES[name], dtype=DTYPES.get(name))
    df.columns = df.columns.str.lower()
    if name == "campaigns":
        df = df.rename(columns={"description": "campaign_type"})
    if name == "campaign_households":
        df = df.rename(columns={"description": "campaign_type"})
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df
