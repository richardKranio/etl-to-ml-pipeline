"""Silver layer: Data contracts validation, deduplication, and type normalization."""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_CUSTOMER_COLS = {"customer_id", "age", "segment", "country", "signup_date", "churned"}
REQUIRED_TX_COLS = {"tx_id", "customer_id", "tx_date", "amount"}


def validate_schema(df: pd.DataFrame, required_cols: set[str], dataset_name: str) -> None:
    """Validate that required columns exist and dataset is non-empty."""
    if df.empty:
        raise ValueError(f"Data Contract violation: '{dataset_name}' is empty.")
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Data Contract violation in '{dataset_name}': missing columns {sorted(missing)}.")


def clean_bronze_to_silver(
    customers_path: Path,
    transactions_path: Path,
    silver_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean raw Bronze CSVs into validated, deduplicated Silver Parquet tables."""
    customers_df = pd.read_csv(customers_path)
    transactions_df = pd.read_csv(transactions_path)

    validate_schema(customers_df, REQUIRED_CUSTOMER_COLS, "customers_raw")
    validate_schema(transactions_df, REQUIRED_TX_COLS, "transactions_raw")

    # Deduplicate entities and normalize timestamps to UTC
    customers_clean = customers_df.drop_duplicates(subset=["customer_id"]).copy()
    customers_clean["signup_date"] = pd.to_datetime(customers_clean["signup_date"], utc=True)
    customers_clean["churned"] = customers_clean["churned"].astype(int)

    initial_tx = len(transactions_df)
    tx_clean = transactions_df.drop_duplicates(subset=["tx_id"]).copy()
    tx_clean["tx_date"] = pd.to_datetime(tx_clean["tx_date"], utc=True)
    tx_clean["amount"] = pd.to_numeric(tx_clean["amount"], errors="coerce")
    tx_clean = tx_clean.dropna(subset=["customer_id", "tx_date", "amount"])
    tx_clean = tx_clean[tx_clean["amount"] >= 0]

    silver_dir.mkdir(parents=True, exist_ok=True)
    customers_clean.to_parquet(silver_dir / "customers_silver.parquet", index=False)
    tx_clean.to_parquet(silver_dir / "transactions_silver.parquet", index=False)

    logger.info(
        "Silver cleaned: %d customers, %d transactions (%d duplicates removed)",
        len(customers_clean),
        len(tx_clean),
        initial_tx - len(tx_clean),
    )
    return customers_clean, tx_clean
