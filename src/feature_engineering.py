"""Gold layer: Leakage-safe RFM Feature Engineering for ML models."""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def build_rfm_features(
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    cutoff_date: str,
) -> pd.DataFrame:
    """Build RFM features strictly up to cutoff_date to prevent temporal Data Leakage."""
    cutoff = pd.to_datetime(cutoff_date, utc=True)
    tx_dates = pd.to_datetime(transactions_df["tx_date"], utc=True)

    # Filter strictly on or before cutoff_date so future events never leak into training
    tx_valid = transactions_df[tx_dates <= cutoff].copy()
    tx_valid["tx_date"] = pd.to_datetime(tx_valid["tx_date"], utc=True)

    window_90d_start = cutoff - pd.Timedelta(days=90)
    tx_90d = tx_valid[tx_valid["tx_date"] >= window_90d_start]

    recency_monetary = (
        tx_valid.groupby("customer_id")
        .agg(
            recency_days=("tx_date", lambda s: (cutoff - s.max()).days),
            monetary_avg=("amount", "mean"),
            monetary_total=("amount", "sum"),
        )
        .reset_index()
    )

    freq_90d = (
        tx_90d.groupby("customer_id")
        .agg(tx_frequency_90d=("tx_id", "count"))
        .reset_index()
    )

    dataset = customers_df.merge(recency_monetary, on="customer_id", how="left")
    dataset = dataset.merge(freq_90d, on="customer_id", how="left")

    # Explicit inactivity defaults for customers without transactions prior to cutoff
    dataset["recency_days"] = dataset["recency_days"].fillna(999).astype(float)
    for col in ("tx_frequency_90d", "monetary_avg", "monetary_total"):
        dataset[col] = dataset[col].fillna(0.0).astype(float)

    return dataset


def save_gold_features(
    customers_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    cutoff_date: str,
    gold_dir: Path,
) -> pd.DataFrame:
    """Persist the ML-ready Gold feature table in Parquet format."""
    gold_df = build_rfm_features(customers_df, transactions_df, cutoff_date)
    gold_dir.mkdir(parents=True, exist_ok=True)
    output_path = gold_dir / "churn_features_gold.parquet"
    gold_df.to_parquet(output_path, index=False)
    logger.info("Gold ML-ready dataset saved to %s (%d rows)", output_path, len(gold_df))
    return gold_df
