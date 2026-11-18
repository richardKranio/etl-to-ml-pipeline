"""Unit tests for Data Contracts, Silver deduplication, Leakage-safe RFM, and ML Pipeline."""

import pandas as pd
import pytest
from src.etl_cleaning import validate_schema
from src.feature_engineering import build_rfm_features
from src.train_model import build_ml_pipeline


def test_validate_schema_raises_on_missing_columns() -> None:
    """Data contract must fail fast when required columns are missing."""
    df = pd.DataFrame({"customer_id": ["C1"], "age": [30]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(df, {"customer_id", "age", "segment"}, "customers_raw")


def test_build_rfm_features_prevents_temporal_leakage() -> None:
    """Transactions after cutoff_date must be completely ignored in RFM aggregation."""
    customers = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "age": [35.0, 42.0],
        "segment": ["Retail", "Premium"],
        "country": ["CO", "DO"],
        "churned": [0, 1],
    })
    transactions = pd.DataFrame({
        "tx_id": ["T1", "T2", "T_FUTURE"],
        "customer_id": ["C1", "C1", "C1"],
        "tx_date": ["2026-03-01", "2026-03-15", "2026-04-10"],
        "amount": [100.0, 200.0, 9999.0],
    })

    result = build_rfm_features(customers, transactions, cutoff_date="2026-03-31")
    c1 = result.loc[result["customer_id"] == "C1"].iloc[0]
    c2 = result.loc[result["customer_id"] == "C2"].iloc[0]

    # T_FUTURE (9999.0) must not be included
    assert c1["monetary_total"] == pytest.approx(300.0)
    assert c1["tx_frequency_90d"] == pytest.approx(2.0)
    assert c1["recency_days"] == pytest.approx(16.0)

    # C2 had no transactions -> explicit inactivity defaults
    assert c2["recency_days"] == pytest.approx(999.0)
    assert c2["monetary_total"] == pytest.approx(0.0)


def test_ml_pipeline_handles_nulls_and_unseen_categories() -> None:
    """Unified sklearn Pipeline must impute NaNs and ignore unseen categories at inference."""
    train_df = pd.DataFrame({
        "age": [25.0, None, 45.0, 52.0, 31.0, 40.0],
        "recency_days": [5.0, 120.0, 10.0, 200.0, 12.0, 180.0],
        "tx_frequency_90d": [10.0, 1.0, 8.0, 0.0, 6.0, 1.0],
        "monetary_avg": [50.0, 20.0, 80.0, 0.0, 60.0, 15.0],
        "monetary_total": [500.0, 20.0, 640.0, 0.0, 360.0, 15.0],
        "segment": ["Retail", None, "Premium", "PyME", "Retail", "PyME"],
        "country": ["CO", "DO", "MX", "CL", "CO", "DO"],
    })
    y_train = pd.Series([0, 1, 0, 1, 0, 1])

    pipeline = build_ml_pipeline(random_state=42)
    pipeline.fit(train_df, y_train)

    inference_df = pd.DataFrame({
        "age": [None],
        "recency_days": [15.0],
        "tx_frequency_90d": [4.0],
        "monetary_avg": [45.0],
        "monetary_total": [180.0],
        "segment": ["VIP_UNSEEN"],
        "country": ["ES_UNSEEN"],
    })
    probs = pipeline.predict_proba(inference_df)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0
