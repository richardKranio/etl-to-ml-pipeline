"""Synthetic data generator simulating raw Bronze layer ingestion."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_synthetic_bronze_data(
    output_dir: Path,
    n_customers: int = 800,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Generate synthetic customer and transaction datasets with realistic data issues.

    Includes duplicate transactions, missing demographic fields, and events spanning
    before and after the feature cutoff date to demonstrate leakage-safe processing.
    """
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    segments = rng.choice(["Retail", "Premium", "PyME", None], size=n_customers, p=[0.5, 0.25, 0.2, 0.05])
    countries = rng.choice(["CO", "DO", "CL", "MX"], size=n_customers, p=[0.4, 0.3, 0.15, 0.15])
    ages = rng.integers(20, 70, size=n_customers).astype(float)
    ages[rng.random(n_customers) < 0.06] = np.nan

    # Simulate behavioral churn signal: inactive customers churn more often
    activity_score = rng.uniform(0.0, 1.0, size=n_customers)
    churn_prob = np.clip(0.75 - 0.65 * activity_score + rng.normal(0, 0.08, n_customers), 0.02, 0.95)
    churned = rng.binomial(1, churn_prob)

    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "segment": segments,
        "country": countries,
        "signup_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 180, n_customers), unit="D"),
        "churned": churned,
    })

    # Generate transactions between 2025-07-01 and 2026-04-30 (cutoff will be 2026-03-31)
    tx_rows = []
    tx_counter = 1
    for cid, score, is_churn in zip(customer_ids, activity_score, churned):
        n_tx = int(rng.poisson(lam=12 * score + 1))
        for _ in range(n_tx):
            # Active users transact closer to cutoff; churned users stopped earlier
            max_day = 265 if is_churn else 300
            day_offset = int(rng.integers(0, max_day))
            tx_date = pd.Timestamp("2025-07-01") + pd.Timedelta(days=day_offset)
            amount = round(float(rng.gamma(shape=2.5, scale=45.0 * (0.5 + score))), 2)
            tx_rows.append({
                "tx_id": f"TX_{tx_counter:07d}",
                "customer_id": cid,
                "tx_date": tx_date.isoformat(),
                "amount": amount,
            })
            tx_counter += 1

    transactions_df = pd.DataFrame(tx_rows)
    # Inject ~2% duplicate webhook deliveries to test Silver deduplication
    duplicates = transactions_df.sample(frac=0.02, random_state=seed)
    transactions_df = pd.concat([transactions_df, duplicates], ignore_index=True)

    customers_path = output_dir / "customers_raw.csv"
    transactions_path = output_dir / "transactions_raw.csv"
    customers_df.to_csv(customers_path, index=False)
    transactions_df.to_csv(transactions_path, index=False)

    logger.info("Bronze generated: %d customers, %d raw transactions", len(customers_df), len(transactions_df))
    return customers_path, transactions_path
