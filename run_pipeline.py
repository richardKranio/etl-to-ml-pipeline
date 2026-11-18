"""End-to-end orchestrator: Bronze -> Silver -> Gold RFM -> ML Pipeline."""

import logging
from pathlib import Path
from src.data_generator import generate_synthetic_bronze_data
from src.etl_cleaning import clean_bronze_to_silver
from src.feature_engineering import save_gold_features
from src.train_model import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


def main() -> None:
    """Execute the full ETL-to-ML pipeline locally."""
    base_dir = Path(__file__).resolve().parent
    bronze_dir = base_dir / "data" / "bronze"
    silver_dir = base_dir / "data" / "silver"
    gold_dir = base_dir / "data" / "gold"
    artifacts_dir = base_dir / "artifacts"

    # 1. Bronze: Extract synthetic raw data
    customers_raw, tx_raw = generate_synthetic_bronze_data(bronze_dir, n_customers=800, seed=42)

    # 2. Silver: Data contracts, deduplication, and UTC normalization
    customers_silver, tx_silver = clean_bronze_to_silver(customers_raw, tx_raw, silver_dir)

    # 3. Gold: Leakage-safe RFM Feature Engineering up to cutoff_date
    gold_df = save_gold_features(
        customers_silver,
        tx_silver,
        cutoff_date="2026-03-31",
        gold_dir=gold_dir,
    )

    # 4. ML: Unified preprocessing + RandomForest training
    metrics = train_and_evaluate(gold_df, artifacts_dir=artifacts_dir)
    logging.getLogger(__name__).info("Pipeline completed successfully: %s", metrics)


if __name__ == "__main__":
    main()
