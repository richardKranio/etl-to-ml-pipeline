"""Reproducible ML training pipeline preventing Data Leakage and Training-Serving Skew."""

import logging
import pickle
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["age", "recency_days", "tx_frequency_90d", "monetary_avg", "monetary_total"]
CATEGORICAL_FEATURES = ["segment", "country"]
TARGET_COL = "churned"


def build_ml_pipeline(random_state: int = 42) -> Pipeline:
    """Create a unified sklearn Pipeline encapsulating preprocessing and estimator."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=random_state)),
    ])


def train_and_evaluate(
    gold_df: pd.DataFrame,
    artifacts_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Split dataset before fitting any transformer, train model, and persist artifact."""
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = gold_df[features]
    y = gold_df[TARGET_COL]

    # Golden Rule: Split BEFORE calling fit_transform to prevent Data Leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pipeline = build_ml_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = float(roc_auc_score(y_test, y_prob))
    accuracy = float((y_pred == y_test).mean())

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "churn_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    logger.info("Model trained and saved to %s | ROC-AUC: %.4f | Accuracy: %.4f", model_path, roc_auc, accuracy)
    logger.info("\n%s", classification_report(y_test, y_pred, digits=4))
    return {"roc_auc": roc_auc, "accuracy": accuracy}
