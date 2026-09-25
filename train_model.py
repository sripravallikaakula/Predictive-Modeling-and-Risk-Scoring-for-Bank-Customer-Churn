from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from .feature_engineering import add_engineered_features
except ImportError:
    from feature_engineering import add_engineered_features


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "European_Bank.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "Exited"
DROP_COLUMNS = ["CustomerId", "Surname", "Year"]

CATEGORICAL_FEATURES = ["Geography", "Gender"]
NUMERIC_FEATURES = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "BalanceSalaryRatio",
    "ProductDensity",
    "EngagementProductInteraction",
    "AgeTenureInteraction",
]


def make_preprocessor():
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "classification_report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    # Drop non-informative / non-predictive identifiers.
    feature_df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    feature_df = add_engineered_features(feature_df)

    X = feature_df.drop(columns=[TARGET])
    y = feature_df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    results = {}
    fitted = {}

    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        fitted[name] = pipeline

        print(
            f"{name}: "
            f"Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    # Choose the model with the highest ROC-AUC.
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model = fitted[best_name]

    joblib.dump(best_model, MODEL_DIR / "churn_model.joblib")

    metadata = {
        "best_model": best_name,
        "selection_metric": "roc_auc",
        "threshold": 0.50,
        "features": CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        "results": results,
    }

    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Create business-friendly feature importance where supported.
    try:
        preprocessor = best_model.named_steps["preprocessor"]
        estimator = best_model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(estimator, "feature_importances_"):
            importance_values = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importance_values = np.abs(estimator.coef_[0])
        else:
            importance_values = None

        if importance_values is not None:
            importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": importance_values,
                }
            ).sort_values("importance", ascending=False)

            importance.to_csv(
                MODEL_DIR / "feature_importance.csv",
                index=False,
            )
    except Exception as exc:
        print("Feature importance export skipped:", exc)

    print(f"\nBest model: {best_name}")
    print(f"Saved to: {MODEL_DIR / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
