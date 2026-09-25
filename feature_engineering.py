import pandas as pd
import numpy as np


RAW_FEATURES = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the project-required engineered features.

    Derived features:
    1. Balance-to-Salary ratio
    2. Product density indicator
    3. Engagement-product interaction
    4. Age-tenure interaction
    """
    out = df.copy()

    # Avoid division by zero.
    salary = out["EstimatedSalary"].replace(0, np.nan)
    out["BalanceSalaryRatio"] = (out["Balance"] / salary).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    # Products used relative to tenure. +1 avoids division by zero.
    out["ProductDensity"] = out["NumOfProducts"] / (out["Tenure"] + 1.0)

    # Active customers holding more products have stronger engagement.
    out["EngagementProductInteraction"] = (
        out["IsActiveMember"] * out["NumOfProducts"]
    )

    # Captures interaction between age and relationship length.
    out["AgeTenureInteraction"] = out["Age"] * out["Tenure"]

    return out
