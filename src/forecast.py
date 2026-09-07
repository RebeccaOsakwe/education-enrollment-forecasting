from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Project paths and settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_enrollment_funnel.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FORECAST_HORIZONS = [12, 8, 4]

# Full multivariable model.
MODEL_FEATURES = [
    "enrollment_target",
    "applications_cumulative",
    "admits_cumulative",
    "deposits_cumulative",
    "aid_offers_cumulative",
    "deposit_rate_to_date",
    "applications_yoy_change",
    "deposits_yoy_change",
]

# Narrower operational model focused on deposit-related signals.
OPERATIONAL_FEATURES = [
    "enrollment_target",
    "deposits_cumulative",
    "deposit_rate_to_date",
    "deposits_yoy_change",
]


def load_data():
    """Load the validated synthetic enrollment dataset."""

    return pd.read_csv(DATA_PATH)


def prepare_output_directory():
    """Create the output directory if needed."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_horizon_snapshot(df, weeks_to_census):
    """
    Return one observation per enrollment cycle at a specified
    forecast horizon.
    """

    snapshot = (
        df[df["weeks_to_census"] == weeks_to_census]
        .copy()
        .sort_values("cycle_year")
        .reset_index(drop=True)
    )

    return snapshot


def prior_year_forecast(train, test_row):
    """
    Baseline forecast:
    assume final enrollment will equal the prior year's final enrollment.
    """

    prior_year = int(test_row["cycle_year"]) - 1

    prior = train[
        train["cycle_year"] == prior_year
    ]

    if prior.empty:
        return np.nan

    return float(
        prior["final_enrollment"].iloc[0]
    )


def deposit_conversion_forecast(train, test_row):
    """
    Simple operational forecast.

    Estimate the historical relationship between deposits observed
    at the forecast horizon and eventual final enrollment, then apply
    that average conversion factor to the current cycle.
    """

    valid_train = train[
        train["deposits_cumulative"] > 0
    ].copy()

    if valid_train.empty:
        return np.nan

    conversion_rates = (
        valid_train["final_enrollment"]
        / valid_train["deposits_cumulative"]
    )

    historical_conversion = conversion_rates.mean()

    return float(
        test_row["deposits_cumulative"]
        * historical_conversion
    )


def build_ridge_model():
    """
    Create a regularized linear model.

    Ridge regression is used instead of a more complex machine-learning
    model because the number of independent historical cycles is small.
    """

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def prepare_model_data(df):
    """
    Prepare model features while handling year-over-year features
    that are unavailable for the first cycle.
    """

    model_df = df.copy()

    yoy_columns = [
        "applications_yoy_change",
        "deposits_yoy_change",
    ]

    model_df[yoy_columns] = (
        model_df[yoy_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return model_df


def ridge_forecast(train, test_row, features):
    """
    Fit Ridge regression on historical cycles and predict
    final enrollment for one unseen cycle using the supplied
    feature set.
    """

    train = prepare_model_data(train)

    test_df = prepare_model_data(
        pd.DataFrame([test_row])
    )

    X_train = train[features]
    y_train = train["final_enrollment"]

    X_test = test_df[features]

    model = build_ridge_model()

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(X_test)[0]

    return float(prediction)


def walk_forward_validation(snapshot):
    """
    Evaluate all forecasting methods using expanding-window,
    out-of-time validation.

    Example:
        train 2018-2021 -> test 2022
        train 2018-2022 -> test 2023
        ...
    """

    evaluation_years = [
        year
        for year in snapshot["cycle_year"].unique()
        if 2022 <= year <= 2025
    ]

    results = []

    for test_year in evaluation_years:

        train = snapshot[
            snapshot["cycle_year"] < test_year
        ].copy()

        test = snapshot[
            snapshot["cycle_year"] == test_year
        ].copy()

        if test.empty:
            continue

        test_row = test.iloc[0]

        actual = float(
            test_row["final_enrollment"]
        )

        forecasts = {
            "prior_year_baseline": prior_year_forecast(
                train,
                test_row,
            ),
            "deposit_conversion": deposit_conversion_forecast(
                train,
                test_row,
            ),
            "ridge_regression": ridge_forecast(
                train,
                test_row,
                MODEL_FEATURES,
            ),
            "operational_ridge": ridge_forecast(
                train,
                test_row,
                OPERATIONAL_FEATURES,
            ),
        }

        for method, prediction in forecasts.items():

            if np.isnan(prediction):
                continue

            results.append(
                {
                    "test_year": test_year,
                    "method": method,
                    "actual_enrollment": actual,
                    "predicted_enrollment": round(
                        prediction, 1
                    ),
                    "absolute_error": round(
                        abs(actual - prediction), 1
                    ),
                }
            )

    return pd.DataFrame(results)


def summarize_validation(results):
    """
    Calculate mean absolute error for each forecasting method.
    """

    summary = (
        results
        .groupby("method", as_index=False)
        .agg(
            mae=(
                "absolute_error",
                "mean",
            ),
            median_absolute_error=(
                "absolute_error",
                "median",
            ),
            n_forecasts=(
                "absolute_error",
                "count",
            ),
        )
        .sort_values("mae")
        .reset_index(drop=True)
    )

    summary["mae"] = (
        summary["mae"].round(1)
    )

    summary["median_absolute_error"] = (
        summary["median_absolute_error"].round(1)
    )

    return summary


def forecast_2026(snapshot):
    """
    Train each method using all historical cycles through 2025,
    then forecast the unseen 2026 outcome.
    """

    train = snapshot[
        snapshot["cycle_year"] < 2026
    ].copy()

    test = snapshot[
        snapshot["cycle_year"] == 2026
    ].copy()

    if test.empty:
        raise ValueError(
            "2026 cycle was not found."
        )

    test_row = test.iloc[0]

    forecasts = {
        "prior_year_baseline": prior_year_forecast(
            train,
            test_row,
        ),
        "deposit_conversion": deposit_conversion_forecast(
            train,
            test_row,
        ),
        "ridge_regression": ridge_forecast(
            train,
            test_row,
            MODEL_FEATURES,
        ),
        "operational_ridge": ridge_forecast(
            train,
            test_row,
            OPERATIONAL_FEATURES,
        ),
    }

    results = []

    for method, prediction in forecasts.items():

        results.append(
            {
                "method": method,
                "predicted_enrollment": round(
                    prediction, 1
                ),
                "institutional_target": int(
                    test_row["enrollment_target"]
                ),
                "forecast_gap_to_target": round(
                    prediction
                    - test_row["enrollment_target"],
                    1,
                ),
            }
        )

    return pd.DataFrame(results)


def evaluate_all_horizons(df):
    """
    Run walk-forward validation and 2026 forecasting at
    12, 8, and 4 weeks before census.
    """

    validation_results = []
    validation_summaries = []
    forecasts_2026 = []

    for horizon in FORECAST_HORIZONS:

        snapshot = get_horizon_snapshot(
            df,
            horizon,
        )

        results = walk_forward_validation(
            snapshot
        )

        results[
            "weeks_to_census"
        ] = horizon

        validation_results.append(
            results
        )

        summary = summarize_validation(
            results
        )

        summary[
            "weeks_to_census"
        ] = horizon

        validation_summaries.append(
            summary
        )

        current_forecast = forecast_2026(
            snapshot
        )

        current_forecast[
            "weeks_to_census"
        ] = horizon

        forecasts_2026.append(
            current_forecast
        )

    validation_results = pd.concat(
        validation_results,
        ignore_index=True,
    )

    validation_summaries = pd.concat(
        validation_summaries,
        ignore_index=True,
    )

    forecasts_2026 = pd.concat(
        forecasts_2026,
        ignore_index=True,
    )

    return (
        validation_results,
        validation_summaries,
        forecasts_2026,
    )


def save_outputs(
    validation_results,
    validation_summary,
    forecasts_2026,
):
    """Save model evaluation outputs for later reporting."""

    validation_results.to_csv(
        OUTPUT_DIR
        / "walk_forward_validation.csv",
        index=False,
    )

    validation_summary.to_csv(
        OUTPUT_DIR
        / "forecast_method_summary.csv",
        index=False,
    )

    forecasts_2026.to_csv(
        OUTPUT_DIR
        / "forecast_2026.csv",
        index=False,
    )


def main():

    prepare_output_directory()

    df = load_data()

    (
        validation_results,
        validation_summary,
        forecasts_2026,
    ) = evaluate_all_horizons(df)

    print(
        "\nWalk-forward validation results:"
    )

    print(
        validation_results
        .sort_values(
            [
                "weeks_to_census",
                "test_year",
                "method",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .to_string(index=False)
    )

    print(
        "\nMean absolute error by method and horizon:"
    )

    print(
        validation_summary[
            [
                "weeks_to_census",
                "method",
                "mae",
                "median_absolute_error",
                "n_forecasts",
            ]
        ]
        .sort_values(
            [
                "weeks_to_census",
                "mae",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .to_string(index=False)
    )

    print(
        "\n2026 enrollment forecasts:"
    )

    print(
        forecasts_2026[
            [
                "weeks_to_census",
                "method",
                "predicted_enrollment",
                "institutional_target",
                "forecast_gap_to_target",
            ]
        ]
        .sort_values(
            [
                "weeks_to_census",
                "method",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .to_string(index=False)
    )

    save_outputs(
        validation_results,
        validation_summary,
        forecasts_2026,
    )


if __name__ == "__main__":
    main()
