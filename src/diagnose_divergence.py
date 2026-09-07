from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project paths and settings
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_enrollment_funnel.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FORECAST_HORIZONS = [12, 8, 4]
RECENT_CYCLES = [2023, 2024, 2025, 2026]


def load_data():
    """Load the validated synthetic enrollment dataset."""

    return pd.read_csv(DATA_PATH)


def prepare_output_directory():
    """Create output directory if it does not exist."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_horizon_snapshot(df, weeks_to_census):
    """
    Return one row per enrollment cycle at the selected
    forecast horizon.
    """

    snapshot = (
        df[df["weeks_to_census"] == weeks_to_census]
        .copy()
        .sort_values("cycle_year")
        .reset_index(drop=True)
    )

    return snapshot


def select_diagnostic_columns(snapshot):
    """
    Keep the variables most relevant to diagnosing
    funnel divergence.
    """

    columns = [
        "cycle_year",
        "applications_cumulative",
        "applications_yoy_change",
        "admits_cumulative",
        "deposits_cumulative",
        "deposits_yoy_change",
        "deposit_rate_to_date",
        "enrollment_target",
        "final_enrollment",
        "target_gap",
    ]

    return snapshot[columns].copy()


def calculate_funnel_divergence(snapshot):
    """
    Create a descriptive divergence indicator.

    Positive values indicate applications are performing
    better year-over-year than deposits.

    Example:
        application growth = +12%
        deposit growth     = -8%

        divergence = +20 percentage points

    A large positive divergence suggests strong top-of-funnel
    activity is not translating into downstream commitment.
    """

    result = snapshot.copy()

    result["funnel_divergence"] = (
        result["applications_yoy_change"]
        - result["deposits_yoy_change"]
    )

    return result


def calculate_historical_reference(snapshot):
    """
    Calculate historical distribution of funnel divergence
    using cycles before 2026.
    """

    historical = snapshot[
        snapshot["cycle_year"] < 2026
    ].copy()

    divergence = historical[
        "funnel_divergence"
    ].replace([np.inf, -np.inf], np.nan).dropna()

    if divergence.empty:
        return {
            "mean": np.nan,
            "std": np.nan,
            "upper_warning": np.nan,
        }

    mean = divergence.mean()
    std = divergence.std(ddof=1)

    upper_warning = mean + (1.5 * std)

    return {
        "mean": mean,
        "std": std,
        "upper_warning": upper_warning,
    }


def classify_2026_risk(snapshot, reference):
    """
    Classify 2026 funnel divergence relative to historical behavior.

    This is an explanatory warning indicator, not a predictive model.
    """

    current = snapshot[
        snapshot["cycle_year"] == 2026
    ]

    if current.empty:
        raise ValueError(
            "2026 cycle not found in horizon snapshot."
        )

    divergence = float(
        current["funnel_divergence"].iloc[0]
    )

    warning_threshold = reference["upper_warning"]

    if np.isnan(warning_threshold):
        risk_status = "Not available"

    elif divergence > warning_threshold:
        risk_status = "High divergence"

    elif divergence > reference["mean"]:
        risk_status = "Elevated divergence"

    else:
        risk_status = "Within historical range"

    return {
        "funnel_divergence": divergence,
        "risk_status": risk_status,
    }


def build_horizon_diagnostic(df, weeks_to_census):
    """
    Build diagnostic tables and 2026 warning classification
    for one forecast horizon.
    """

    snapshot = get_horizon_snapshot(
        df,
        weeks_to_census,
    )

    diagnostic = select_diagnostic_columns(
        snapshot
    )

    diagnostic = calculate_funnel_divergence(
        diagnostic
    )

    reference = calculate_historical_reference(
        diagnostic
    )

    current_risk = classify_2026_risk(
        diagnostic,
        reference,
    )

    return (
        diagnostic,
        reference,
        current_risk,
    )


def build_summary_record(
    weeks_to_census,
    diagnostic,
    reference,
    current_risk,
):
    """
    Summarize the 2026 warning signal at one horizon.
    """

    current = diagnostic[
        diagnostic["cycle_year"] == 2026
    ].iloc[0]

    return {
        "weeks_to_census": weeks_to_census,
        "applications_cumulative": int(
            current["applications_cumulative"]
        ),
        "applications_yoy_change": round(
            current["applications_yoy_change"],
            4,
        ),
        "deposits_cumulative": int(
            current["deposits_cumulative"]
        ),
        "deposits_yoy_change": round(
            current["deposits_yoy_change"],
            4,
        ),
        "deposit_rate_to_date": round(
            current["deposit_rate_to_date"],
            4,
        ),
        "funnel_divergence": round(
            current_risk["funnel_divergence"],
            4,
        ),
        "historical_divergence_mean": round(
            reference["mean"],
            4,
        ),
        "historical_warning_threshold": round(
            reference["upper_warning"],
            4,
        ),
        "risk_status": current_risk[
            "risk_status"
        ],
        "enrollment_target": int(
            current["enrollment_target"]
        ),
        "final_enrollment": int(
            current["final_enrollment"]
        ),
        "target_gap": int(
            current["target_gap"]
        ),
    }


def run_diagnostics(df):
    """
    Run funnel-divergence diagnostics at all forecast horizons.
    """

    summary_records = []

    for horizon in FORECAST_HORIZONS:

        (
            diagnostic,
            reference,
            current_risk,
        ) = build_horizon_diagnostic(
            df,
            horizon,
        )

        recent = diagnostic[
            diagnostic["cycle_year"].isin(
                RECENT_CYCLES
            )
        ].copy()

        display_columns = [
            "cycle_year",
            "applications_cumulative",
            "applications_yoy_change",
            "deposits_cumulative",
            "deposits_yoy_change",
            "deposit_rate_to_date",
            "funnel_divergence",
            "final_enrollment",
            "target_gap",
        ]

        display = recent[
            display_columns
        ].copy()

        numeric_columns = [
            "applications_yoy_change",
            "deposits_yoy_change",
            "deposit_rate_to_date",
            "funnel_divergence",
        ]

        display[numeric_columns] = (
            display[numeric_columns]
            .round(3)
        )

        print(
            f"\nDiagnostic at {horizon} weeks before census:"
        )

        print(
            display.to_string(
                index=False
            )
        )

        print(
            "\nHistorical divergence reference:"
        )

        print(
            f"Mean: "
            f"{reference['mean']:.3f}"
        )

        print(
            f"Warning threshold: "
            f"{reference['upper_warning']:.3f}"
        )

        print(
            f"2026 risk status: "
            f"{current_risk['risk_status']}"
        )

        summary_records.append(
            build_summary_record(
                horizon,
                diagnostic,
                reference,
                current_risk,
            )
        )

    return pd.DataFrame(
        summary_records
    )


def save_summary(summary):
    """Save the diagnostic summary for reporting."""

    output_path = (
        OUTPUT_DIR
        / "funnel_divergence_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved diagnostic summary to {output_path}"
    )


def main():

    prepare_output_directory()

    df = load_data()

    summary = run_diagnostics(
        df
    )

    print(
        "\n2026 funnel-divergence summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    save_summary(
        summary
    )


if __name__ == "__main__":
    main()
