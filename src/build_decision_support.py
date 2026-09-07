from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "forecast_2026.csv"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_enrollment_funnel.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_forecasts():
    """Load 2026 forecast results from the forecasting experiment."""

    return pd.read_csv(FORECAST_PATH)


def load_enrollment_data():
    """Load the synthetic enrollment dataset."""

    return pd.read_csv(DATA_PATH)


# ---------------------------------------------------------------------
# Decision-support calculations
# ---------------------------------------------------------------------

def get_actual_2026_outcome(df):
    """
    Return the institutional target and final enrollment
    for the synthetic 2026 cycle.
    """

    current = (
        df[df["cycle_year"] == 2026]
        [
            [
                "enrollment_target",
                "final_enrollment",
            ]
        ]
        .drop_duplicates()
    )

    if current.empty:
        raise ValueError(
            "2026 enrollment outcome not found."
        )

    row = current.iloc[0]

    return {
        "enrollment_target": int(
            row["enrollment_target"]
        ),
        "final_enrollment": int(
            row["final_enrollment"]
        ),
    }


def build_decision_summary(forecasts):
    """
    Build one executive decision-support record
    for each forecast horizon.
    """

    records = []

    for horizon in sorted(
        forecasts["weeks_to_census"].unique(),
        reverse=True,
    ):

        current = forecasts[
            forecasts["weeks_to_census"] == horizon
        ].copy()

        full_ridge = current[
            current["method"] == "ridge_regression"
        ]

        deposit = current[
            current["method"] == "deposit_conversion"
        ]

        prior_year = current[
            current["method"] == "prior_year_baseline"
        ]

        operational = current[
            current["method"] == "operational_ridge"
        ]

        if (
            full_ridge.empty
            or deposit.empty
            or prior_year.empty
            or operational.empty
        ):
            raise ValueError(
                f"Missing forecast method at {horizon} weeks."
            )

        ridge_forecast = float(
            full_ridge["predicted_enrollment"].iloc[0]
        )

        deposit_forecast = float(
            deposit["predicted_enrollment"].iloc[0]
        )

        prior_forecast = float(
            prior_year["predicted_enrollment"].iloc[0]
        )

        operational_forecast = float(
            operational["predicted_enrollment"].iloc[0]
        )

        target = int(
            full_ridge["institutional_target"].iloc[0]
        )

        forecasts_to_compare = [
            ridge_forecast,
            deposit_forecast,
            prior_forecast,
            operational_forecast,
        ]

        forecast_low = min(
            forecasts_to_compare
        )

        forecast_high = max(
            forecasts_to_compare
        )

        disagreement = (
            forecast_high
            - forecast_low
        )

        downside_gap = (
            forecast_low
            - target
        )

        ridge_gap = (
            ridge_forecast
            - target
        )

        # These thresholds are demonstration decision rules,
        # not statistically estimated uncertainty intervals.
        if disagreement >= 60:
            decision_status = (
                "High model disagreement / downside exposure"
            )

        elif disagreement >= 35:
            decision_status = (
                "Elevated model disagreement"
            )

        elif forecast_low < target - 25:
            decision_status = (
                "Downside exposure"
            )

        else:
            decision_status = (
                "Within planning range"
            )

        records.append(
            {
                "weeks_to_census": horizon,
                "institutional_target": target,
                "ridge_forecast": round(
                    ridge_forecast,
                    1,
                ),
                "operational_ridge_forecast": round(
                    operational_forecast,
                    1,
                ),
                "deposit_conversion_forecast": round(
                    deposit_forecast,
                    1,
                ),
                "prior_year_forecast": round(
                    prior_forecast,
                    1,
                ),
                "forecast_low": round(
                    forecast_low,
                    1,
                ),
                "forecast_high": round(
                    forecast_high,
                    1,
                ),
                "model_disagreement": round(
                    disagreement,
                    1,
                ),
                "lowest_forecast_gap_to_target": round(
                    downside_gap,
                    1,
                ),
                "ridge_gap_to_target": round(
                    ridge_gap,
                    1,
                ),
                "decision_status": decision_status,
            }
        )

    return pd.DataFrame(
        records
    )


# ---------------------------------------------------------------------
# Executive interpretation
# ---------------------------------------------------------------------

def build_executive_message(row):
    """
    Translate model outputs into a concise executive interpretation.
    """

    horizon = int(
        row["weeks_to_census"]
    )

    target = int(
        row["institutional_target"]
    )

    ridge = float(
        row["ridge_forecast"]
    )

    deposit = float(
        row["deposit_conversion_forecast"]
    )

    disagreement = float(
        row["model_disagreement"]
    )

    forecast_low = float(
        row["forecast_low"]
    )

    forecast_high = float(
        row["forecast_high"]
    )

    status = row[
        "decision_status"
    ]

    message = (
        f"{horizon} weeks before census, the institutional "
        f"target is {target} students. Forecasts range from "
        f"{forecast_low:.0f} to {forecast_high:.0f}. "
        f"The multivariable Ridge forecast is {ridge:.0f}, "
        f"while the deposit-based operational forecast is "
        f"{deposit:.0f}. The models differ by "
        f"{disagreement:.0f} students. "
        f"Decision status: {status}. "
        f"Leadership should avoid relying on a single point "
        f"forecast and investigate conversion behavior before "
        f"making enrollment-dependent budget or staffing commitments."
    )

    return message


def add_executive_messages(summary):
    """Attach an executive message to each forecast horizon."""

    result = summary.copy()

    result["executive_message"] = result.apply(
        build_executive_message,
        axis=1,
    )

    return result


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------

def plot_forecast_comparison(summary, actual):
    """
    Compare forecasting methods with the institutional target
    at each forecast horizon.

    The retrospective final outcome is shown separately because
    it would not have been known at forecast time.
    """

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_df = summary.sort_values(
        "weeks_to_census",
        ascending=False,
    )

    x = list(
        range(len(plot_df))
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        x,
        plot_df["ridge_forecast"],
        marker="o",
        linewidth=2,
        label="Ridge forecast",
    )

    ax.plot(
        x,
        plot_df["deposit_conversion_forecast"],
        marker="o",
        linewidth=2,
        label="Deposit conversion forecast",
    )

    ax.plot(
        x,
        plot_df["operational_ridge_forecast"],
        marker="o",
        linewidth=2,
        label="Operational Ridge forecast",
    )

    ax.plot(
        x,
        plot_df["prior_year_forecast"],
        marker="o",
        linewidth=2,
        label="Prior-year baseline",
    )

    ax.plot(
        x,
        plot_df["institutional_target"],
        linestyle="--",
        linewidth=2,
        label="Institutional target",
    )

    ax.axhline(
        actual["final_enrollment"],
        linestyle=":",
        linewidth=2,
        label="Retrospective final outcome",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            f"{int(value)} weeks"
            for value in plot_df["weeks_to_census"]
        ]
    )

    ax.set_title(
        "2026 Enrollment Forecast Comparison"
    )

    ax.set_xlabel(
        "Forecast horizon before census"
    )

    ax.set_ylabel(
        "Forecast final enrollment"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        / "forecast_comparison_2026.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        fig
    )

    print(
        f"Saved {output_path}"
    )


def plot_forecast_range(summary, actual):
    """
    Show the range between the lowest and highest forecasts
    at each forecast horizon.

    This is a model-spread visualization, not a statistical
    confidence or prediction interval.
    """

    plot_df = summary.sort_values(
        "weeks_to_census",
        ascending=False,
    )

    x = list(
        range(len(plot_df))
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for position, (_, row) in zip(
        x,
        plot_df.iterrows(),
    ):

        low = row["forecast_low"]
        high = row["forecast_high"]

        ax.vlines(
            position,
            low,
            high,
            linewidth=5,
        )

        ax.scatter(
            position,
            low,
            s=70,
        )

        ax.scatter(
            position,
            high,
            s=70,
        )

    ax.axhline(
        plot_df[
            "institutional_target"
        ].iloc[0],
        linestyle="--",
        linewidth=2,
        label="Institutional target",
    )

    ax.axhline(
        actual["final_enrollment"],
        linestyle=":",
        linewidth=2,
        label="Retrospective final outcome",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            f"{int(value)} weeks"
            for value in plot_df["weeks_to_census"]
        ]
    )

    ax.set_title(
        "2026 Forecast Range vs. Enrollment Target"
    )

    ax.set_xlabel(
        "Forecast horizon before census"
    )

    ax.set_ylabel(
        "Forecast final enrollment"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    output_path = (
        FIGURE_DIR
        / "forecast_range_2026.png"
    )

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(
        fig
    )

    print(
        f"Saved {output_path}"
    )


# ---------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------

def save_summary(summary):
    """Save the final decision-support table."""

    output_path = (
        OUTPUT_DIR
        / "decision_support_summary.csv"
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved {output_path}"
    )


def print_summary(summary, actual):
    """Print the executive decision-support output."""

    print(
        "\n2026 Executive Decision-Support Summary:"
    )

    display_columns = [
        "weeks_to_census",
        "institutional_target",
        "ridge_forecast",
        "deposit_conversion_forecast",
        "operational_ridge_forecast",
        "prior_year_forecast",
        "forecast_low",
        "forecast_high",
        "model_disagreement",
        "decision_status",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nExecutive interpretations:"
    )

    for _, row in summary.iterrows():

        print(
            f"\n- {row['executive_message']}"
        )

    print(
        "\nRetrospective synthetic outcome:"
    )

    print(
        f"Target: {actual['enrollment_target']}"
    )

    print(
        f"Final enrollment: {actual['final_enrollment']}"
    )

    print(
        "Note: the final outcome is shown only for retrospective "
        "evaluation and would not have been available at forecast time."
    )

    print(
        "\nMethodological note:"
    )

    print(
        "The forecast range reflects disagreement across modeling "
        "approaches. It is not a statistical confidence interval "
        "or prediction interval."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    forecasts = load_forecasts()

    enrollment_data = load_enrollment_data()

    actual = get_actual_2026_outcome(
        enrollment_data
    )

    summary = build_decision_summary(
        forecasts
    )

    summary = add_executive_messages(
        summary
    )

    print_summary(
        summary,
        actual,
    )

    plot_forecast_comparison(
        summary,
        actual,
    )

    plot_forecast_range(
        summary,
        actual,
    )

    save_summary(
        summary
    )


if __name__ == "__main__":
    main()
