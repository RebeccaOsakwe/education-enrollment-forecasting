from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_enrollment_funnel.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_data():
    """Load the validated synthetic enrollment dataset."""

    return pd.read_csv(DATA_PATH)


def prepare_output_directory():
    """Create the figure directory if it does not already exist."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def get_annual_summary(df):
    """
    Return one record per enrollment cycle with the final outcome
    and institutional target.
    """

    annual = (
        df[
            [
                "cycle_year",
                "enrollment_target",
                "final_enrollment",
                "target_gap",
            ]
        ]
        .drop_duplicates()
        .sort_values("cycle_year")
        .reset_index(drop=True)
    )

    annual["target_attainment_rate"] = (
        annual["final_enrollment"]
        / annual["enrollment_target"]
    )

    return annual


def plot_enrollment_vs_target(annual):
    """
    Compare final enrollment with the institutional target
    across enrollment cycles.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        annual["cycle_year"],
        annual["final_enrollment"],
        marker="o",
        linewidth=2,
        label="Final enrollment",
    )

    ax.plot(
        annual["cycle_year"],
        annual["enrollment_target"],
        marker="o",
        linewidth=2,
        linestyle="--",
        label="Enrollment target",
    )

    ax.set_title("Final Enrollment vs. Institutional Target")
    ax.set_xlabel("Enrollment cycle")
    ax.set_ylabel("Students")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    output_path = FIGURE_DIR / "enrollment_vs_target.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Saved {output_path}")


def plot_application_trajectories(df):
    """
    Show how cumulative applications develop throughout
    each enrollment cycle.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for year, group in df.groupby("cycle_year"):
        group = group.sort_values("week")

        ax.plot(
            group["week"],
            group["applications_cumulative"],
            linewidth=1.8,
            label=str(year),
        )

    ax.set_title("Cumulative Applications by Enrollment Cycle")
    ax.set_xlabel("Week in admissions cycle")
    ax.set_ylabel("Cumulative applications")
    ax.legend(
        title="Cycle",
        ncol=3,
        fontsize=8,
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()

    output_path = FIGURE_DIR / "application_trajectories.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Saved {output_path}")


def plot_deposit_trajectories(df):
    """
    Show how cumulative deposits develop throughout
    each enrollment cycle.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for year, group in df.groupby("cycle_year"):
        group = group.sort_values("week")

        ax.plot(
            group["week"],
            group["deposits_cumulative"],
            linewidth=1.8,
            label=str(year),
        )

    ax.set_title("Cumulative Deposits by Enrollment Cycle")
    ax.set_xlabel("Week in admissions cycle")
    ax.set_ylabel("Cumulative deposits")
    ax.legend(
        title="Cycle",
        ncol=3,
        fontsize=8,
    )
    ax.grid(alpha=0.25)

    fig.tight_layout()

    output_path = FIGURE_DIR / "deposit_trajectories.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Saved {output_path}")


def calculate_signal_correlations(df):
    """
    Measure how strongly applications and deposits observed at
    selected forecast horizons are associated with final enrollment.

    Correlations are descriptive only. With nine enrollment cycles,
    they should not be interpreted as precise estimates of
    predictive performance.
    """

    horizons = [12, 8, 4]

    results = []

    for horizon in horizons:
        snapshot = df[
            df["weeks_to_census"] == horizon
        ].copy()

        results.append(
            {
                "weeks_to_census": horizon,
                "application_correlation": (
                    snapshot["applications_cumulative"]
                    .corr(snapshot["final_enrollment"])
                ),
                "deposit_correlation": (
                    snapshot["deposits_cumulative"]
                    .corr(snapshot["final_enrollment"])
                ),
                "deposit_rate_correlation": (
                    snapshot["deposit_rate_to_date"]
                    .corr(snapshot["final_enrollment"])
                ),
            }
        )

    return pd.DataFrame(results)


def print_annual_summary(annual):
    """Print a concise annual outcome table."""

    display = annual.copy()

    display["target_attainment_rate"] = (
        display["target_attainment_rate"] * 100
    ).round(1)

    display = display.rename(
        columns={
            "target_attainment_rate": "target_attainment_pct"
        }
    )

    print("\nAnnual enrollment outcomes:")
    print(display.to_string(index=False))


def print_signal_correlations(correlations):
    """Print descriptive signal correlations by forecast horizon."""

    display = correlations.copy()

    numeric_columns = [
        "application_correlation",
        "deposit_correlation",
        "deposit_rate_correlation",
    ]

    display[numeric_columns] = (
        display[numeric_columns].round(3)
    )

    print("\nSignal correlations with final enrollment:")
    print(display.to_string(index=False))


def main():
    prepare_output_directory()

    df = load_data()
    annual = get_annual_summary(df)

    print_annual_summary(annual)

    correlations = calculate_signal_correlations(df)
    print_signal_correlations(correlations)

    plot_enrollment_vs_target(annual)
    plot_application_trajectories(df)
    plot_deposit_trajectories(df)


if __name__ == "__main__":
    main()
