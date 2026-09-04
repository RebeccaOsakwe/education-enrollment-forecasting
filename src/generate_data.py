from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
START_YEAR = 2018
END_YEAR = 2026
WEEKS_PER_CYCLE = 26


def logistic_progress(week, midpoint, steepness):
    """
    Create a smooth cumulative progression curve between 0 and 1.

    Used to simulate how applications, admits, deposits, and aid offers
    accumulate over the enrollment cycle.
    """
    progress = 1 / (1 + np.exp(-steepness * (week - midpoint)))

    # Rescale so the first week is near 0 and the final week is near 1.
    start = 1 / (1 + np.exp(-steepness * (1 - midpoint)))
    end = 1 / (1 + np.exp(-steepness * (WEEKS_PER_CYCLE - midpoint)))

    return (progress - start) / (end - start)


def generate_year_parameters(year, rng):
    """
    Define the overall enrollment environment for one admission cycle.

    The parameters vary by year so that the synthetic data contain
    realistic differences in application volume, conversion behavior,
    enrollment targets, tuition, and external shocks.
    """

    year_index = year - START_YEAR

    # Gradual long-run growth in demand.
    application_base = 5200 * (1 + 0.025 * year_index)

    # Stable but slowly changing institutional enrollment target.
    enrollment_target = 760 + (10 * year_index)

    # Baseline admission rate.
    admit_rate = 0.64 - (0.003 * year_index)

    # Deposit behavior changes over time.
    deposit_rate = 0.245 - (0.003 * year_index)

    # Yield from deposited student to final enrolled student.
    deposit_to_enroll_rate = 0.92

    # Average projected net tuition rises gradually over time.
    avg_net_tuition = 22500 * (1 + 0.018 * year_index)

    # Aid coverage is represented as the share of admits receiving an offer.
    aid_offer_rate = 0.78

    # Add annual variation.
    application_base *= rng.normal(1.0, 0.035)
    admit_rate += rng.normal(0, 0.012)
    deposit_rate += rng.normal(0, 0.008)
    deposit_to_enroll_rate += rng.normal(0, 0.012)
    avg_net_tuition *= rng.normal(1.0, 0.01)

    # Introduce structural changes and shocks.
    if year == 2020:
        # Application disruption and weaker enrollment conversion.
        application_base *= 0.90
        deposit_rate *= 0.92
        deposit_to_enroll_rate *= 0.94

    elif year == 2021:
        # Partial recovery following the disruption.
        application_base *= 1.04
        deposit_rate *= 0.98

    elif year in (2024, 2025):
        # Stronger application volume but weaker conversion.
        application_base *= 1.08
        deposit_rate *= 0.94

    elif year == 2026:
        # Current cycle: applications remain healthy, but deposit behavior
        # is weaker than historical patterns.
        application_base *= 1.10
        deposit_rate *= 0.89

    return {
        "application_total": max(round(application_base), 1000),
        "enrollment_target": round(enrollment_target),
        "admit_rate": np.clip(admit_rate, 0.45, 0.80),
        "deposit_rate": np.clip(deposit_rate, 0.12, 0.40),
        "deposit_to_enroll_rate": np.clip(
            deposit_to_enroll_rate, 0.80, 0.98
        ),
        "avg_net_tuition": round(avg_net_tuition, 2),
        "aid_offer_rate": aid_offer_rate,
    }


def generate_cycle(year, rng):
    """
    Generate weekly cumulative enrollment-funnel observations
    for a single fall enrollment cycle.
    """

    params = generate_year_parameters(year, rng)

    application_total = params["application_total"]

    final_admits = round(
        application_total * params["admit_rate"]
    )

    final_deposits = round(
        final_admits * params["deposit_rate"]
    )

    final_enrollment = round(
        final_deposits * params["deposit_to_enroll_rate"]
        + rng.normal(0, 18)
    )

    final_enrollment = max(final_enrollment, 0)

    final_aid_offers = round(
        final_admits * params["aid_offer_rate"]
    )

    records = []

    for week in range(1, WEEKS_PER_CYCLE + 1):

        # Different funnel stages accumulate on different schedules.
        application_progress = logistic_progress(
            week=week,
            midpoint=9,
            steepness=0.34
        )

        admit_progress = logistic_progress(
            week=week,
            midpoint=12,
            steepness=0.34
        )

        aid_progress = logistic_progress(
            week=week,
            midpoint=13,
            steepness=0.32
        )

        deposit_progress = logistic_progress(
            week=week,
            midpoint=18,
            steepness=0.42
        )

        applications = round(
            application_total
            * application_progress
            * rng.normal(1.0, 0.012)
        )

        admits = round(
            final_admits
            * admit_progress
            * rng.normal(1.0, 0.012)
        )

        aid_offers = round(
            final_aid_offers
            * aid_progress
            * rng.normal(1.0, 0.015)
        )

        deposits = round(
            final_deposits
            * deposit_progress
            * rng.normal(1.0, 0.018)
        )

        # Prevent cumulative measures from exceeding their final totals.
        applications = int(
            np.clip(applications, 0, application_total)
        )

        admits = int(
            np.clip(admits, 0, final_admits)
        )

        aid_offers = int(
            np.clip(aid_offers, 0, final_aid_offers)
        )

        deposits = int(
            np.clip(deposits, 0, final_deposits)
        )

        deposit_rate_to_date = (
            deposits / admits if admits > 0 else 0
        )

        # Small weekly variation around the year's average net tuition.
        weekly_net_tuition = (
            params["avg_net_tuition"]
            * rng.normal(1.0, 0.006)
        )

        records.append(
            {
                "cycle_year": year,
                "week": week,
                "weeks_to_census": WEEKS_PER_CYCLE - week,
                "enrollment_target": params["enrollment_target"],
                "applications_cumulative": applications,
                "admits_cumulative": admits,
                "deposits_cumulative": deposits,
                "aid_offers_cumulative": aid_offers,
                "avg_net_tuition": round(
                    weekly_net_tuition, 2
                ),
                "deposit_rate_to_date": round(
                    deposit_rate_to_date, 4
                ),
                "final_enrollment": final_enrollment,
                "target_gap": (
                    final_enrollment
                    - params["enrollment_target"]
                ),
            }
        )

    return pd.DataFrame(records)


def add_year_over_year_features(df):
    """
    Compare cumulative application and deposit volume with the
    same week in the prior enrollment cycle.
    """

    df = df.sort_values(
        ["cycle_year", "week"]
    ).copy()

    df["applications_yoy_change"] = (
        df.groupby("week")["applications_cumulative"]
        .pct_change()
    )

    df["deposits_yoy_change"] = (
        df.groupby("week")["deposits_cumulative"]
        .pct_change()
    )

    return df


def generate_dataset():
    """
    Generate all synthetic enrollment cycles.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    cycles = [
        generate_cycle(year, rng)
        for year in range(START_YEAR, END_YEAR + 1)
    ]

    df = pd.concat(
        cycles,
        ignore_index=True
    )

    df = add_year_over_year_features(df)

    return df


def save_dataset(df):
    """
    Save the generated dataset to the project's data directory.
    """

    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data"
    data_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        data_dir
        / "synthetic_enrollment_funnel.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved {len(df):,} rows to {output_path}"
    )


if __name__ == "__main__":
    enrollment_data = generate_dataset()

    print(
        enrollment_data.head()
    )

    print(
        "\nFinal enrollment by cycle:"
    )

    print(
        enrollment_data[
            [
                "cycle_year",
                "enrollment_target",
                "final_enrollment",
                "target_gap",
            ]
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

    save_dataset(
        enrollment_data
    )
