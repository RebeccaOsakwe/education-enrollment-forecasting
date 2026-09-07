from pathlib import Path

import pandas as pd


EXPECTED_WEEKS_PER_CYCLE = 26


def load_data():
    """Load the synthetic enrollment funnel dataset."""

    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "synthetic_enrollment_funnel.csv"

    return pd.read_csv(data_path)


def check_missing_values(df):
    """Report unexpected missing values."""

    missing = df.isna().sum()

    allowed_missing = {
        "applications_yoy_change",
        "deposits_yoy_change",
    }

    unexpected = missing[
        (missing > 0) & (~missing.index.isin(allowed_missing))
    ]

    return unexpected


def check_cycle_lengths(df):
    """Confirm that each cycle contains the expected number of weeks."""

    counts = df.groupby("cycle_year")["week"].nunique()

    return counts[counts != EXPECTED_WEEKS_PER_CYCLE]


def check_cumulative_monotonicity(df):
    """
    Check whether cumulative funnel measures ever decrease
    within an enrollment cycle.
    """

    issues = []

    cumulative_columns = [
        "applications_cumulative",
        "admits_cumulative",
        "deposits_cumulative",
        "aid_offers_cumulative",
    ]

    for year, group in df.groupby("cycle_year"):
        group = group.sort_values("week")

        for column in cumulative_columns:
            decreases = group[column].diff() < 0

            if decreases.any():
                bad_rows = group.loc[
                    decreases,
                    ["cycle_year", "week", column],
                ].copy()

                bad_rows["variable"] = column
                issues.append(bad_rows)

    if issues:
        return pd.concat(issues, ignore_index=True)

    return pd.DataFrame()


def check_funnel_logic(df):
    """
    Confirm that enrollment-funnel counts respect logical constraints.
    """

    issues = []

    admits_over_apps = df[
        df["admits_cumulative"] > df["applications_cumulative"]
    ].copy()

    if not admits_over_apps.empty:
        admits_over_apps["issue"] = "admits exceed applications"
        issues.append(admits_over_apps)

    deposits_over_admits = df[
        df["deposits_cumulative"] > df["admits_cumulative"]
    ].copy()

    if not deposits_over_admits.empty:
        deposits_over_admits["issue"] = "deposits exceed admits"
        issues.append(deposits_over_admits)

    aid_over_admits = df[
        df["aid_offers_cumulative"] > df["admits_cumulative"]
    ].copy()

    if not aid_over_admits.empty:
        aid_over_admits["issue"] = "aid offers exceed admits"
        issues.append(aid_over_admits)

    if issues:
        return pd.concat(issues, ignore_index=True)

    return pd.DataFrame()


def check_rates(df):
    """Check that rates remain within valid bounds."""

    invalid_deposit_rates = df[
        (df["deposit_rate_to_date"] < 0)
        | (df["deposit_rate_to_date"] > 1)
    ]

    return invalid_deposit_rates


def check_targets_and_outcomes(df):
    """Check that key annual values are positive and consistent."""

    issues = df[
        (df["enrollment_target"] <= 0)
        | (df["final_enrollment"] <= 0)
    ]

    return issues


def print_result(label, issues):
    """Print a simple validation result."""

    if len(issues) == 0:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label}: {len(issues)} issue(s)")
        print(issues.head(10).to_string(index=False))
        print()


def main():
    df = load_data()

    print(f"Loaded {len(df):,} rows")
    print(f"Cycles: {df['cycle_year'].min()}-{df['cycle_year'].max()}")
    print()

    print_result(
        "Unexpected missing values",
        check_missing_values(df),
    )

    print_result(
        "Cycle lengths",
        check_cycle_lengths(df),
    )

    print_result(
        "Cumulative measures are non-decreasing",
        check_cumulative_monotonicity(df),
    )

    print_result(
        "Enrollment funnel logic",
        check_funnel_logic(df),
    )

    print_result(
        "Deposit rates are between 0 and 1",
        check_rates(df),
    )

    print_result(
        "Enrollment targets and outcomes are positive",
        check_targets_and_outcomes(df),
    )


if __name__ == "__main__":
    main()
