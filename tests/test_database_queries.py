import pandas as pd

from src.database.queries import get_high_severity_diagnostics


def test_get_high_severity_diagnostics_returns_dataframe():
    result = get_high_severity_diagnostics()

    assert isinstance(result, pd.DataFrame)


def test_get_high_severity_diagnostics_has_expected_columns():
    result = get_high_severity_diagnostics()

    expected_columns = {
        "measurement_id",
        "timestamp",
        "asset_name",
        "asset_type",
        "location",
        "scenario_name",
        "scenario_label",
        "severity_level",
        "anomaly_score",
        "anomaly_probability",
    }

    assert expected_columns.issubset(set(result.columns))


def test_get_high_severity_diagnostics_returns_only_high_severity_rows():
    result = get_high_severity_diagnostics()

    assert not result.empty
    assert set(result["severity_level"].str.lower()) == {"high"}