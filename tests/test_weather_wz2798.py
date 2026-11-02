import weather_wz2798
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from weather_wz2798 import (
    celsius_to_fahrenheit,
    clean_readings,
    summarize_temperatures,
)

def test_version_is_defined():
    assert isinstance(weather_wz2798.__version__, str)

# =============================================================================
# PART 1: UNIT TESTS
# =============================================================================


@pytest.mark.parametrize(
    "celsius, expected",
    [(0, 32),
     (100, 212),
     (-40, -40),
     (37, 98.6),
     (-273.15, -459.67)],)
def test_celsius_to_fahrenheit(celsius, expected):
    """
    Ensure temperatures in Celsius convert to the correct Fahrenheit values.
    """
    assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad_input, expected_error",
    [(True, TypeError),
     ('thirty seven', TypeError),
     (None, TypeError),
     (-273.16, ValueError)],)
def test_celsius_to_fahrenheit_invalid(bad_input, expected_error):
    """
    Non-numeric inputs raise TypeError and temperatures below absolute zero raise ValueError.
    """
    with pytest.raises(expected_error):
        celsius_to_fahrenheit(bad_input)


def test_summarize_temperatures():
    """
    Ensure summary statistics are computed correctly;
    The NaN reading is skipped and counted in n_missing.
    """
    # Arrange
    temp_list = [0, 100, -40, float('nan'), 80, -100]
    expect_mean = 8
    expect_min = -100
    expect_max = 100
    expect_n_missing = 1
    # Act
    result = summarize_temperatures(temp_list)
    # Assert
    assert result["mean"] == pytest.approx(expect_mean)
    assert result["min"] == expect_min
    assert result["max"] == expect_max
    assert result["n_missing"] == expect_n_missing



# =============================================================================
# PART 2: FIXTURES
# =============================================================================


@pytest.fixture
def raw_readings():
    """
    Provide raw readings with:
    - one missing temperature;
    - one implausible value;
    - unsorted rows.
    """
    return pd.DataFrame({
    "station": ["JFK", "CPK", "JFK", "CPK", "LGA"],
    "date": ["2026-07-02", "2026-07-01", "2026-07-01", "2026-07-02", "2026-07-01"],
    "temp_c": [30.0, 25.0, float("nan"), 999.0, 28.0],
})


def test_raw_readings_fixture(raw_readings):
    """
    Ensure the fixture has the columns, number of rows, and missing values as expected.
    """
    assert list(raw_readings.columns) == ["station", "date", "temp_c"]
    assert len(raw_readings) == 5
    assert raw_readings["temp_c"].isna().sum() == 1


# =============================================================================
# PART 3: TESTING DATAFRAMES
# =============================================================================


def test_clean_readings(raw_readings):
    """
    Test by matching a hand-built expected DataFrame.
    Ensure raw readings are cleaned as expected:
    - no missing values in temp_c;
    - no implausible readings in temp_c;
    - a temp_f column is added;
    - sorted by station, then date, with the index reset.
    """
    expected_cleaned_reading = pd.DataFrame({
        "station": ["CPK", "JFK", "LGA"],
        "date": ["2026-07-01", "2026-07-02", "2026-07-01"],
        "temp_c": [25.0, 30.0, 28.0],
        "temp_f": [77.0, 86.0, 82.4],
    })
    cleaned_reading = clean_readings(raw_readings)
    assert_frame_equal(cleaned_reading, expected_cleaned_reading)
