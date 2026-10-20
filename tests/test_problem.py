"""Unit tests for dcmbench.problem (ChoiceProblem, attributes, units)."""

import numpy as np
import pandas as pd
import pytest

from dcmbench.problem import Attribute, ChoiceProblem, parse_cost_unit, to_currency_per_hour


def _problem(**overrides):
    kwargs = dict(
        name="toy",
        alternatives=("bus", "car"),
        choice_column="CHOICE",
        choice_codes={1: "bus", 2: "car"},
        availability={"car": "CAR_AV"},
        attributes={
            "time": Attribute({"bus": "BUS_TT", "car": "CAR_TT"}, unit="minute", kind="time"),
            "cost": Attribute({"bus": "BUS_CO", "car": "CAR_CO"}, unit="CHF", kind="cost"),
        },
        group_column="ID",
    )
    kwargs.update(overrides)
    return ChoiceProblem(**kwargs)


def _data():
    return pd.DataFrame(
        {
            "ID": [1, 1, 2, 2],
            "CHOICE": [1, 2, 1, 1],
            "CAR_AV": [1, 1, 0, 1],
            "BUS_TT": [30, 40, 35, 50],
            "CAR_TT": [20, 25, 0, 30],
            "BUS_CO": [2.0, 2.0, 3.0, 2.5],
            "CAR_CO": [5.0, 4.0, 0.0, 6.0],
        }
    )


def test_valid_problem_and_data():
    problem = _problem()
    problem.validate(_data())
    assert problem.choice_mapping == {1: "bus", 2: "car"}
    assert list(problem.choices(_data())) == ["bus", "car", "bus", "bus"]
    assert problem.availability_matrix(_data()).to_dict("list") == {
        "bus": [True, True, True, True],
        "car": [True, True, False, True],
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"alternatives": ("bus", "bus")}, "unique"),
        ({"choice_codes": {1: "bus", 2: "tram"}}, "unknown alternatives"),
        ({"choice_codes": {1: "bus"}}, "no code for alternatives"),
        ({"availability": {"tram": "X"}}, "availability names unknown"),
        ({"attributes": {"time": Attribute({"tram": "T"}, unit="minute", kind="time")}}, "unknown alternatives"),
    ],
)
def test_invalid_problem_definitions(overrides, message):
    with pytest.raises(ValueError, match=message):
        _problem(**overrides)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns="CAR_TT"), "columns missing"),
        (lambda d: d.assign(CHOICE=[1, 2, 3, 1]), "choice codes not in choice_codes"),
        (lambda d: d.assign(CAR_AV=[1, 1, 2, 1]), "only 0/1"),
        (lambda d: d.assign(CHOICE=[1, 2, 2, 1]), "chose an unavailable alternative"),
        (lambda d: d.assign(BUS_TT=[30, np.nan, 35, 50]), "finite numbers"),
        (lambda d: d.assign(ID=[1, None, 2, 2]), "missing values"),
    ],
)
def test_validate_rejects_bad_data(mutate, message):
    with pytest.raises(ValueError, match=message):
        _problem().validate(mutate(_data()))


def test_attribute_units_are_checked():
    with pytest.raises(ValueError, match="time unit"):
        Attribute({"bus": "T"}, unit="minutes", kind="time")
    with pytest.raises(ValueError, match="cost unit"):
        Attribute({"bus": "C"}, unit="francs", kind="cost")
    Attribute({"bus": "F"}, unit="anything", kind="other")


def test_value_of_time_unit_conversion():
    value, currency = to_currency_per_hour(0.5, time_unit="minute", cost_unit="CHF")
    assert currency == "CHF" and value == pytest.approx(30.0)
    value, currency = to_currency_per_hour([2.0, 4.0], time_unit="hour", cost_unit="GBP")
    assert currency == "GBP" and list(value) == [2.0, 4.0]
    value, _ = to_currency_per_hour(50.0, time_unit="minute", cost_unit="0.01 CHF")
    assert value == pytest.approx(30.0)
    assert parse_cost_unit("0.01 CHF") == ("CHF", 0.01)
