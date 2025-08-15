"""Unit tests for the datawrangle.cleaning module.

These tests verify that the cleaning functions behave as expected on
simple synthetic data.  They do not exercise the full data model used
in the original assignment but serve as regression tests for the
refactored code.
"""

import logging

import numpy as np
import pandas as pd

import datawrangle.cleaning as cl


def test_get_time_of_day_with_order_type():
    assert cl.get_time_of_day("Breakfast", cl.time_class(9)) == 0
    assert cl.get_time_of_day("Lunch", cl.time_class(13)) == 1
    assert cl.get_time_of_day("Dinner", cl.time_class(19)) == 2


def test_get_time_of_day_without_order_type():
    # Use time fallback when order_type is None or invalid
    assert cl.get_time_of_day(None, cl.time_class(9)) == 0
    assert cl.get_time_of_day("", cl.time_class(14)) == 1
    # 21:00 is outside defined ranges → dinner by default
    assert cl.get_time_of_day(None, cl.time_class(21)) == 2


def test_apply_transformation():
    s = pd.Series([0, 1, 3, 9], dtype=float)
    # log1p transformation
    logged = cl.apply_transformation(s, "log")
    assert np.allclose(logged, np.log1p(s))
    # sqrt transformation
    sq = cl.apply_transformation(s, "sqrt")
    assert np.allclose(sq, np.sqrt(s))
    # None transformation
    assert cl.apply_transformation(s, "none").equals(s)


def test_correct_branch_code(caplog):
    df = pd.DataFrame({"branch_code": ["ns", "TP", "zz"]})
    with caplog.at_level(logging.WARNING):
        result = cl.correct_branch_code(df, valid_codes=("NS", "TP", "BK"))
    # Codes should be uppercased
    assert list(result["branch_code"]) == ["NS", "TP", "ZZ"]
    # Warn about invalid code ZZ
    assert any("invalid" in rec.message for rec in caplog.records)


def test_correct_order_items(caplog):
    df = pd.DataFrame({"numOrderItems": [-1, 2.7, "5", "invalid"]})
    with caplog.at_level(logging.INFO):
        result = cl.correct_order_items(df)
    # Negative value becomes zero, 2.7 rounds to 3, strings are coerced
    assert list(result["numOrderItems"]) == [0, 3, 5, 0]
    # Logging message should mention number of corrected values
    assert any("Corrected" in rec.message for rec in caplog.records)


def test_compute_time_features():
    df = pd.DataFrame({
        "date": ["2024-05-23", "2024-05-25"],
        "time": ["09:30:00", "18:00:00"],
        "order_type": ["Breakfast", "Dinner"],
    })
    out = cl.compute_time_features(df)
    # date should be datetime
    assert np.issubdtype(out["date"].dtype, np.datetime64)
    # weekend column: Thursday → 0, Saturday → 1
    assert list(out["weekend"]) == [0, 1]
    # time_of_day matches order_type mapping
    assert list(out["time_of_day"]) == [0, 2]


def test_correct_lat_lon():
    df = pd.DataFrame({
        "customer_lat": [-40.0, -37.5, -36.0],
        "customer_lon": [143.0, 145.5, 146.0],
    })
    out = cl.correct_lat_lon(df, lat_range=(-38.0, -37.0), lon_range=(144.0, 145.0))
    # Values are clipped to bounds
    assert list(out["customer_lat"]) == [-38.0, -37.5, -37.0]
    assert list(out["customer_lon"]) == [144.0, 145.0, 145.0]


def test_compute_distance_to_customer_without_kdtree(caplog):
    # When KDTree is unavailable, function should log a warning and leave column unchanged
    df = pd.DataFrame({"customer_lat": [ -37.8 ], "customer_lon": [ 144.9 ], "distance_to_customer_KM": [ 0.0 ]})
    with caplog.at_level(logging.WARNING):
        out = cl.compute_distance_to_customer(df, nodes=None, edges=None)
    assert out.equals(df)
    assert any("KDTree" in rec.message for rec in caplog.records)
