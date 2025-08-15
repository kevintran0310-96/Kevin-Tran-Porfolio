"""Core cleaning functions for the data wrangling project.

This module contains reusable, pure functions for cleaning and
feature‑engineering tabular data used in the food delivery case study.
Each function takes a :class:`pandas.DataFrame` as input and returns a
new DataFrame rather than mutating the original.  Logging is used
instead of print statements, and type hints help downstream users
understand expected input types.

These functions are deliberately granular – they perform exactly one
task each.  You can chain them together to build a complete cleaning
pipeline tailored to your needs.
"""

from __future__ import annotations

import logging
from datetime import time as time_class
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx  # type: ignore
except ImportError:
    nx = None  # Optional dependency for distance calculations

try:
    from scipy.spatial import KDTree  # type: ignore
except ImportError:
    KDTree = None  # Optional dependency

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configure a default logger if the user hasn't done so yet.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def get_time_of_day(order_type: str | None, t: time_class) -> int:
    """Categorise an order into breakfast, lunch or dinner.

    This helper function returns an integer label representing the meal
    period based either on the provided ``order_type`` or, if that
    value is missing or invalid, based on the clock time.  Breakfast is
    encoded as ``0``, lunch as ``1`` and dinner as ``2``.  Any times
    outside the normal service hours (08:00–20:00) are assigned to
    dinner by default.

    Parameters
    ----------
    order_type:
        The declared meal period, expected to be one of "Breakfast",
        "Lunch", or "Dinner" (case sensitive).  If ``None`` or an
        unrecognised value is passed, the time will be used instead.
    t:
        A :class:`datetime.time` object representing the order time.

    Returns
    -------
    int
        ``0`` for breakfast, ``1`` for lunch, ``2`` for dinner.
    """
    mapping = {"Breakfast": 0, "Lunch": 1, "Dinner": 2}
    if order_type in mapping:
        return mapping[order_type]
    # Fallback based on time
    # Breakfast: 08:00:00 – 12:00:00 inclusive
    if t >= time_class(8, 0, 0) and t <= time_class(12, 0, 0):
        return 0
    # Lunch: 12:00:01 – 16:00:00
    if t > time_class(12, 0, 0) and t <= time_class(16, 0, 0):
        return 1
    # Dinner for all remaining times
    return 2


def apply_transformation(series: pd.Series, transformation: str) -> pd.Series:
    """Apply a mathematical transformation to a numeric series.

    Supported transformations are:

    * ``"log"`` – natural log of ``(1 + x)``; useful for right‑skewed data.
    * ``"sqrt"`` – square root of ``x``; also useful for reducing skew.
    * any other string – the identity transform (returns the input as is).

    Parameters
    ----------
    series:
        A :class:`pandas.Series` of numeric type.
    transformation:
        A lowercase string indicating which transformation to apply.

    Returns
    -------
    pandas.Series
        A transformed copy of the input.
    """
    if transformation == "log":
        logger.debug("Applying log1p transformation")
        # Use log1p to handle zero values gracefully
        return np.log1p(series)
    if transformation == "sqrt":
        logger.debug("Applying square root transformation")
        return np.sqrt(series)
    logger.debug("No transformation applied")
    return series


def correct_branch_code(df: pd.DataFrame, valid_codes: Iterable[str] = ("NS", "TP", "BK")) -> pd.DataFrame:
    """Standardise branch codes and warn on invalid entries.

    The `branch_code` field in the raw dataset contains mixed case
    values (e.g. ``"bk"`` instead of ``"BK"``).  This function
    normalises the codes to uppercase and logs a warning if any code is
    not one of the expected values.

    Parameters
    ----------
    df:
        DataFrame containing a `branch_code` column.
    valid_codes:
        An iterable of valid branch codes.  Invalid codes will remain
        unchanged but will be reported via the logger.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with a normalised `branch_code` column.
    """
    out = df.copy()
    out["branch_code"] = out["branch_code"].astype(str).str.upper()
    invalid = ~out["branch_code"].isin(valid_codes)
    if invalid.any():
        logger.warning("Found invalid branch codes: %s", out.loc[invalid, "branch_code"].unique())
    return out


def correct_lat_lon(
    df: pd.DataFrame,
    lat_range: Tuple[float, float] = (-38.0, -37.0),
    lon_range: Tuple[float, float] = (144.0, 145.0),
    ) -> pd.DataFrame:
    """Clip latitude and longitude values to a specified bounding box.

    Latitude and longitude recorded for customers can sometimes fall
    outside the expected Melbourne bounds.  This helper clips the
    `customer_lat` and `customer_lon` columns to the provided ranges.

    Parameters
    ----------
    df:
        DataFrame with columns `customer_lat` and `customer_lon`.
    lat_range:
        Two‑element tuple specifying the minimum and maximum allowed
        latitudes.
    lon_range:
        Two‑element tuple specifying the minimum and maximum allowed
        longitudes.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with clipped latitude and longitude.
    """
    out = df.copy()
    out["customer_lat"] = out["customer_lat"].clip(*lat_range)
    out["customer_lon"] = out["customer_lon"].clip(*lon_range)
    return out


def correct_order_items(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `numOrderItems` is a non‑negative integer.

    Some entries in the raw dataset may have negative or float values in
    the `numOrderItems` column.  This function rounds the numbers to
    the nearest integer and enforces a minimum of zero.  It also logs
    how many values were corrected.

    Parameters
    ----------
    df:
        DataFrame containing a `numOrderItems` column.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with a cleaned `numOrderItems` column.
    """
    out = df.copy()
    original = out["numOrderItems"].copy()
    # Convert to numeric, coercing errors to NaN then fill NaN with zero
    numeric = pd.to_numeric(out["numOrderItems"], errors="coerce").fillna(0)
    # Round and clip at zero
    cleaned = numeric.round().astype(int).clip(lower=0)
    out["numOrderItems"] = cleaned
    corrected = (original != cleaned).sum()
    if corrected > 0:
        logger.info("Corrected %d values in numOrderItems", corrected)
    return out


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date/time columns and derive weekend and time‑of‑day features.

    This convenience wrapper performs several common transformations on
    the `date` and `time` columns:

    * Converts the `date` column to datetime and extracts a boolean
      `weekend` column (1 for Saturday/Sunday, 0 for weekdays).
    * Converts the `time` column from string to :class:`datetime.time`.
    * Derives a new `time_of_day` integer feature via
      :func:`~datawrangle.cleaning.get_time_of_day`.

    Parameters
    ----------
    df:
        DataFrame with `date`, `time` and `order_type` columns.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with three additional columns: `date` (as
        datetime), `weekend` (int), and `time_of_day` (int).
    """
    out = df.copy()
    # Convert `date` to pandas datetime; coerce errors to NaT
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    # Weekend flag: True for Saturday (5) and Sunday (6)
    out["weekend"] = (out["date"].dt.dayofweek >= 5).astype(int)
    # Convert `time` to python time; coerce errors to NaT then fill with midnight
    out["time"] = pd.to_datetime(out["time"], format="%H:%M:%S", errors="coerce").dt.time
    # Compute `time_of_day` using our helper
    out["time_of_day"] = out.apply(lambda row: get_time_of_day(row.get("order_type"), row.get("time")), axis=1)
    return out


def compute_distance_to_customer(
    df: pd.DataFrame,
    nodes: pd.DataFrame | None = None,
    edges: pd.DataFrame | None = None,
    distance_column: str = "distance_to_customer_KM",
) -> pd.DataFrame:
    """Calculate an approximate distance from a branch to each customer.

    In the original assignment the distance between a branch and a
    customer was computed via the shortest path on a road network.  This
    function provides a simplified, optional implementation: if the
    optional :mod:`networkx` and :mod:`scipy` dependencies are
    installed and node/edge tables are supplied, it builds a KDTree on
    the node coordinates and assigns the Euclidean distance in
    kilometres between each customer location and the nearest node.  If
    either dependency is missing or the auxiliary tables are not
    provided, the function logs a warning and leaves the distance column
    unchanged.

    Parameters
    ----------
    df:
        DataFrame containing `customer_lat` and `customer_lon` columns.
    nodes:
        Optional DataFrame of road network nodes with columns `lat` and
        `lon`.
    edges:
        Optional DataFrame of road network edges; currently ignored but
        included for API compatibility with more sophisticated
        implementations.
    distance_column:
        Name of the distance column to compute; defaults to
        ``"distance_to_customer_KM"``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with the distance column populated.  If
        computation could not be performed, the original column is left
        untouched.
    """
    out = df.copy()
    # Only proceed if dependencies are available and nodes are provided
    if KDTree is None or nodes is None:
        logger.warning(
            "KDTree or nodes data missing – distance_to_customer_KM will not be recomputed"
        )
        return out
    # Build a KDTree on the node coordinates (lat/lon) in degrees
    coords = nodes[["lat", "lon"]].dropna().to_numpy()
    if coords.size == 0:
        logger.warning("No coordinates available in nodes to compute distances")
        return out
    tree = KDTree(coords)
    # Query distances (in degrees) to nearest node for each customer location
    query_points = out[["customer_lat", "customer_lon"]].to_numpy()
    # Replace NaNs with zeros to avoid exceptions
    query_points = np.nan_to_num(query_points, nan=0.0)
    distances_deg, _ = tree.query(query_points)
    # Convert degrees to approximate kilometres (1 deg lat ~ 111 km)
    distances_km = distances_deg * 111.0
    out[distance_column] = distances_km
    logger.info("Computed distances for %d records", len(out))
    return out
