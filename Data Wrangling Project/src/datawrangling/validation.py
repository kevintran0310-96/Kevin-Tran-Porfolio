"""Data schema definitions for validating inputs and outputs.

We use the `pandera` package to define data contracts that our
cleaning functions must satisfy.  Schemas help catch unexpected
changes to the data early in the pipeline and provide documentation
about expected fields and types.  Pandera models are type‑checked at
runtime, so they complement static type hints without replacing them.

See https://pandera.readthedocs.io/ for more information.
"""

from __future__ import annotations

# Pandera is an optional dependency.  If it is not installed the
# following try/except block defines very small shim classes so that
# downstream code still functions.  The shim does not perform any
# validation and should not be used in production.
try:
    import pandera as pa  # type: ignore
    from pandera import DataFrameModel  # type: ignore
    from pandera.typing import Index, Series  # type: ignore
except ModuleNotFoundError:
    pa = None  # type: ignore

    class DataFrameModel:  # type: ignore
        """Fallback base class used when pandera is unavailable."""

        @classmethod
        def validate(cls, df, *args, **kwargs):  # noqa: D401
            """Return the DataFrame unchanged (no validation performed)."""
            return df

        class Config:
            extra = "allow"

    class Series:  # type: ignore
        def __class_getitem__(cls, item):  # type: ignore
            return object

    class Index:  # type: ignore
        def __class_getitem__(cls, item):  # type: ignore
            return object
    # Provide a dummy `Field` so that annotations do not error when
    # `pandera` is unavailable.  The dummy accepts any arguments and
    # returns None.  Without this shim `pa.Field` would raise an
    # AttributeError.
    from types import SimpleNamespace as _SimpleNamespace  # type: ignore
    pa = _SimpleNamespace(Field=lambda *args, **kwargs: None)  # type: ignore
import pandas as pd


class OrdersSchema(DataFrameModel):
    """Schema for food delivery orders.

    This schema captures a minimal set of columns expected in the
    assignment’s datasets.  Additional columns are permitted by
    default; only those listed here will be validated.  Modify the
    column names and checks to reflect your own data accurately.
    """

    # Index
    index: Index[int] = pa.Field(ge=0)

    # Date and time columns
    date: Series[pd.Timestamp] = pa.Field(coerce=True)
    time: Series[object] = pa.Field()

    # Categorical columns
    order_type: Series[str] = pa.Field(isin={"Breakfast", "Lunch", "Dinner"})
    branch_code: Series[str] = pa.Field(regex="^(NS|TP|BK)$")

    # Numerical columns
    customer_lat: Series[float] = pa.Field(ge=-90.0, le=90.0)
    customer_lon: Series[float] = pa.Field(ge=-180.0, le=180.0)
    orderPrice: Series[float] = pa.Field(ge=0.0)
    numOrderItems: Series[int] = pa.Field(ge=0)
    distance_to_customer_KM: Series[float] = pa.Field(ge=0.0)
    customerHasloyalty: Series[int] = pa.Field(isin={0, 1})

    class Config:
        extra = "allow"
