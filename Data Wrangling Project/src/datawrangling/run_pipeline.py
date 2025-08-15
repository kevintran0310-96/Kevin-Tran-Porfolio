"""Example pipeline script for cleaning data.

This module demonstrates how to compose the functions provided by this
package into a simple end‑to‑end workflow.  It is not meant to be
executed automatically but can serve as a starting point for your own
command‑line interface or notebook.
"""

from __future__ import annotations

import logging

from .io import load_config, load_datasets, save_dataframe
from .cleaning import (
    correct_branch_code,
    correct_lat_lon,
    correct_order_items,
    compute_time_features,
    compute_distance_to_customer,
)
from .validation import OrdersSchema


def main(config_path: str | None = None) -> None:
    """Run the cleaning pipeline.

    This function loads the datasets specified in the configuration,
    applies a series of cleaning steps to the dirty dataset and writes
    the cleaned result to the configured output path.  The missing and
    outlier tasks can be implemented analogously.

    Parameters
    ----------
    config_path:
        Optional path to the YAML configuration.  If omitted the
        default configuration is used.
    """
    logging.basicConfig(level=logging.INFO)
    config = load_config(config_path)
    datasets = load_datasets(config)
    dirty = datasets.get("dirty_data")
    if dirty is None:
        raise RuntimeError("dirty_data key not found in configuration")
    # Apply cleaning steps
    dirty = correct_branch_code(dirty, valid_codes=config.get("valid_branch_codes", []))
    dirty = correct_lat_lon(dirty, lat_range=tuple(config.get("lat_range", [-38.0, -37.0])), lon_range=tuple(config.get("lon_range", [144.0, 145.0])))
    dirty = correct_order_items(dirty)
    dirty = compute_time_features(dirty)
    # Optionally compute distances if road network data is available
    nodes = datasets.get("nodes")
    edges = datasets.get("edges")
    dirty = compute_distance_to_customer(dirty, nodes=nodes, edges=edges)
    # Validate final schema (extra columns are allowed)
    OrdersSchema.validate(dirty)
    # Save result
    save_dataframe(dirty, name="dirty_solution", config=config)
    logging.info("Cleaning pipeline completed successfully")


if __name__ == "__main__":
    main()
