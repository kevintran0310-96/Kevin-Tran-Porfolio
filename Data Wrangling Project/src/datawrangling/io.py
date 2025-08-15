"""I/O helpers for the data wrangling project.

This module centralises all logic for reading and writing tabular
datasets.  Input files are specified via a YAML configuration file
(`configs/default.yaml` by default), allowing you to change file paths
without touching the code.  All functions return pandas DataFrames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str | Path | None = None) -> Dict[str, object]:
    """Load the YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML config file.  If ``None``, defaults to
        ``configs/default.yaml`` relative to the project root.

    Returns
    -------
    dict
        A dictionary representation of the configuration.
    """
    if path is None:
        # Compute a path relative to this file (../.. from src/datawrangle)
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.debug("Loaded config from %s", path)
    return config


def load_datasets(config: Optional[Dict[str, object]] = None) -> Dict[str, pd.DataFrame]:
    """Load all datasets defined in the configuration.

    The configuration should map logical names (e.g. ``"dirty_data"``) to
    file paths.  The returned dictionary uses the same keys and holds
    DataFrames as values.  If a file cannot be found an exception is
    raised.

    Parameters
    ----------
    config:
        Optional configuration dictionary.  If omitted a new one will
        be loaded via :func:`load_config`.

    Returns
    -------
    dict
        Mapping of dataset names to pandas DataFrames.
    """
    if config is None:
        config = load_config()
    datasets: Dict[str, pd.DataFrame] = {}
    # Only load keys that look like file paths; ignore non‑string values
    for key, value in config.items():
        if not isinstance(value, str):
            continue
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        ext = path.suffix.lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        datasets[key] = df
        logger.info("Loaded %s from %s", key, path)
    return datasets


def save_dataframe(df: pd.DataFrame, name: str, config: Optional[Dict[str, object]] = None) -> None:
    """Write a DataFrame to the configured output location.

    Given a DataFrame and a logical name (e.g. ``"dirty_solution"``), this
    function looks up the corresponding path in the configuration and
    writes the data to that file.  The format (CSV/Excel) is inferred
    from the extension.

    Parameters
    ----------
    df:
        The DataFrame to write.
    name:
        Logical name of the output file as specified in the config.
    config:
        Optional configuration dictionary.  If omitted a new one will
        be loaded via :func:`load_config`.
    """
    if config is None:
        config = load_config()
    if name not in config or not isinstance(config[name], str):
        raise KeyError(f"No path configured for {name}")
    path = Path(config[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        raise ValueError(f"Unsupported file extension for output: {ext}")
    logger.info("Saved DataFrame to %s", path)
