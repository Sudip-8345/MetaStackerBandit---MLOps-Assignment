import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch MLOps pipeline: rolling-mean signal generator."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON.")
    parser.add_argument("--log-file", required=True, help="Path to log file.")
    return parser.parse_args()


def setup_logging(log_file: str) -> logging.Logger:
    """Configure and return a logger that writes to the specified file."""
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def load_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Load YAML config, validate required fields, and return as dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None or not isinstance(config, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping.")

    required_fields = {"seed": int, "window": int, "version": str}
    for field, expected_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required config field: '{field}'")
        if not isinstance(config[field], expected_type):
            raise TypeError(
                f"Config field '{field}' must be {expected_type.__name__}, "
                f"got {type(config[field]).__name__}"
            )

    if config["window"] < 1:
        raise ValueError("Config field 'window' must be >= 1.")

    logger.info("Config loaded: seed=%d, window=%d, version=%s",
                config["seed"], config["window"], config["version"])
    return config


def load_data(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """Load CSV, validate structure, and return DataFrame."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Input CSV has no data rows.")

    if "close" not in df.columns:
        raise ValueError("Input CSV missing required column: 'close'")

    logger.info("Data loaded: %d rows", len(df))
    return df


def compute_rolling_mean(df: pd.DataFrame, window: int,
                         logger: logging.Logger) -> pd.Series:
    rolling_mean = df["close"].rolling(window=window, min_periods=window).mean()
    logger.info("Rolling mean calculated (window=%d)", window)
    return rolling_mean


def generate_signals(df: pd.DataFrame, rolling_mean: pd.Series,
                     logger: logging.Logger) -> pd.Series:
    signal = (df["close"] > rolling_mean).astype(int)
    # Initial rows where rolling_mean is NaN get signal 0
    signal = signal.fillna(0).astype(int)
    logger.info("Signals generated: %d total", len(signal))
    return signal


def compute_metrics(df: pd.DataFrame, signal: pd.Series, version: str,
                    seed: int, latency_ms: int,
                    logger: logging.Logger) -> Dict[str, Any]:
    rows_processed = len(df)
    signal_rate = float(signal.mean())

    metrics = {
        "version": version,
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": latency_ms,
        "seed": seed,
        "status": "success",
    }

    logger.info("Metrics: rows_processed=%d, signal_rate=%.4f, latency_ms=%d",
                rows_processed, signal_rate, latency_ms)
    return metrics


def write_output(metrics: Dict[str, Any], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def build_error_output(version: str, error_message: str) -> Dict[str, Any]:
    return {
        "version": version,
        "status": "error",
        "error_message": error_message,
    }


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file)

    version = "v1"  # default fallback if config cannot be loaded

    try:
        start_time = time.monotonic()
        logger.info("Job started")

        # --- Load config ---
        config = load_config(args.config, logger)
        version = config["version"]
        seed = config["seed"]
        window = config["window"]

        # --- Set random seed ---
        np.random.seed(seed)
        logger.info("Random seed set to %d", seed)

        # --- Load data ---
        df = load_data(args.input, logger)

        # --- Compute rolling mean ---
        rolling_mean = compute_rolling_mean(df, window, logger)

        # --- Generate signals ---
        signal = generate_signals(df, rolling_mean, logger)

        # --- Compute metrics ---
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        metrics = compute_metrics(df, signal, version, seed, elapsed_ms, logger)

        # --- Write output ---
        write_output(metrics, args.output)
        logger.info("Job completed successfully")

    except Exception as exc:
        logger.error("Pipeline failed: %s", str(exc))
        error_output = build_error_output(version, str(exc))
        write_output(error_output, args.output)
        sys.exit(1)


if __name__ == "__main__":
    main()
