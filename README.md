# MetaStackerBandit

Batch-style MLOps pipeline that processes cryptocurrency OHLCV data, computes a rolling-mean signal, and outputs structured metrics as JSON.

## Project Structure

```
MetaStackerBandit/
├── run.py            # Main pipeline script
├── config.yaml       # Pipeline configuration
├── data.csv          # Input OHLCV data (close column required)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker build file
├── metrics.json      # Output metrics (generated)
├── run.log           # Execution log (generated)
└── README.md         # This file
```

## Quick Start (Local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

### 3. Check output

- `metrics.json` — structured metrics
- `run.log` — detailed execution log

## Docker Usage

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

The container prints `metrics.json` to stdout and exits with code 0 on success, non-zero on failure.

## Configuration (`config.yaml`)

| Field   | Type   | Description                          |
|---------|--------|--------------------------------------|
| seed    | int    | Random seed for reproducibility      |
| window  | int    | Rolling mean window size (>= 1)     |
| version | string | Pipeline version tag                 |

## Input Data

CSV file with at least a `close` column containing numeric price data. Other OHLCV columns (open, high, low, volume) are optional and ignored.

## Output Format

**Success:**

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 30,
  "seed": 42,
  "status": "success"
}
```

**Error:**

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description here"
}
```

## Pipeline Logic

1. Load and validate `config.yaml`
2. Set random seed from config
3. Load and validate input CSV
4. Compute rolling mean on `close` column using configured window
5. Generate binary signal: `1` if `close > rolling_mean`, else `0`
6. Compute `signal_rate` (mean of signal column) and `latency_ms`
7. Write metrics JSON and log file
