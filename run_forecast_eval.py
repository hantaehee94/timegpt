from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


FORECAST_COL = "TimeGPT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TimeGPT forecast from a selected CSV window and compare it with actual values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=Path, default=Path("inputs/sample_series.csv"))
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--target-col", default="value")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--series-id", default=None)
    parser.add_argument("--start", default="0", help="Row index or timestamp to start the input window.")
    parser.add_argument("--input-steps", type=int, default=36)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--freq", default=None, help="Pandas frequency such as D, H, MS. Omit to let TimeGPT infer it.")
    parser.add_argument("--model", default="timegpt-1")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--dry-run", action="store_true", help="Validate the split without calling TimeGPT.")
    return parser.parse_args()


def load_series(args: argparse.Namespace) -> pd.DataFrame:
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if args.id_col is None and "unique_id" in df.columns:
        args.id_col = "unique_id"

    required_cols = [args.time_col, args.target_col]
    if args.id_col:
        required_cols.append(args.id_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    if df[[args.time_col, args.target_col]].isna().any().any():
        raise ValueError("Time and target columns must not contain missing or invalid values.")

    if args.id_col:
        series_ids = sorted(df[args.id_col].dropna().astype(str).unique().tolist())
        if args.series_id is None and len(series_ids) > 1:
            raise ValueError(f"Multiple series found. Pick one with --series-id. Available: {series_ids[:10]}")
        if args.series_id is not None:
            df = df[df[args.id_col].astype(str) == str(args.series_id)].copy()
            if df.empty:
                raise ValueError(f"No rows found for --series-id {args.series_id}")

    sort_cols = [args.id_col, args.time_col] if args.id_col else [args.time_col]
    return df.sort_values(sort_cols).reset_index(drop=True)


def resolve_start_index(df: pd.DataFrame, time_col: str, start: str) -> int:
    try:
        start_idx = int(start)
    except ValueError:
        start_ts = pd.to_datetime(start)
        matches = df.index[df[time_col] >= start_ts].tolist()
        if not matches:
            raise ValueError(f"No timestamp greater than or equal to --start {start}")
        start_idx = matches[0]

    if start_idx < 0 or start_idx >= len(df):
        raise ValueError(f"--start index must be between 0 and {len(df) - 1}")
    return start_idx


def split_train_test(
    df: pd.DataFrame,
    time_col: str,
    start: str,
    input_steps: int,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if input_steps <= 0 or horizon <= 0:
        raise ValueError("--input-steps and --horizon must be positive integers.")

    start_idx = resolve_start_index(df, time_col, start)
    train_end = start_idx + input_steps
    test_end = train_end + horizon
    if test_end > len(df):
        raise ValueError(
            "Not enough rows for the requested window: "
            f"need rows up to index {test_end - 1}, but CSV has {len(df)} rows."
        )

    return df.iloc[start_idx:train_end].copy(), df.iloc[train_end:test_end].copy()


def run_timegpt_forecast(
    train_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    from nixtla import NixtlaClient

    api_key = os.getenv("NIXTLA_API_KEY")
    if not api_key:
        raise ValueError("NIXTLA_API_KEY is missing. Add it to .env or your shell environment.")

    client = NixtlaClient(api_key=api_key)
    forecast_kwargs = {
        "df": train_df,
        "h": args.horizon,
        "time_col": args.time_col,
        "target_col": args.target_col,
        "model": args.model,
    }
    if args.freq:
        forecast_kwargs["freq"] = args.freq
    if args.id_col:
        forecast_kwargs["id_col"] = args.id_col

    return client.forecast(**forecast_kwargs)


def build_comparison(
    forecast_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if FORECAST_COL not in forecast_df.columns:
        raise ValueError(f"Forecast output does not contain the expected column: {FORECAST_COL}")

    forecast_df = forecast_df.copy()
    test_df = test_df.copy()
    forecast_df[args.time_col] = pd.to_datetime(forecast_df[args.time_col])
    test_df[args.time_col] = pd.to_datetime(test_df[args.time_col])

    keys = [args.time_col]
    if args.id_col and args.id_col in forecast_df.columns:
        keys.insert(0, args.id_col)

    comparison = forecast_df.merge(
        test_df[keys + [args.target_col]],
        on=keys,
        how="inner",
    )
    if comparison.empty:
        raise ValueError("Forecast timestamps did not match the actual test timestamps.")

    comparison = comparison.rename(columns={args.target_col: "actual", FORECAST_COL: "forecast"})
    comparison["error"] = comparison["actual"] - comparison["forecast"]
    comparison["abs_error"] = comparison["error"].abs()
    comparison["pct_error"] = comparison["abs_error"] / comparison["actual"].abs()
    comparison.loc[comparison["actual"] == 0, "pct_error"] = math.nan
    return comparison


def compute_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    errors = comparison["error"]
    abs_errors = comparison["abs_error"]
    actual_abs_sum = comparison["actual"].abs().sum()
    mape = comparison["pct_error"].mean() * 100
    wape = abs_errors.sum() / actual_abs_sum * 100 if actual_abs_sum else math.nan

    rows = [
        ("points", len(comparison)),
        ("mae", abs_errors.mean()),
        ("rmse", math.sqrt((errors.pow(2)).mean())),
        ("mape_percent", mape),
        ("wape_percent", wape),
        ("mean_error", errors.mean()),
        ("accuracy_from_mape_percent", 100 - mape if not math.isnan(mape) else math.nan),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def plot_result(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    comparison: pd.DataFrame,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", ".mplconfig")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 5))
    plt.plot(train_df[args.time_col], train_df[args.target_col], label="input")
    plt.plot(test_df[args.time_col], test_df[args.target_col], label="actual")
    plt.plot(comparison[args.time_col], comparison["forecast"], label="TimeGPT forecast", marker="o")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_window(name: str, df: pd.DataFrame, time_col: str) -> None:
    print(f"- {name}: {df[time_col].iloc[0]} -> {df[time_col].iloc[-1]} ({len(df)} rows)")


def main() -> None:
    load_dotenv()
    args = parse_args()
    df = load_series(args)
    train_df, test_df = split_train_test(
        df=df,
        time_col=args.time_col,
        start=args.start,
        input_steps=args.input_steps,
        horizon=args.horizon,
    )

    print("[window]")
    print_window("input", train_df, args.time_col)
    print_window("actual", test_df, args.time_col)

    if args.dry_run:
        print("\nDry run complete. TimeGPT was not called.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast_df = run_timegpt_forecast(train_df, args)
    comparison = build_comparison(forecast_df, test_df, args)
    metrics = compute_metrics(comparison)

    comparison_path = args.output_dir / "forecast_comparison.csv"
    metrics_path = args.output_dir / "metrics.csv"
    plot_path = args.output_dir / "forecast_comparison.png"

    comparison.to_csv(comparison_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    plot_result(train_df, test_df, comparison, args, plot_path)

    print("\n[metrics]")
    for row in metrics.itertuples(index=False):
        value = f"{row.value:.4f}" if isinstance(row.value, float) else row.value
        print(f"- {row.metric}: {value}")

    print("\n[saved]")
    print(f"- {comparison_path}")
    print(f"- {metrics_path}")
    print(f"- {plot_path}")


if __name__ == "__main__":
    main()
