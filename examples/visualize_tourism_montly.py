from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
import random

import matplotlib.pyplot as plt
import pandas as pd


INPUT_DIR = Path("inputs")
TARGET_FILE = INPUT_DIR / "tourism_monthly_dataset.tsf"
OUTPUT_DIR = Path("outputs")
RANDOM_SEED = 42
SAMPLE_SERIES_COUNT = 9


def parse_tsf_file(file_path: Path) -> tuple[dict, list[dict]]:
    metadata: dict = {
        "relation": None,
        "attributes": [],
        "frequency": None,
        "horizon": None,
        "missing": None,
        "equallength": None,
    }
    records: list[dict] = []
    in_data_section = False

    with file_path.open("r", encoding="utf-8", errors="replace") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line == "@data":
                in_data_section = True
                continue

            if not in_data_section:
                if line.startswith("@relation"):
                    metadata["relation"] = line.split(maxsplit=1)[1]
                elif line.startswith("@attribute"):
                    parts = line.split()
                    metadata["attributes"].append(
                        {"name": parts[1], "type": parts[2]}
                    )
                elif line.startswith("@frequency"):
                    metadata["frequency"] = line.split(maxsplit=1)[1]
                elif line.startswith("@horizon"):
                    metadata["horizon"] = line.split(maxsplit=1)[1]
                elif line.startswith("@missing"):
                    metadata["missing"] = line.split(maxsplit=1)[1]
                elif line.startswith("@equallength"):
                    metadata["equallength"] = line.split(maxsplit=1)[1]
                continue

            parts = line.split(":")
            if len(parts) < 3:
                continue

            series_name = parts[0]
            start_timestamp = parts[1]
            raw_values = parts[2].split(",")

            observed_values: list[float] = []
            missing_count = 0

            for value in raw_values:
                if value == "?":
                    missing_count += 1
                    continue
                observed_values.append(float(value))

            records.append(
                {
                    "series_name": series_name,
                    "start_timestamp": start_timestamp,
                    "length": len(observed_values),
                    "raw_length": len(raw_values),
                    "missing_count": missing_count,
                    "values": observed_values,
                }
            )

    return metadata, records


def parse_start_timestamp(timestamp: str) -> pd.Timestamp | None:
    text = timestamp.strip()

    for fmt in ("%Y-%m", "%Y-%m-%d", "%Y %m", "%Y %m %d"):
        try:
            return pd.to_datetime(text, format=fmt)
        except ValueError:
            pass

    try:
        return pd.to_datetime(text)
    except Exception:
        return None


def build_series_dataframe(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []

    for record in records:
        start_ts = parse_start_timestamp(record["start_timestamp"])
        if start_ts is None:
            continue

        for index, value in enumerate(record["values"]):
            current_ts = start_ts + pd.DateOffset(months=index)
            rows.append(
                {
                    "series_name": record["series_name"],
                    "timestamp": current_ts,
                    "value": value,
                    "month": current_ts.month,
                    "year": current_ts.year,
                }
            )

    return pd.DataFrame(rows)


def build_summary_dataframe(records: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []

    for record in records:
        values = record["values"]
        start_ts = parse_start_timestamp(record["start_timestamp"])

        if not values:
            continue

        rows.append(
            {
                "series_name": record["series_name"],
                "start_timestamp": record["start_timestamp"],
                "start_year": start_ts.year if start_ts is not None else None,
                "length": record["length"],
                "raw_length": record["raw_length"],
                "missing_count": record["missing_count"],
                "missing_ratio": record["missing_count"] / record["raw_length"]
                if record["raw_length"] > 0
                else 0.0,
                "mean": pd.Series(values).mean(),
                "std": pd.Series(values).std(),
                "min": min(values),
                "max": max(values),
            }
        )

    return pd.DataFrame(rows)


def plot_sample_series(df_long: pd.DataFrame, output_dir: Path, sample_count: int = 9) -> None:
    series_names = sorted(df_long["series_name"].unique().tolist())
    random.seed(RANDOM_SEED)
    selected = random.sample(series_names, k=min(sample_count, len(series_names)))

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=False)
    axes = axes.flatten()

    for ax, series_name in zip(axes, selected):
        data = (
            df_long[df_long["series_name"] == series_name]
            .sort_values("timestamp")
        )
        ax.plot(data["timestamp"], data["value"])
        ax.set_title(series_name, fontsize=10)
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle("Sample tourism monthly series", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "sample_series.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_length_distribution(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(summary_df["length"], bins=20)
    plt.title("Distribution of observed series lengths")
    plt.xlabel("Observed length")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "length_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_start_year_distribution(summary_df: pd.DataFrame, output_dir: Path) -> None:
    valid = summary_df.dropna(subset=["start_year"])
    plt.figure(figsize=(8, 5))
    plt.hist(valid["start_year"], bins=min(20, valid["start_year"].nunique()))
    plt.title("Distribution of series start years")
    plt.xlabel("Start year")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "start_year_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_stats(summary_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(summary_df["mean"], bins=20)
    axes[0, 0].set_title("Mean by series")

    axes[0, 1].hist(summary_df["std"].dropna(), bins=20)
    axes[0, 1].set_title("Std by series")

    axes[1, 0].hist(summary_df["max"], bins=20)
    axes[1, 0].set_title("Max by series")

    axes[1, 1].hist(summary_df["missing_ratio"], bins=20)
    axes[1, 1].set_title("Missing ratio by series")

    fig.suptitle("Summary statistics distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_stats_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_seasonality(df_long: pd.DataFrame, output_dir: Path) -> None:
    month_profile = (
        df_long.groupby("month", as_index=False)["value"]
        .mean()
        .sort_values("month")
    )

    plt.figure(figsize=(8, 5))
    plt.plot(month_profile["month"], month_profile["value"], marker="o")
    plt.title("Average monthly seasonality pattern")
    plt.xlabel("Month")
    plt.ylabel("Average value")
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_seasonality.png", dpi=150, bbox_inches="tight")
    plt.close()


def print_basic_summary(metadata: dict, summary_df: pd.DataFrame) -> None:
    print(f"[FILE] {TARGET_FILE}")
    print(f"- relation: {metadata['relation']}")
    print(f"- frequency: {metadata['frequency']}")
    print(f"- horizon: {metadata['horizon']}")
    print(f"- series count: {len(summary_df)}")
    print(f"- mean observed length: {summary_df['length'].mean():.2f}")
    print(f"- min observed length: {summary_df['length'].min()}")
    print(f"- max observed length: {summary_df['length'].max()}")
    print(f"- average missing ratio: {summary_df['missing_ratio'].mean():.4f}")

    top_start_years = Counter(summary_df["start_year"].dropna().astype(int)).most_common(5)
    print(f"- common start years: {top_start_years}")


def main() -> None:
    if not TARGET_FILE.exists():
        raise FileNotFoundError(f"파일이 없습니다: {TARGET_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata, records = parse_tsf_file(TARGET_FILE)
    summary_df = build_summary_dataframe(records)
    df_long = build_series_dataframe(records)

    if summary_df.empty or df_long.empty:
        raise ValueError("시각화할 데이터가 비어 있습니다.")

    print_basic_summary(metadata, summary_df)

    plot_sample_series(df_long, OUTPUT_DIR, sample_count=SAMPLE_SERIES_COUNT)
    plot_length_distribution(summary_df, OUTPUT_DIR)
    plot_start_year_distribution(summary_df, OUTPUT_DIR)
    plot_summary_stats(summary_df, OUTPUT_DIR)
    plot_monthly_seasonality(df_long, OUTPUT_DIR)

    print("\nSaved plots:")
    print(f"- {OUTPUT_DIR / 'sample_series.png'}")
    print(f"- {OUTPUT_DIR / 'length_distribution.png'}")
    print(f"- {OUTPUT_DIR / 'start_year_distribution.png'}")
    print(f"- {OUTPUT_DIR / 'summary_stats_distribution.png'}")
    print(f"- {OUTPUT_DIR / 'monthly_seasonality.png'}")


if __name__ == "__main__":
    main()