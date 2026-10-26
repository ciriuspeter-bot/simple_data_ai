"""Simple data-processing AI pipeline: clean, summarize, and generate insights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

INSIGHT_RULES = {
    "low_units": 5,
    "high_revenue_share": 0.35,
    "outlier_z": 2.0,
}


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        raise SystemExit(f"Unsupported file type: {path.suffix}. Use a CSV file.")
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip().lower().replace(" ", "_") for col in cleaned.columns]
    cleaned = cleaned.dropna(how="all")

    text_cols = cleaned.select_dtypes(include=["object", "string", "str"]).columns
    for col in text_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    for col in ("units", "revenue"):
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    if "units" in cleaned.columns:
        cleaned["units"] = cleaned["units"].fillna(0)
    if "product" in cleaned.columns:
        cleaned = cleaned.dropna(subset=["product"])

    return cleaned.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    summary: dict = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
    }

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        stats = numeric.describe().round(2)
        summary["numeric_stats"] = json.loads(stats.to_json())

    if "region" in df.columns:
        summary["by_region"] = json.loads(
            df.groupby("region", dropna=False)[["units", "revenue"]]
            .sum(numeric_only=True)
            .round(2)
            .to_json(orient="index")
        )

    if "product" in df.columns:
        summary["by_product"] = json.loads(
            df.groupby("product", dropna=False)[["units", "revenue"]]
            .sum(numeric_only=True)
            .round(2)
            .to_json(orient="index")
        )

    if "date" in df.columns and df["date"].notna().any():
        daily = df.dropna(subset=["date"]).groupby("date")["revenue"].sum().sort_index()
        summary["date_range"] = {
            "start": str(daily.index.min().date()),
            "end": str(daily.index.max().date()),
        }
        if len(daily) >= 2:
            summary["revenue_trend"] = "up" if daily.iloc[-1] >= daily.iloc[0] else "down"

    return summary


def generate_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if df.empty:
        return ["No rows left after cleaning, so no insights were generated."]

    if "revenue" in df.columns and df["revenue"].sum() > 0:
        total = float(df["revenue"].sum())
        if "product" in df.columns:
            top_product = df.groupby("product")["revenue"].sum().idxmax()
            share = float(df.groupby("product")["revenue"].sum().max() / total)
            insights.append(
                f"{top_product} is the top product, with {share:.0%} of total revenue."
            )
            if share >= INSIGHT_RULES["high_revenue_share"]:
                insights.append(
                    "Revenue is concentrated in one product. Diversifying sales would reduce risk."
                )

        if "region" in df.columns:
            top_region = df.groupby("region")["revenue"].sum().idxmax()
            insights.append(f"{top_region} is the strongest region by revenue.")

        mean = df["revenue"].mean()
        std = df["revenue"].std(ddof=0)
        if std and std > 0:
            z = (df["revenue"] - mean) / std
            outliers = df.loc[z.abs() >= INSIGHT_RULES["outlier_z"]]
            if not outliers.empty:
                insights.append(
                    f"{len(outliers)} row(s) look unusual (revenue z-score >= {INSIGHT_RULES['outlier_z']})."
                )

    if "units" in df.columns and "product" in df.columns:
        slow = df[df["units"] <= INSIGHT_RULES["low_units"]]
        if not slow.empty:
            products = ", ".join(sorted(slow["product"].dropna().unique()))
            insights.append(
                f"Low-volume rows (<= {INSIGHT_RULES['low_units']} units) appear for: {products}."
            )

    if "date" in df.columns and "revenue" in df.columns:
        daily = df.dropna(subset=["date"]).groupby("date")["revenue"].sum().sort_index()
        if len(daily) >= 4:
            mid = len(daily) // 2
            first, second = daily.iloc[:mid].mean(), daily.iloc[mid:].mean()
            if second > first * 1.1:
                insights.append("Recent daily revenue is higher than the earlier period.")
            elif first > second * 1.1:
                insights.append("Recent daily revenue is lower than the earlier period.")
            else:
                insights.append("Daily revenue is relatively stable across the period.")

    if not insights:
        insights.append("Data looks consistent. No strong heuristic signals were found.")
    return insights


def write_outputs(cleaned: pd.DataFrame, summary: dict, insights: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_dir / "cleaned.csv", index=False)
    report = {"summary": summary, "insights": insights}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a CSV, summarize it, and print simple AI-style insights."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="data/sample_sales.csv",
        help="Path to a CSV file (default: data/sample_sales.csv)",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Directory for cleaned.csv and report.json (default: output)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.csv_path)
    if not source.exists():
        raise SystemExit(f"File not found: {source}")

    raw = load_table(source)
    cleaned = clean_table(raw)
    summary = summarize(cleaned)
    insights = generate_insights(cleaned)
    write_outputs(cleaned, summary, insights, Path(args.output))

    print(f"Loaded {len(raw)} rows from {source}")
    print(f"Kept {len(cleaned)} rows after cleaning")
    print("\nInsights:")
    for item in insights:
        print(f"- {item}")
    print(f"\nWrote {args.output}/cleaned.csv and {args.output}/report.json")


if __name__ == "__main__":
    main()
