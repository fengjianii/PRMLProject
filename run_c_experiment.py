import argparse
import os

import numpy as np
import pandas as pd

from backtest import run_from_predictions
from log import log


def compare_policies(predictions, cost_bps):
    """Run a small, report-friendly policy comparison grid."""
    configs = [
        {
            "policy_name": "top_bottom_5_5",
            "top_frac": 0.05,
            "bottom_frac": 0.05,
            "long_only": False,
        },
        {
            "policy_name": "top_bottom_10_10",
            "top_frac": 0.10,
            "bottom_frac": 0.10,
            "long_only": False,
        },
        {
            "policy_name": "top_bottom_20_20",
            "top_frac": 0.20,
            "bottom_frac": 0.20,
            "long_only": False,
        },
        {
            "policy_name": "long_only_top_10",
            "top_frac": 0.10,
            "bottom_frac": 0.0,
            "long_only": True,
        },
    ]
    rows = []
    for cfg in configs:
        _, summary = run_from_predictions(
            predictions,
            policy="top_bottom",
            top_frac=cfg["top_frac"],
            bottom_frac=cfg["bottom_frac"],
            long_only=cfg["long_only"],
            cost_bps=cost_bps,
        )
        rows.append({"policy_name": cfg["policy_name"], **summary})
    return pd.DataFrame(rows)


def summarize_deciles(
        predictions,
        prediction_col="forecast",
        target_col="fret12",
        group_cols=("date", "interval")):
    """Summarize realized returns by cross-sectional forecast decile."""
    required = [prediction_col, target_col] + list(group_cols)
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(missing)))

    out = predictions[required].copy()
    out[[prediction_col, target_col]] = (
        out[[prediction_col, target_col]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    ranks = out.groupby(list(group_cols))[prediction_col].rank(
        method="first", pct=True)
    out["forecast_decile"] = np.ceil(ranks * 10).clip(1, 10).astype(int)

    summary = (
        out.groupby("forecast_decile")
        .agg(
            samples=(target_col, "size"),
            mean_forecast=(prediction_col, "mean"),
            mean_target=(target_col, "mean"),
        )
        .reset_index()
    )
    summary["mean_target_bps"] = summary["mean_target"] * 10000.0
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run C's report-ready Trading Agent experiments.")
    parser.add_argument("--predictions", required=True,
                        help="CSV file with forecast and fret12 columns.")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory for C experiment summaries.")
    parser.add_argument("--cost-bps", type=float, default=1.0,
                        help="Per-trade simplified cost in basis points.")
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    os.makedirs(args.output_dir, exist_ok=True)

    policy_comparison = compare_policies(predictions, cost_bps=args.cost_bps)
    policy_path = os.path.join(args.output_dir, "C_policy_comparison.csv")
    policy_comparison.to_csv(policy_path, index=False)

    deciles = summarize_deciles(predictions)
    decile_path = os.path.join(args.output_dir, "C_decile_analysis.csv")
    deciles.to_csv(decile_path, index=False)

    base = policy_comparison[
        policy_comparison["policy_name"] == "top_bottom_10_10"].iloc[0]
    top_decile = deciles.loc[deciles["forecast_decile"] == 10, "mean_target"].iloc[0]
    bottom_decile = deciles.loc[deciles["forecast_decile"] == 1, "mean_target"].iloc[0]

    log.inf("Saved C policy comparison to {}".format(policy_path))
    log.inf("Saved C decile analysis to {}".format(decile_path))
    log.inf(
        "Default C policy: spread={:.8f}, mean_net={:.8f}, hit_rate={:.8f}".format(
            base["long_short_spread"],
            base["mean_net_return"],
            base["hit_rate"],
        )
    )
    log.inf(
        "Forecast decile spread: top10 mean - bottom10 mean = {:.8f}".format(
            top_decile - bottom_decile
        )
    )


if __name__ == "__main__":
    main()
