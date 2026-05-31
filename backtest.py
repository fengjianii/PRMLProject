import argparse
import os

import numpy as np
import pandas as pd

from agent_policy import (
    ACTION_COL,
    clean_prediction_values,
    describe_actions,
    threshold_policy,
    top_bottom_policy,
)
from log import log


def run_signal_backtest(
        df,
        target_col="fret12",
        action_col=ACTION_COL,
        cost_bps=0.0):
    """Evaluate one-step signal returns from actions and realized returns.

    This is a lightweight signal validation, not a full exchange simulator.
    It does not model order matching, queue position, slippage, or overlapping
    portfolio holdings.
    """
    out = clean_prediction_values(df, columns=(target_col,))
    if target_col not in out.columns:
        raise ValueError("Missing target column: {}".format(target_col))
    if action_col not in out.columns:
        raise ValueError("Missing action column: {}".format(action_col))

    out[action_col] = out[action_col].fillna(0).astype(int)
    cost = float(cost_bps) / 10000.0
    out["gross_return"] = out[action_col] * out[target_col]
    out["transaction_cost"] = cost * out[action_col].abs()
    out["net_return"] = out["gross_return"] - out["transaction_cost"]
    return out


def summarize_backtest(df, target_col="fret12", action_col=ACTION_COL):
    """Return report-friendly strategy summary metrics."""
    if len(df) == 0:
        raise ValueError("Cannot summarize an empty backtest")

    traded = df[df[action_col] != 0]
    summary = describe_actions(df, action_col=action_col)
    summary.update({
        "mean_gross_return": float(df["gross_return"].mean()),
        "mean_net_return": float(df["net_return"].mean()),
        "sum_net_return": float(df["net_return"].sum()),
        "trade_mean_net_return": float(
            traded["net_return"].mean()) if len(traded) else 0.0,
        "hit_rate": float(
            (traded["gross_return"] > 0).mean()) if len(traded) else 0.0,
        "long_mean_target": float(
            df.loc[df[action_col] == 1, target_col].mean())
            if (df[action_col] == 1).any() else 0.0,
        "short_mean_target": float(
            df.loc[df[action_col] == -1, target_col].mean())
            if (df[action_col] == -1).any() else 0.0,
    })
    summary["long_short_spread"] = (
        summary["long_mean_target"] - summary["short_mean_target"])
    return summary


def format_summary(summary):
    lines = ["Agent signal backtest summary:"]
    ordered_keys = [
        "samples",
        "long_count",
        "short_count",
        "hold_count",
        "trade_ratio",
        "mean_gross_return",
        "mean_net_return",
        "sum_net_return",
        "trade_mean_net_return",
        "hit_rate",
        "long_mean_target",
        "short_mean_target",
        "long_short_spread",
    ]
    for key in ordered_keys:
        if key in summary:
            value = summary[key]
            if isinstance(value, float):
                lines.append("  {}={:.8f}".format(key, value))
            else:
                lines.append("  {}={}".format(key, value))
    return "\n".join(lines)


def build_policy(df, args):
    if args.policy == "top_bottom":
        return top_bottom_policy(
            df,
            prediction_col=args.prediction_col,
            group_cols=tuple(args.group_cols),
            top_frac=args.top_frac,
            bottom_frac=args.bottom_frac,
            allow_short=not args.long_only,
        )
    return threshold_policy(
        df,
        prediction_col=args.prediction_col,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        allow_short=not args.long_only,
    )


def run_from_predictions(
        predictions,
        policy="top_bottom",
        prediction_col="forecast",
        target_col="fret12",
        group_cols=("date", "interval"),
        top_frac=0.10,
        bottom_frac=0.10,
        buy_threshold=0.0,
        sell_threshold=0.0,
        long_only=False,
        cost_bps=0.0):
    args = argparse.Namespace(
        policy=policy,
        prediction_col=prediction_col,
        target_col=target_col,
        group_cols=list(group_cols),
        top_frac=top_frac,
        bottom_frac=bottom_frac,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        long_only=long_only,
        cost_bps=cost_bps,
    )
    signals = build_policy(predictions, args)
    result = run_signal_backtest(
        signals,
        target_col=target_col,
        action_col=ACTION_COL,
        cost_bps=cost_bps,
    )
    return result, summarize_backtest(result, target_col=target_col)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a lightweight Trading Agent signal backtest.")
    parser.add_argument("--predictions", required=True,
                        help="CSV file with forecast and fret12 columns.")
    parser.add_argument("--output", default=None,
                        help="Optional CSV path for per-row backtest output.")
    parser.add_argument("--summary-output", default=None,
                        help="Optional CSV path for one-row summary output.")
    parser.add_argument("--policy", choices=("top_bottom", "threshold"),
                        default="top_bottom")
    parser.add_argument("--prediction-col", default="forecast")
    parser.add_argument("--target-col", default="fret12")
    parser.add_argument("--group-cols", nargs="+",
                        default=["date", "interval"])
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--bottom-frac", type=float, default=0.10)
    parser.add_argument("--buy-threshold", type=float, default=0.0)
    parser.add_argument("--sell-threshold", type=float, default=0.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--cost-bps", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = pd.read_csv(args.predictions)
    result, summary = run_from_predictions(
        predictions,
        policy=args.policy,
        prediction_col=args.prediction_col,
        target_col=args.target_col,
        group_cols=tuple(args.group_cols),
        top_frac=args.top_frac,
        bottom_frac=args.bottom_frac,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        long_only=args.long_only,
        cost_bps=args.cost_bps,
    )
    log.inf(format_summary(summary))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result.to_csv(args.output, index=False)
        log.inf("Saved row-level backtest output to {}".format(args.output))
    if args.summary_output:
        os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
        pd.DataFrame([summary]).to_csv(args.summary_output, index=False)
        log.inf("Saved backtest summary to {}".format(args.summary_output))


if __name__ == "__main__":
    main()
