import argparse
import html
import os

import numpy as np
import pandas as pd

from backtest import run_from_predictions, summarize_period_backtest
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
        result, summary = run_from_predictions(
            predictions,
            policy="top_bottom",
            top_frac=cfg["top_frac"],
            bottom_frac=cfg["bottom_frac"],
            long_only=cfg["long_only"],
            cost_bps=cost_bps,
        )
        period_summary = summarize_period_backtest(result)
        rows.append({
            "policy_name": cfg["policy_name"],
            **summary,
            **period_summary,
        })
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


def summarize_default_periods(
        predictions,
        cost_bps,
        output_path,
        top_frac=0.10,
        bottom_frac=0.10):
    result, _ = run_from_predictions(
        predictions,
        policy="top_bottom",
        top_frac=top_frac,
        bottom_frac=bottom_frac,
        cost_bps=cost_bps,
    )

    def per_period(group):
        long_target = group.loc[group["action"] == 1, "fret12"]
        short_target = group.loc[group["action"] == -1, "fret12"]
        long_mean = long_target.mean() if len(long_target) else 0.0
        short_mean = short_target.mean() if len(short_target) else 0.0
        traded = group[group["action"] != 0]
        return pd.Series({
            "long_count": int((group["action"] == 1).sum()),
            "short_count": int((group["action"] == -1).sum()),
            "period_mean_net_return": group["net_return"].mean(),
            "period_trade_mean_net_return": (
                traded["net_return"].mean() if len(traded) else 0.0),
            "period_long_mean_target": long_mean,
            "period_short_mean_target": short_mean,
            "period_long_short_spread": long_mean - short_mean,
        })

    periods = (
        result.groupby(["date", "interval"], sort=True)
        .apply(per_period)
        .reset_index()
    )
    periods.to_csv(output_path, index=False)
    return periods


def save_bar_svg(path, labels, values, title, y_label):
    width = 920
    height = 520
    left = 86
    right = 32
    top = 62
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_value = min(0.0, min(values))
    max_value = max(0.0, max(values))
    if max_value == min_value:
        max_value = min_value + 1.0
    padding = (max_value - min_value) * 0.12
    min_value -= padding
    max_value += padding

    def y_pos(value):
        return top + (max_value - value) / (max_value - min_value) * plot_h

    zero_y = y_pos(0.0)
    bar_gap = 14
    bar_w = (plot_w - bar_gap * (len(values) - 1)) / len(values)
    colors = ["#2F6BFF" if value >= 0 else "#D64545" for value in values]

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="34" font-family="Arial, sans-serif" '
        'font-size="22" font-weight="700" fill="#1f2933">{}</text>'.format(
            left, html.escape(title)),
        '<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" '
        'stroke="#6b7280" stroke-width="1.2"/>'.format(left, zero_y, width - right),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" '
        'stroke="#9ca3af" stroke-width="1"/>'.format(left, top, height - bottom),
        '<text x="24" y="{}" font-family="Arial, sans-serif" font-size="13" '
        'fill="#4b5563" transform="rotate(-90 24,{})">{}</text>'.format(
            top + plot_h / 2, top + plot_h / 2, html.escape(y_label)),
    ]

    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + i * (bar_w + bar_gap)
        y = min(y_pos(value), zero_y)
        h = abs(y_pos(value) - zero_y)
        svg.append(
            '<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" '
            'rx="2" fill="{}"/>'.format(x, y, bar_w, h, color))
        svg.append(
            '<text x="{:.2f}" y="{}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" fill="#374151">{}</text>'.format(
                x + bar_w / 2, height - 56, html.escape(str(label))))
        value_label = "{:.2f}".format(value)
        value_y = y - 8 if value >= 0 else y + h + 18
        svg.append(
            '<text x="{:.2f}" y="{:.2f}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="12" fill="#111827">{}</text>'.format(
                x + bar_w / 2, value_y, value_label))

    svg.append(
        '<text x="{}" y="{}" font-family="Arial, sans-serif" '
        'font-size="12" fill="#6b7280">单位：bps；正值表示未来 12 分钟平均收益为正</text>'.format(
            left, height - 20))
    svg.append("</svg>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def save_presentation_assets(policy_comparison, deciles, output_dir):
    decile_svg = os.path.join(output_dir, "C_decile_mean_target.svg")
    policy_svg = os.path.join(output_dir, "C_policy_spread.svg")

    save_bar_svg(
        decile_svg,
        labels=deciles["forecast_decile"].astype(str).tolist(),
        values=deciles["mean_target_bps"].tolist(),
        title="Forecast Decile vs. Realized 12-min Return",
        y_label="Mean target return (bps)",
    )

    plot_rows = policy_comparison[
        policy_comparison["policy_name"].isin([
            "top_bottom_5_5",
            "top_bottom_10_10",
            "top_bottom_20_20",
        ])
    ]
    save_bar_svg(
        policy_svg,
        labels=["5/5", "10/10", "20/20"],
        values=(plot_rows["long_short_spread"] * 10000.0).tolist(),
        title="Top-Bottom Policy Spread Sensitivity",
        y_label="Long-short spread (bps)",
    )
    return decile_svg, policy_svg


def write_summary_markdown(path, policy_comparison, deciles, period_summary):
    base = policy_comparison[
        policy_comparison["policy_name"] == "top_bottom_10_10"].iloc[0]
    top_decile = deciles.loc[deciles["forecast_decile"] == 10].iloc[0]
    bottom_decile = deciles.loc[deciles["forecast_decile"] == 1].iloc[0]
    top5 = policy_comparison[
        policy_comparison["policy_name"] == "top_bottom_5_5"].iloc[0]
    top20 = policy_comparison[
        policy_comparison["policy_name"] == "top_bottom_20_20"].iloc[0]

    content = """# C Experiment Summary

## Default Agent

- policy: top-bottom 10% per `date + interval`
- cost: 1 bps per traded row
- samples: {samples:,}
- trade_ratio: {trade_ratio:.4f}
- mean_net_return: {mean_net_return:.8f}
- trade_mean_net_return: {trade_mean_net_return:.8f}
- hit_rate: {hit_rate:.4f}
- long_mean_target: {long_mean_target:.8f}
- short_mean_target: {short_mean_target:.8f}
- long_short_spread: {long_short_spread:.8f} ({spread_bps:.2f} bps)

## Period-Level Check

- period_count: {period_count:,}
- period_long_short_spread: {period_spread:.8f} ({period_spread_bps:.2f} bps)
- positive_spread_period_rate: {positive_spread_rate:.4f}
- period_trade_mean_net_return: {period_trade_mean:.8f}

## Decile Check

- bottom decile mean target: {bottom_bps:.2f} bps
- top decile mean target: {top_bps:.2f} bps
- top minus bottom: {decile_spread_bps:.2f} bps

## Sensitivity

- top-bottom 5% spread: {top5_bps:.2f} bps
- top-bottom 10% spread: {top10_bps:.2f} bps
- top-bottom 20% spread: {top20_bps:.2f} bps

Presentation sentence:

> C does not claim to simulate a full exchange. It checks whether B's forecast can be converted into directionally meaningful buy/hold/sell actions. The top forecast decile earns about {top_bps:.2f} bps over the next 12 minutes, while the bottom decile earns about {bottom_bps:.2f} bps, giving a spread of {decile_spread_bps:.2f} bps.
""".format(
        samples=int(base["samples"]),
        trade_ratio=base["trade_ratio"],
        mean_net_return=base["mean_net_return"],
        trade_mean_net_return=base["trade_mean_net_return"],
        hit_rate=base["hit_rate"],
        long_mean_target=base["long_mean_target"],
        short_mean_target=base["short_mean_target"],
        long_short_spread=base["long_short_spread"],
        spread_bps=base["long_short_spread"] * 10000.0,
        period_count=int(period_summary["period_count"]),
        period_spread=period_summary["period_long_short_spread"],
        period_spread_bps=period_summary["period_long_short_spread"] * 10000.0,
        positive_spread_rate=period_summary["period_positive_spread_rate"],
        period_trade_mean=period_summary["period_trade_mean_net_return"],
        bottom_bps=bottom_decile["mean_target_bps"],
        top_bps=top_decile["mean_target_bps"],
        decile_spread_bps=top_decile["mean_target_bps"] - bottom_decile["mean_target_bps"],
        top5_bps=top5["long_short_spread"] * 10000.0,
        top10_bps=base["long_short_spread"] * 10000.0,
        top20_bps=top20["long_short_spread"] * 10000.0,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


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

    period_path = os.path.join(args.output_dir, "C_default_period_returns.csv")
    periods = summarize_default_periods(
        predictions,
        cost_bps=args.cost_bps,
        output_path=period_path,
    )

    base_period_summary = {
        "period_count": int(len(periods)),
        "period_long_short_spread": float(
            periods["period_long_short_spread"].mean()),
        "period_positive_spread_rate": float(
            (periods["period_long_short_spread"] > 0).mean()),
        "period_trade_mean_net_return": float(
            periods["period_trade_mean_net_return"].mean()),
    }

    summary_path = os.path.join(args.output_dir, "C_experiment_summary.md")
    write_summary_markdown(
        summary_path,
        policy_comparison,
        deciles,
        base_period_summary,
    )

    decile_svg, policy_svg = save_presentation_assets(
        policy_comparison,
        deciles,
        args.output_dir,
    )

    base = policy_comparison[
        policy_comparison["policy_name"] == "top_bottom_10_10"].iloc[0]
    top_decile = deciles.loc[deciles["forecast_decile"] == 10, "mean_target"].iloc[0]
    bottom_decile = deciles.loc[deciles["forecast_decile"] == 1, "mean_target"].iloc[0]

    log.inf("Saved C policy comparison to {}".format(policy_path))
    log.inf("Saved C decile analysis to {}".format(decile_path))
    log.inf("Saved C default period returns to {}".format(period_path))
    log.inf("Saved C summary markdown to {}".format(summary_path))
    log.inf("Saved C presentation SVGs to {}, {}".format(decile_svg, policy_svg))
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
