import argparse
import html
import os

import numpy as np
import pandas as pd

from agent_policy import top_bottom_policy
from log import log


JOIN_KEYS = ["symbol", "date", "interval"]
MARKET_COLS = [
    "symbol",
    "interval",
    "midpx",
    "bid0",
    "ask0",
    "bsize0",
    "asize0",
    "bsize0_4",
    "asize0_4",
    "tradeBuyQty",
    "tradeSellQty",
]


def clean_numeric(frame, columns):
    for col in columns:
        frame[col] = (
            frame[col].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    return frame


def load_market_data(data_dir, dates):
    frames = []
    for date in sorted(set(int(x) for x in dates)):
        path = os.path.join(data_dir, "{}.h5".format(date))
        if not os.path.exists(path):
            raise ValueError("Missing h5 file: {}".format(path))
        df = pd.read_hdf(path)
        df = df[[col for col in MARKET_COLS if col in df.columns]]
        df["date"] = date
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def attach_market_data(predictions, data_dir):
    market = load_market_data(data_dir, predictions["date"].unique())
    merged = predictions.merge(market, on=JOIN_KEYS, how="left", validate="one_to_one")
    missing = merged["midpx"].isna().sum()
    if missing:
        raise ValueError("Missing market rows after join: {}".format(missing))
    return merged


def prepare_signals(args):
    predictions = pd.read_csv(args.predictions)
    signals = top_bottom_policy(
        predictions,
        prediction_col=args.prediction_col,
        group_cols=("date", "interval"),
        top_frac=args.top_frac,
        bottom_frac=args.bottom_frac,
        allow_short=not args.long_only,
    )
    data = attach_market_data(signals, args.data_dir)
    data = clean_numeric(data, [
        "forecast",
        "fret12",
        "midpx",
        "bid0",
        "ask0",
        "bsize0",
        "asize0",
        "bsize0_4",
        "asize0_4",
        "tradeBuyQty",
        "tradeSellQty",
    ])
    return data.sort_values(JOIN_KEYS).reset_index(drop=True)


def add_exit_prices(data, horizon):
    data = data.sort_values(["symbol", "date", "interval"]).copy()
    group = data.groupby(["symbol", "date"], sort=False)
    data["exit_mid"] = group["midpx"].shift(-horizon)
    data["exit_bid0"] = group["bid0"].shift(-horizon)
    data["exit_ask0"] = group["ask0"].shift(-horizon)

    fallback_mid = data["midpx"] * (1.0 + data["fret12"])
    data["exit_mid"] = data["exit_mid"].fillna(fallback_mid)
    spread = (data["ask0"] - data["bid0"]).clip(lower=0.0)
    data["exit_bid0"] = data["exit_bid0"].fillna(data["exit_mid"] - spread / 2.0)
    data["exit_ask0"] = data["exit_ask0"].fillna(data["exit_mid"] + spread / 2.0)
    return data


def apply_position_caps(trades, notional_per_trade, max_gross_notional):
    trades = trades.copy()
    trades["base_order_notional"] = float(notional_per_trade)
    desired = trades.groupby(["date", "interval"])["base_order_notional"].transform("sum")
    scale = np.minimum(1.0, float(max_gross_notional) / desired.replace(0.0, np.nan))
    trades["position_scale"] = scale.fillna(0.0)
    trades["order_notional"] = trades["base_order_notional"] * trades["position_scale"]
    trades["order_qty"] = trades["order_notional"] / trades["midpx"].replace(0.0, np.nan)
    trades["order_qty"] = trades["order_qty"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return trades


def market_execution(trades, impact_k):
    side = trades["action"]
    spread = (trades["ask0"] - trades["bid0"]).clip(lower=0.0)
    depth = np.where(side > 0, trades["asize0_4"], trades["bsize0_4"])
    depth = pd.Series(depth, index=trades.index).replace(0.0, np.nan)
    participation = (trades["order_qty"] / depth).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    impact_bps = float(impact_k) * participation.clip(lower=0.0, upper=10.0)
    impact_px = trades["midpx"] * impact_bps / 10000.0

    entry_price = trades["midpx"] + side * (spread / 2.0 + impact_px)
    fill_qty = trades["order_qty"]
    fill_ratio = pd.Series(1.0, index=trades.index)
    return fill_qty, entry_price, fill_ratio, impact_bps


def limit_execution(trades, queue_ahead_frac):
    side = trades["action"]
    queue_depth = np.where(side > 0, trades["bsize0"], trades["asize0"])
    opposite_trade_qty = np.where(
        side > 0, trades["tradeSellQty"], trades["tradeBuyQty"])
    queue_depth = pd.Series(queue_depth, index=trades.index).clip(lower=0.0)
    opposite_trade_qty = pd.Series(opposite_trade_qty, index=trades.index).clip(lower=0.0)

    queue_ahead = queue_depth * float(queue_ahead_frac)
    fillable = (opposite_trade_qty - queue_ahead).clip(lower=0.0)
    fill_qty = np.minimum(trades["order_qty"], fillable)
    entry_price = np.where(side > 0, trades["bid0"], trades["ask0"])
    entry_price = pd.Series(entry_price, index=trades.index)
    participation = (fill_qty / trades["order_qty"].replace(0.0, np.nan)).fillna(0.0)
    return fill_qty, entry_price, participation, pd.Series(0.0, index=trades.index)


def hybrid_execution(trades, queue_ahead_frac, impact_k):
    limit_qty, limit_price, limit_participation, _ = limit_execution(
        trades, queue_ahead_frac)
    remaining = (trades["order_qty"] - limit_qty).clip(lower=0.0)
    market_side = trades.copy()
    market_side["order_qty"] = remaining
    market_qty, market_price, participation, impact_bps = market_execution(
        market_side, impact_k)
    fill_qty = limit_qty + market_qty
    value = limit_qty * limit_price + market_qty * market_price
    entry_price = value / fill_qty.replace(0.0, np.nan)
    entry_price = entry_price.fillna(market_price)
    effective_participation = (
        fill_qty / trades["order_qty"].replace(0.0, np.nan)).fillna(0.0)
    blended_impact = impact_bps * (
        market_qty / fill_qty.replace(0.0, np.nan)).fillna(0.0)
    return fill_qty, entry_price, effective_participation, blended_impact


def simulate_execution(data, args):
    data = add_exit_prices(data, args.horizon)
    trades = data[data["action"] != 0].copy()
    trades = trades[(trades["midpx"] > 0) & (trades["orderable"].fillna(True)
                    if "orderable" in trades.columns else True)]
    trades = apply_position_caps(
        trades,
        notional_per_trade=args.notional_per_trade,
        max_gross_notional=args.max_gross_notional,
    )
    trades = trades[trades["order_qty"] > 0].copy()
    submitted_orders = int(len(trades))

    if args.order_style == "market":
        fill_qty, entry_price, fill_ratio, impact_bps = market_execution(
            trades, args.impact_k)
    elif args.order_style == "limit":
        fill_qty, entry_price, fill_ratio, impact_bps = limit_execution(
            trades, args.queue_ahead_frac)
    else:
        fill_qty, entry_price, fill_ratio, impact_bps = hybrid_execution(
            trades, args.queue_ahead_frac, args.impact_k)

    side = trades["action"]
    exit_spread = (trades["exit_ask0"] - trades["exit_bid0"]).clip(lower=0.0)
    exit_impact_px = trades["exit_mid"] * float(args.exit_impact_bps) / 10000.0
    exit_price = trades["exit_mid"] - side * (exit_spread / 2.0 + exit_impact_px)

    trades["fill_qty"] = fill_qty
    trades["fill_ratio"] = fill_ratio.clip(lower=0.0, upper=1.0)
    trades["entry_price"] = entry_price
    trades["exit_price"] = exit_price
    trades["impact_bps"] = impact_bps
    trades["filled_notional"] = trades["fill_qty"] * trades["entry_price"]
    trades["gross_pnl"] = side * (trades["exit_price"] - trades["entry_price"]) * trades["fill_qty"]
    trades["fee"] = trades["filled_notional"].abs() * float(args.fee_bps) / 10000.0
    trades["net_pnl"] = trades["gross_pnl"] - trades["fee"]
    trades["execution_return"] = (
        trades["net_pnl"] / trades["filled_notional"].replace(0.0, np.nan))
    trades["execution_return"] = trades["execution_return"].replace(
        [np.inf, -np.inf], np.nan).fillna(0.0)
    mean_fill_ratio_all = float(trades["fill_ratio"].mean()) if len(trades) else 0.0
    executed = trades[trades["fill_qty"] > 0].copy()
    executed.attrs["submitted_orders"] = submitted_orders
    executed.attrs["mean_fill_ratio_all"] = mean_fill_ratio_all
    return executed


def build_equity_curve(trades, initial_capital):
    if len(trades) == 0:
        return pd.DataFrame(columns=[
            "date",
            "interval",
            "period_pnl",
            "filled_notional",
            "long_notional",
            "short_notional",
            "equity",
            "period_return",
            "drawdown",
        ])
    grouped = trades.groupby(["date", "interval"], sort=True).agg(
        period_pnl=("net_pnl", "sum"),
        filled_notional=("filled_notional", "sum"),
        long_notional=("filled_notional", lambda x: x[trades.loc[x.index, "action"] == 1].sum()),
        short_notional=("filled_notional", lambda x: x[trades.loc[x.index, "action"] == -1].sum()),
        trade_count=("net_pnl", "size"),
        mean_execution_return=("execution_return", "mean"),
    ).reset_index()
    grouped["equity"] = float(initial_capital) + grouped["period_pnl"].cumsum()
    grouped["period_return"] = grouped["period_pnl"] / float(initial_capital)
    running_max = grouped["equity"].cummax()
    grouped["drawdown"] = grouped["equity"] / running_max - 1.0
    return grouped


def summarize_execution(trades, equity, args):
    submitted_orders = int(trades.attrs.get("submitted_orders", len(trades)))
    mean_fill_ratio_all = float(trades.attrs.get(
        "mean_fill_ratio_all",
        trades["fill_ratio"].mean() if len(trades) else 0.0,
    ))
    if len(trades) == 0:
        return {
            "order_style": args.order_style,
            "orders": 0,
            "submitted_orders": submitted_orders,
            "filled_orders": 0,
            "fill_rate": 0.0,
            "mean_fill_ratio": mean_fill_ratio_all,
            "final_equity": float(args.initial_capital),
            "total_pnl": 0.0,
        }
    return {
        "order_style": args.order_style,
        "orders": submitted_orders,
        "submitted_orders": submitted_orders,
        "filled_orders": int((trades["fill_qty"] > 0).sum()),
        "fill_rate": float((trades["fill_qty"] > 0).sum() / max(submitted_orders, 1)),
        "mean_fill_ratio": mean_fill_ratio_all,
        "avg_filled_notional": float(trades["filled_notional"].mean()),
        "mean_execution_return": float(trades["execution_return"].mean()),
        "trade_hit_rate": float((trades["net_pnl"] > 0).mean()),
        "total_pnl": float(trades["net_pnl"].sum()),
        "final_equity": float(equity["equity"].iloc[-1]) if len(equity) else float(args.initial_capital),
        "max_drawdown": float(equity["drawdown"].min()) if len(equity) else 0.0,
        "period_count": int(len(equity)),
        "positive_period_rate": float((equity["period_pnl"] > 0).mean()) if len(equity) else 0.0,
        "mean_period_pnl": float(equity["period_pnl"].mean()) if len(equity) else 0.0,
    }


def save_equity_svg(path, equity):
    width = 940
    height = 500
    left = 84
    right = 28
    top = 58
    bottom = 62
    if len(equity) == 0:
        return
    values = equity["equity"].to_numpy()
    min_value = float(values.min())
    max_value = float(values.max())
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    padding = (max_value - min_value) * 0.08
    min_value -= padding
    max_value += padding
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_pos(i):
        if len(values) == 1:
            return left
        return left + i / (len(values) - 1) * plot_w

    def y_pos(value):
        return top + (max_value - value) / (max_value - min_value) * plot_h

    points = " ".join(
        "{:.2f},{:.2f}".format(x_pos(i), y_pos(v))
        for i, v in enumerate(values)
    )
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">Execution Equity Curve</text>'.format(left),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#9ca3af"/>'.format(left, top, height - bottom),
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#9ca3af"/>'.format(left, height - bottom, width - right),
        '<polyline fill="none" stroke="#2563eb" stroke-width="2.4" points="{}"/>'.format(points),
        '<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">Final equity: {:.2f}</text>'.format(left, height - 22, values[-1]),
        "</svg>",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def write_report(path, summary, args):
    lines = [
        "# Agent Execution Simulation Report",
        "",
        "This is a simplified execution simulator built from order-book snapshots and aggregate trades.",
        "It is not a real exchange matching engine.",
        "",
        "## Settings",
        "",
        "- order_style: `{}`".format(args.order_style),
        "- notional_per_trade: {:.2f}".format(args.notional_per_trade),
        "- max_gross_notional_per_period: {:.2f}".format(args.max_gross_notional),
        "- queue_ahead_frac: {:.2f}".format(args.queue_ahead_frac),
        "- impact_k: {:.2f}".format(args.impact_k),
        "- fee_bps: {:.2f}".format(args.fee_bps),
        "- exit_impact_bps: {:.2f}".format(args.exit_impact_bps),
        "",
        "## Results",
        "",
        "```text",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append("{} = {:.8f}".format(key, value))
        else:
            lines.append("{} = {}".format(key, value))
    lines.extend([
        "```",
        "",
        "## Interpretation",
        "",
        "- Matching: market orders cross the spread; limit orders fill only when opposite-side trade quantity exceeds a queue-ahead estimate.",
        "- Slippage: marketable quantity consumes displayed depth and adds impact proportional to participation.",
        "- Queue: unfilled passive quantity is canceled at the end of the interval in this approximation.",
        "- Position management: each date+interval gross notional is capped before execution.",
        "- Equity curve: period PnL is aggregated by date+interval and accumulated from initial capital.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simplified execution simulator for C Trading Agent.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--prediction-col", default="forecast")
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--bottom-frac", type=float, default=0.10)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--order-style", choices=("market", "limit", "hybrid"),
                        default="hybrid")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--notional-per-trade", type=float, default=10000.0)
    parser.add_argument("--max-gross-notional", type=float, default=10000000.0)
    parser.add_argument("--queue-ahead-frac", type=float, default=0.50)
    parser.add_argument("--impact-k", type=float, default=2.0,
                        help="Impact bps per 100% displayed-depth participation.")
    parser.add_argument("--exit-impact-bps", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=0.5)
    parser.add_argument("--initial-capital", type=float, default=10000000.0)
    parser.add_argument("--save-trades", action="store_true")
    parser.add_argument("--output-prefix", default="C_execution",
                        help="Output file prefix inside output-dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    data = prepare_signals(args)
    trades = simulate_execution(data, args)
    equity = build_equity_curve(trades, args.initial_capital)
    summary = summarize_execution(trades, equity, args)

    prefix = args.output_prefix
    summary_path = os.path.join(args.output_dir, "{}_summary.csv".format(prefix))
    equity_path = os.path.join(args.output_dir, "{}_equity_curve.csv".format(prefix))
    report_path = os.path.join(args.output_dir, "{}_report.md".format(prefix))
    svg_path = os.path.join(args.output_dir, "{}_equity_curve.svg".format(prefix))
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    equity.to_csv(equity_path, index=False)
    write_report(report_path, summary, args)
    save_equity_svg(svg_path, equity)
    if args.save_trades:
        trades.to_csv(os.path.join(
            args.output_dir, "{}_trades.csv".format(prefix)), index=False)

    log.inf("Saved execution summary to {}".format(summary_path))
    log.inf("Saved equity curve to {}".format(equity_path))
    log.inf("Saved execution report to {}".format(report_path))
    log.inf("Saved equity SVG to {}".format(svg_path))
    log.inf("Execution summary: {}".format(summary))


if __name__ == "__main__":
    main()
