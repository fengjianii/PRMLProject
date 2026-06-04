import argparse
import os

import numpy as np
import pandas as pd


OUTPUT_FILES = {
    "policy": "C_policy_comparison.csv",
    "decile": "C_decile_analysis.csv",
    "factor": "C_factor_mining_summary.csv",
    "factor_family": "C_factor_family_summary.csv",
    "factor_agent": "C_factor_agent_summary.csv",
    "default_period": "C_default_period_returns.csv",
    "execution_market": "C_execution_market_summary.csv",
    "execution_hybrid": "C_execution_hybrid_summary.csv",
    "execution_limit": "C_execution_limit_summary.csv",
    "rl_summary": "C_rl_policy_summary.csv",
    "rl_q": "C_rl_q_table.csv",
}


def read_csv_if_exists(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def read_outputs(output_dir):
    frames = {}
    for name, filename in OUTPUT_FILES.items():
        frames[name] = read_csv_if_exists(os.path.join(output_dir, filename))
    return frames


def first_row(frame):
    if frame is None or len(frame) == 0:
        return None
    return frame.iloc[0]


def select_row(frame, column, value):
    if frame is None or column not in frame.columns:
        return None
    rows = frame[frame[column] == value]
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def value(row, key, default=np.nan):
    if row is None:
        return default
    if key not in row:
        return default
    return row[key]


def fmt_number(x, digits=4, missing="missing"):
    if pd.isna(x):
        return missing
    if isinstance(x, (int, np.integer)):
        return "{:,}".format(int(x))
    return ("{:,." + str(digits) + "f}").format(float(x))


def fmt_decimal(x, digits=8, missing="missing"):
    if pd.isna(x):
        return missing
    return ("{:." + str(digits) + "f}").format(float(x))


def fmt_pct(x, digits=2, missing="missing"):
    if pd.isna(x):
        return missing
    return ("{:." + str(digits) + "f}%").format(float(x) * 100.0)


def fmt_bps(x, digits=2, missing="missing"):
    if pd.isna(x):
        return missing
    return ("{:." + str(digits) + "f} bps").format(float(x) * 10000.0)


def fmt_money(x, missing="missing"):
    if pd.isna(x):
        return missing
    return "{:,.2f}".format(float(x))


def format_symbol(symbol):
    if pd.isna(symbol):
        return ""
    if isinstance(symbol, (int, np.integer)):
        return str(int(symbol))
    if isinstance(symbol, (float, np.floating)) and float(symbol).is_integer():
        return str(int(symbol))
    text = str(symbol)
    if text.endswith(".0"):
        return text[:-2]
    return text


def build_decile_lookup(deciles):
    if deciles is None or len(deciles) == 0:
        return {}
    if "forecast_decile" not in deciles.columns:
        return {}
    if "mean_target_bps" in deciles.columns:
        return {
            int(row["forecast_decile"]): float(row["mean_target_bps"])
            for _, row in deciles.iterrows()
        }
    if "mean_target" in deciles.columns:
        return {
            int(row["forecast_decile"]): float(row["mean_target"]) * 10000.0
            for _, row in deciles.iterrows()
        }
    return {}


def classify_examples(predictions, decile_lookup, cost_bps, n_each):
    required = ["symbol", "date", "interval", "forecast", "fret12"]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise ValueError("Missing required prediction columns: {}".format(
            ", ".join(missing)))

    work = predictions[required].copy()
    work[["forecast", "fret12"]] = (
        work[["forecast", "fret12"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    ranks = work.groupby(["date", "interval"])["forecast"].rank(
        method="first", pct=True)
    work["forecast_percentile"] = ranks
    work["forecast_decile"] = np.ceil(ranks * 10).clip(1, 10).astype(int)

    long_pool = work[work["forecast_percentile"] > 0.90].sort_values(
        ["forecast_percentile", "forecast"], ascending=[False, False])
    short_pool = work[work["forecast_percentile"] <= 0.10].sort_values(
        ["forecast_percentile", "forecast"], ascending=[True, True])
    hold_pool = work[
        (work["forecast_percentile"] >= 0.45)
        & (work["forecast_percentile"] <= 0.55)
    ].copy()
    hold_pool["middle_distance"] = (hold_pool["forecast_percentile"] - 0.50).abs()
    hold_pool = hold_pool.sort_values(["middle_distance", "forecast"])

    examples = []
    for decision, action, side, pool in [
            ("buy", 1, 1.0, long_pool.head(n_each)),
            ("sell_short", -1, -1.0, short_pool.head(n_each)),
            ("hold", 0, 0.0, hold_pool.head(n_each)),
    ]:
        for _, row in pool.iterrows():
            decile = int(row["forecast_decile"])
            group_mean_bps = decile_lookup.get(decile, np.nan)
            expected_reward_bps = (
                side * group_mean_bps - float(cost_bps)
                if action != 0 and not pd.isna(group_mean_bps)
                else 0.0
            )
            realized_reward_bps = (
                side * float(row["fret12"]) * 10000.0 - float(cost_bps)
                if action != 0
                else 0.0
            )
            if decision == "buy":
                reason = (
                    "forecast percentile is in the top decile; "
                    "Agent converts high cross-sectional rank into a buy signal."
                )
            elif decision == "sell_short":
                reason = (
                    "forecast percentile is in the bottom decile; "
                    "Agent treats weak rank as a short signal."
                )
            else:
                reason = (
                    "forecast percentile is near the middle; "
                    "Agent avoids a low-conviction trade."
                )
            examples.append({
                "symbol": format_symbol(row["symbol"]),
                "date": int(row["date"]),
                "interval": int(row["interval"]),
                "forecast": float(row["forecast"]),
                "forecast_percentile": float(row["forecast_percentile"]),
                "forecast_decile": decile,
                "decision": decision,
                "action": action,
                "expected_group_mean_bps": group_mean_bps,
                "expected_action_reward_bps": expected_reward_bps,
                "realized_fret12_bps": float(row["fret12"]) * 10000.0,
                "realized_action_reward_bps": realized_reward_bps,
                "reason": reason,
            })
    return pd.DataFrame(examples)


def write_examples_markdown(path, examples):
    lines = [
        "# C Agent Decision Examples",
        "",
        "These examples are selected from the exported B prediction file.",
        "They show how the C Agent turns one cross-sectional forecast into buy / sell / hold decisions.",
        "",
        "| symbol | date | interval | forecast_pct | decile | decision | expected_reward_bps | realized_reward_bps | reason |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for _, row in examples.iterrows():
        lines.append("| {} | {} | {} | {:.2f}% | {} | {} | {:.2f} | {:.2f} | {} |".format(
            row["symbol"],
            int(row["date"]),
            int(row["interval"]),
            float(row["forecast_percentile"]) * 100.0,
            int(row["forecast_decile"]),
            row["decision"],
            float(row["expected_action_reward_bps"]),
            float(row["realized_action_reward_bps"]),
            row["reason"],
        ))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def best_factor_row(factor_frame):
    if factor_frame is None or len(factor_frame) == 0:
        return None
    if "feature" in factor_frame.columns:
        # The mining report is already ordered for presentation priority.
        return factor_frame.iloc[0]
    if "abs_mean_ic" in factor_frame.columns:
        return factor_frame.sort_values("abs_mean_ic", ascending=False).iloc[0]
    if "mean_ic" in factor_frame.columns:
        work = factor_frame.copy()
        work["_abs_mean_ic"] = work["mean_ic"].abs()
        return work.sort_values("_abs_mean_ic", ascending=False).iloc[0]
    return factor_frame.iloc[0]


def build_dashboard(frames, examples, output_dir):
    policy = select_row(frames["policy"], "policy_name", "top_bottom_10_10")
    if policy is None:
        policy = first_row(frames["policy"])

    deciles = frames["decile"]
    top_decile = select_row(deciles, "forecast_decile", 10)
    bottom_decile = select_row(deciles, "forecast_decile", 1)
    factor = best_factor_row(frames["factor"])
    factor_agent = first_row(frames["factor_agent"])
    exec_market = first_row(frames["execution_market"])
    exec_hybrid = first_row(frames["execution_hybrid"])
    exec_limit = first_row(frames["execution_limit"])
    rl = first_row(frames["rl_summary"])

    present_files = [
        filename for filename in OUTPUT_FILES.values()
        if os.path.exists(os.path.join(output_dir, filename))
    ]
    generated_files = [
        "C_AGENT_DASHBOARD.md",
        "C_agent_decision_examples.csv",
        "C_agent_decision_examples.md",
    ]

    lines = [
        "# C Agent Controller Dashboard",
        "",
        "This file is generated by `agent_controller.py`.",
        "It summarizes the current C-side Trading Agent without retraining A/B models.",
        "",
        "## Agent Status",
        "",
        "```text",
        "market state -> A/B features -> B GBT forecast -> C Agent Controller",
        "C Agent Controller -> signal policy / factor mining / execution simulation / RL policy selector",
        "```",
        "",
        "## Key Numbers",
        "",
        "- B final model Pearson: `0.0677`",
        "- B final model R2: `0.00443`",
        "- default signal policy: `top-bottom 10%`, cost `1 bps`",
        "- signal long-short spread: `{}`".format(
            fmt_bps(value(policy, "long_short_spread"))),
        "- signal trade ratio: `{}`".format(
            fmt_pct(value(policy, "trade_ratio"))),
        "- period positive spread rate: `{}`".format(
            fmt_pct(value(policy, "period_positive_spread_rate"))),
        "- top decile realized mean: `{}`".format(
            fmt_bps(value(top_decile, "mean_target"))),
        "- bottom decile realized mean: `{}`".format(
            fmt_bps(value(bottom_decile, "mean_target"))),
        "- best mined factor: `{}` with mean IC `{}` and spread `{}`".format(
            value(factor, "feature", "missing"),
            fmt_decimal(value(factor, "mean_ic"), digits=6),
            fmt_number(value(factor, "top_bottom_spread_bps"), digits=2) + " bps"
            if not pd.isna(value(factor, "top_bottom_spread_bps")) else "missing",
        ),
        "- factor-aware diagnostic spread: `{}`".format(
            fmt_bps(value(factor_agent, "long_short_spread"))),
        "- market execution final equity: `{}`".format(
            fmt_money(value(exec_market, "final_equity"))),
        "- hybrid execution final equity: `{}`".format(
            fmt_money(value(exec_hybrid, "final_equity"))),
        "- limit execution final equity: `{}`, fill rate `{}`, max drawdown `{}`".format(
            fmt_money(value(exec_limit, "final_equity")),
            fmt_pct(value(exec_limit, "fill_rate")),
            fmt_pct(value(exec_limit, "max_drawdown")),
        ),
        "- RL selector final equity: `{}`, positive reward rate `{}`".format(
            fmt_money(value(rl, "final_equity")),
            fmt_pct(value(rl, "positive_reward_rate")),
        ),
        "",
        "## Decision Examples",
        "",
        "The Controller also exports concrete examples in `C_agent_decision_examples.csv`.",
        "",
        "| decision | count | avg_expected_reward_bps | avg_realized_reward_bps |",
        "| --- | ---: | ---: | ---: |",
    ]
    if len(examples):
        grouped = examples.groupby("decision").agg(
            count=("decision", "size"),
            avg_expected_reward_bps=("expected_action_reward_bps", "mean"),
            avg_realized_reward_bps=("realized_action_reward_bps", "mean"),
        ).reset_index()
        for _, row in grouped.iterrows():
            lines.append("| {} | {} | {:.2f} | {:.2f} |".format(
                row["decision"],
                int(row["count"]),
                float(row["avg_expected_reward_bps"]),
                float(row["avg_realized_reward_bps"]),
            ))
    else:
        lines.append("| missing | 0 | 0.00 | 0.00 |")

    lines.extend([
        "",
        "## What To Say Tomorrow",
        "",
        "1. C did not replace the prediction model. C added the decision layer after B's forecast.",
        "2. The Agent first checks whether forecast ranking has tradable direction. The default top-bottom 10% spread is about 7.74 bps.",
        "3. The factor-mining module upgrades the Agent from report explanation to candidate strategy generation.",
        "4. The execution simulator shows why order type matters: market orders can be eaten by spread and slippage, while limit orders trade off fill rate for better entry price.",
        "5. The RL selector is a lightweight feedback prototype: state -> action -> reward -> updated policy preference.",
        "",
        "## Boundary",
        "",
        "This is still a research prototype. The execution layer uses order-book snapshots and aggregated trade quantities to approximate matching.",
        "It is closer to Agent trading than a pure signal backtest, but it is not a full exchange-level simulator or production trading system.",
        "",
        "## Output Files",
        "",
        "Existing C outputs:",
    ])
    for filename in present_files:
        lines.append("- `outputs/{}`".format(filename))
    lines.append("")
    lines.append("Controller outputs:")
    for filename in generated_files:
        lines.append("- `outputs/{}`".format(filename))
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize C Trading Agent outputs into a final dashboard.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--examples-per-action", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    frames = read_outputs(args.output_dir)
    predictions = pd.read_csv(args.predictions)
    decile_lookup = build_decile_lookup(frames["decile"])
    examples = classify_examples(
        predictions,
        decile_lookup=decile_lookup,
        cost_bps=args.cost_bps,
        n_each=args.examples_per_action,
    )

    examples_csv = os.path.join(args.output_dir, "C_agent_decision_examples.csv")
    examples_md = os.path.join(args.output_dir, "C_agent_decision_examples.md")
    dashboard_md = os.path.join(args.output_dir, "C_AGENT_DASHBOARD.md")

    examples.to_csv(examples_csv, index=False)
    write_examples_markdown(examples_md, examples)
    dashboard = build_dashboard(frames, examples, args.output_dir)
    with open(dashboard_md, "w", encoding="utf-8") as f:
        f.write(dashboard)

    print("Saved Agent dashboard to {}".format(dashboard_md))
    print("Saved decision examples to {}".format(examples_csv))
    print("Saved decision example report to {}".format(examples_md))


if __name__ == "__main__":
    main()
