import argparse
import os
import random

import numpy as np
import pandas as pd


ACTIONS = [
    {"name": "hold", "top_frac": 0.0, "bottom_frac": 0.0},
    {"name": "top_bottom_5_5", "top_frac": 0.05, "bottom_frac": 0.05},
    {"name": "top_bottom_10_10", "top_frac": 0.10, "bottom_frac": 0.10},
    {"name": "top_bottom_20_20", "top_frac": 0.20, "bottom_frac": 0.20},
]


def classify_state(group):
    dispersion = group["forecast"].std()
    volatility = group["fret12"].abs().mean()
    if dispersion < 0.00015:
        signal_state = "weak_signal"
    elif dispersion < 0.00035:
        signal_state = "normal_signal"
    else:
        signal_state = "strong_signal"
    if volatility < 0.0015:
        vol_state = "low_vol"
    elif volatility < 0.0035:
        vol_state = "mid_vol"
    else:
        vol_state = "high_vol"
    return signal_state + "|" + vol_state


def reward_for_action(group, action, cost_bps):
    if action["top_frac"] == 0:
        return 0.0, 0, 0.0
    ranks = group["forecast"].rank(method="first", pct=True)
    signal = pd.Series(0, index=group.index)
    signal.loc[ranks > 1.0 - action["top_frac"]] = 1
    signal.loc[ranks <= action["bottom_frac"]] = -1
    gross = signal * group["fret12"]
    cost = float(cost_bps) / 10000.0 * signal.abs()
    net = gross - cost
    traded = signal != 0
    reward = float(net[traded].mean()) if traded.any() else 0.0
    spread = (
        float(group.loc[signal == 1, "fret12"].mean())
        - float(group.loc[signal == -1, "fret12"].mean())
        if (signal == 1).any() and (signal == -1).any()
        else 0.0
    )
    return reward, int(traded.sum()), spread


def run_q_learning(predictions, args):
    rng = random.Random(args.seed)
    q = {}
    rows = []
    equity = float(args.initial_capital)
    grouped = predictions.sort_values(["date", "interval"]).groupby(
        ["date", "interval"], sort=True)

    for step, ((date, interval), group) in enumerate(grouped, 1):
        state = classify_state(group)
        q.setdefault(state, {action["name"]: 0.0 for action in ACTIONS})

        if rng.random() < args.epsilon:
            action = rng.choice(ACTIONS)
            explore = True
        else:
            best_name = max(q[state], key=q[state].get)
            action = next(item for item in ACTIONS if item["name"] == best_name)
            explore = False

        reward, trade_count, spread = reward_for_action(group, action, args.cost_bps)
        old_q = q[state][action["name"]]
        q[state][action["name"]] = old_q + args.alpha * (reward - old_q)
        pnl = reward * trade_count * args.notional_per_trade
        equity += pnl
        rows.append({
            "step": step,
            "date": date,
            "interval": interval,
            "state": state,
            "action": action["name"],
            "explore": explore,
            "reward": reward,
            "trade_count": trade_count,
            "spread": spread,
            "pnl": pnl,
            "equity": equity,
        })
    q_rows = []
    for state, values in q.items():
        for action_name, value in values.items():
            q_rows.append({
                "state": state,
                "action": action_name,
                "q_value": value,
            })
    return pd.DataFrame(rows), pd.DataFrame(q_rows)


def summarize_rl(history):
    if len(history) == 0:
        return {}
    running_max = history["equity"].cummax()
    drawdown = history["equity"] / running_max - 1.0
    traded = history[history["trade_count"] > 0]
    return {
        "steps": int(len(history)),
        "final_equity": float(history["equity"].iloc[-1]),
        "total_pnl": float(history["pnl"].sum()),
        "mean_reward": float(traded["reward"].mean()) if len(traded) else 0.0,
        "positive_reward_rate": float((traded["reward"] > 0).mean()) if len(traded) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "most_used_action": history["action"].value_counts().idxmax(),
    }


def write_report(path, summary, q_table):
    lines = [
        "# Lightweight RL Policy Report",
        "",
        "This module is a small online Q-learning/contextual-bandit prototype.",
        "It chooses among hold, top-bottom 5%, 10%, and 20% policies based on a simple forecast-dispersion/volatility state.",
        "",
        "## Summary",
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
        "## Learned Q Table",
        "",
        "| state | action | q_value |",
        "| --- | --- | ---: |",
    ])
    for _, row in q_table.sort_values(["state", "q_value"], ascending=[True, False]).iterrows():
        lines.append("| {} | {} | {:.8f} |".format(
            row["state"], row["action"], row["q_value"]))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight RL-style adaptive policy selector.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-capital", type=float, default=10000000.0)
    parser.add_argument("--notional-per-trade", type=float, default=10000.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    predictions[["forecast", "fret12"]] = (
        predictions[["forecast", "fret12"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    history, q_table = run_q_learning(predictions, args)
    summary = summarize_rl(history)
    history_path = os.path.join(args.output_dir, "C_rl_policy_history.csv")
    q_path = os.path.join(args.output_dir, "C_rl_q_table.csv")
    report_path = os.path.join(args.output_dir, "C_rl_policy_report.md")
    summary_path = os.path.join(args.output_dir, "C_rl_policy_summary.csv")
    history.to_csv(history_path, index=False)
    q_table.to_csv(q_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    write_report(report_path, summary, q_table)
    print("Saved RL history to {}".format(history_path))
    print("Saved RL Q table to {}".format(q_path))
    print("Saved RL report to {}".format(report_path))
    print(summary)


if __name__ == "__main__":
    main()
