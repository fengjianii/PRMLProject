import argparse
import html
import json
import os

import numpy as np
import pandas as pd

from backtest import run_from_predictions, summarize_period_backtest
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from log import log
from tradingcalendar import Calendar


GROUP_COLS = ("date", "interval")
META_COLS = ("symbol", "date", "interval")
TARGET_COL = "fret12"


FACTOR_FAMILY_RULES = [
    ("interaction", (
        "_x_",
        "spread_x_vol",
        "highlow_x_imb0",
    )),
    ("P1_cross_section", (
        "cs_rank_ret_",
        "cs_rank_rolling_",
        "cs_rank_high_low_range",
        "cs_rank_overnight_gap",
    )),
    ("P0_temporal_state", (
        "_roll_",
        "_change_",
        "rolling_vol_roll_mean",
        "ret_1_roll_skew",
    )),
    ("order_book_pressure", (
        "ob_imb",
        "buy_pressure",
        "amount_imb",
        "depth_sum",
        "depth_delta",
        "micro_price",
        "bid_ask_bias",
    )),
    ("liquidity_spread", (
        "spread",
        "relative_spread",
        "weighted_spread",
    )),
    ("trade_aggressiveness", (
        "trade_",
        "turnover",
        "buy_trade",
        "sell_trade",
    )),
    ("momentum_reversal", (
        "ret_",
        "rolling_mean_ret",
        "rolling_vol",
        "high_low_range",
        "price_position",
        "overnight_gap",
        "lagret12",
    )),
]


def infer_family(feature):
    for family, patterns in FACTOR_FAMILY_RULES:
        if any(pattern in feature for pattern in patterns):
            return family
    return "other"


def clean_numeric(series):
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def safe_corr(x, y, method="pearson"):
    if x.nunique(dropna=False) <= 1 or y.nunique(dropna=False) <= 1:
        return 0.0
    value = x.corr(y, method=method)
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)


def load_feature_frame(args):
    if args.feature_csv:
        log.inf("Loading factor frame from {}".format(args.feature_csv))
        return pd.read_csv(args.feature_csv)

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.isdir(data_dir):
        raise ValueError("Data directory does not exist: {}".format(data_dir))

    calendar = Calendar()
    dates = calendar.range(args.start_date, args.end_date)
    if not dates:
        raise ValueError("No trading dates in range {}-{}".format(
            args.start_date, args.end_date))

    raw = MeowDataLoader(data_dir).loadDates(dates)
    xdf, ydf = MeowFeatureGenerator(cacheDir=None).genFeatures(raw)
    return xdf.join(ydf).reset_index()


def choose_features(df, requested_features):
    if requested_features:
        missing = [feature for feature in requested_features if feature not in df.columns]
        if missing:
            raise ValueError("Missing requested features: {}".format(
                ", ".join(missing)))
        return requested_features

    known_features = [
        feature for feature in MeowFeatureGenerator.featureNames()
        if feature in df.columns
    ]
    if known_features:
        return known_features

    excluded = set(META_COLS + (TARGET_COL, "forecast", "action"))
    return [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def period_ic(df, feature, target_col, min_group_size):
    work = df[list(GROUP_COLS) + [feature, target_col]].copy()
    work[feature] = clean_numeric(work[feature])
    work[target_col] = clean_numeric(work[target_col])
    work["_x2"] = work[feature] * work[feature]
    work["_y2"] = work[target_col] * work[target_col]
    work["_xy"] = work[feature] * work[target_col]

    grouped = work.groupby(list(GROUP_COLS), sort=False).agg(
        n=(feature, "size"),
        sum_x=(feature, "sum"),
        sum_y=(target_col, "sum"),
        sum_x2=("_x2", "sum"),
        sum_y2=("_y2", "sum"),
        sum_xy=("_xy", "sum"),
    )
    grouped = grouped[grouped["n"] >= min_group_size]
    if len(grouped) == 0:
        return pd.Series(dtype=float)

    n = grouped["n"].astype(float)
    numerator = n * grouped["sum_xy"] - grouped["sum_x"] * grouped["sum_y"]
    denom_x = n * grouped["sum_x2"] - grouped["sum_x"] ** 2
    denom_y = n * grouped["sum_y2"] - grouped["sum_y"] ** 2
    denom = np.sqrt(denom_x.clip(lower=0) * denom_y.clip(lower=0))
    ic = numerator / denom.replace(0, np.nan)
    return ic.replace([np.inf, -np.inf], np.nan).dropna()


def decile_spread(df, feature, target_col, top_frac, bottom_frac):
    x = clean_numeric(df[feature])
    y = clean_numeric(df[target_col])
    ranks = x.groupby([df[col] for col in GROUP_COLS]).rank(
        method="first", pct=True)
    top = y[ranks > 1.0 - top_frac]
    bottom = y[ranks <= bottom_frac]
    top_mean = float(top.mean()) if len(top) else 0.0
    bottom_mean = float(bottom.mean()) if len(bottom) else 0.0
    return top_mean, bottom_mean, top_mean - bottom_mean


def mine_factors(
        df,
        features,
        target_col=TARGET_COL,
        top_frac=0.10,
        bottom_frac=0.10,
        min_group_size=20):
    if target_col not in df.columns:
        raise ValueError("Missing target column: {}".format(target_col))

    target = clean_numeric(df[target_col])
    rows = []
    for i, feature in enumerate(features, 1):
        log.inf("Mining factor [{}/{}]: {}".format(i, len(features), feature))
        values = clean_numeric(df[feature])
        ic = period_ic(df, feature, target_col, min_group_size=min_group_size)
        top_mean, bottom_mean, spread = decile_spread(
            df,
            feature,
            target_col,
            top_frac=top_frac,
            bottom_frac=bottom_frac,
        )
        ic_std = float(ic.std()) if len(ic) > 1 else 0.0
        ic_mean = float(ic.mean()) if len(ic) else 0.0
        rows.append({
            "feature": feature,
            "family": infer_family(feature),
            "samples": int(len(df)),
            "coverage": float(values.notna().mean()),
            "global_pearson": safe_corr(values, target, method="pearson"),
            "global_spearman": safe_corr(values, target, method="spearman"),
            "period_count": int(len(ic)),
            "mean_ic": ic_mean,
            "std_ic": ic_std,
            "ic_ir": float(ic_mean / ic_std) if ic_std else 0.0,
            "positive_ic_rate": float((ic > 0).mean()) if len(ic) else 0.0,
            "top_mean_target": top_mean,
            "bottom_mean_target": bottom_mean,
            "top_bottom_spread": spread,
            "top_bottom_spread_bps": spread * 10000.0,
        })

    out = pd.DataFrame(rows)
    out["abs_mean_ic"] = out["mean_ic"].abs()
    out["abs_global_spearman"] = out["global_spearman"].abs()
    out["abs_top_bottom_spread_bps"] = out["top_bottom_spread_bps"].abs()
    out["ic_consistency"] = np.maximum(
        out["positive_ic_rate"], 1.0 - out["positive_ic_rate"])

    score_cols = [
        "abs_mean_ic",
        "abs_global_spearman",
        "abs_top_bottom_spread_bps",
        "ic_consistency",
    ]
    for col in score_cols:
        out["_rank_" + col] = out[col].rank(pct=True)
    out["agent_score"] = out[["_rank_" + col for col in score_cols]].mean(axis=1)
    out["direction"] = np.where(out["top_bottom_spread"] >= 0, "positive", "reverse")
    out["agent_action"] = out.apply(suggest_action, axis=1)
    out = out.drop(columns=["_rank_" + col for col in score_cols])
    return out.sort_values("agent_score", ascending=False).reset_index(drop=True)


def suggest_action(row):
    spread = abs(row["top_bottom_spread_bps"])
    mean_ic = abs(row["mean_ic"])
    consistency = row["ic_consistency"]
    if spread >= 4.0 and mean_ic >= 0.005 and consistency >= 0.58:
        if row["direction"] == "reverse":
            return "priority_reverse_signal"
        return "priority_long_signal"
    if spread >= 2.0 and consistency >= 0.55:
        return "candidate_monitor"
    if spread < 1.0 and mean_ic < 0.002:
        return "weak_or_noisy"
    return "review"


def summarize_families(metrics):
    idx = metrics.groupby("family")["agent_score"].idxmax()
    best = metrics.loc[idx, ["family", "feature"]].rename(
        columns={"feature": "best_feature"})
    summary = metrics.groupby("family").agg(
        factor_count=("feature", "size"),
        mean_agent_score=("agent_score", "mean"),
        max_agent_score=("agent_score", "max"),
        mean_abs_ic=("abs_mean_ic", "mean"),
        max_abs_spread_bps=("abs_top_bottom_spread_bps", "max"),
    ).reset_index()
    return (
        summary.merge(best, on="family", how="left")
        .sort_values("max_agent_score", ascending=False)
        .reset_index(drop=True)
    )


def build_factor_agent_config(metrics, top_n):
    candidates = metrics[metrics["agent_action"] != "weak_or_noisy"].head(top_n)
    if len(candidates) == 0:
        candidates = metrics.head(top_n)
    weight_base = candidates["agent_score"].clip(lower=0.0)
    total = float(weight_base.sum())
    if total <= 0:
        weight_base = pd.Series(1.0, index=candidates.index)
        total = float(weight_base.sum())

    factors = []
    for idx, row in candidates.iterrows():
        sign = 1.0 if row["direction"] == "positive" else -1.0
        factors.append({
            "feature": row["feature"],
            "family": row["family"],
            "direction": row["direction"],
            "sign": sign,
            "weight": float(weight_base.loc[idx] / total),
            "agent_score": float(row["agent_score"]),
            "mean_ic": float(row["mean_ic"]),
            "top_bottom_spread_bps": float(row["top_bottom_spread_bps"]),
            "agent_action": row["agent_action"],
        })
    return {
        "type": "factor_aware_agent_config",
        "score_formula": "sum(weight * sign * centered_cross_section_rank(feature))",
        "rank_centering": "centered_rank = 2 * (rank_pct - 0.5)",
        "factors": factors,
    }


def score_factor_agent(df, config, target_col=TARGET_COL):
    out = df[list(META_COLS) + [target_col]].copy()
    score = pd.Series(0.0, index=df.index)
    for factor in config["factors"]:
        feature = factor["feature"]
        values = clean_numeric(df[feature])
        ranks = values.groupby([df[col] for col in GROUP_COLS]).rank(
            method="first", pct=True)
        centered_rank = (ranks.fillna(0.5) - 0.5) * 2.0
        score = score + factor["weight"] * factor["sign"] * centered_rank
    out["agent_factor_score"] = score.to_numpy()
    return out


def run_factor_agent_diagnostic(
        df,
        config,
        target_col,
        top_frac,
        bottom_frac,
        cost_bps):
    signal_frame = score_factor_agent(df, config, target_col=target_col)
    result, row_summary = run_from_predictions(
        signal_frame,
        policy="top_bottom",
        prediction_col="agent_factor_score",
        target_col=target_col,
        top_frac=top_frac,
        bottom_frac=bottom_frac,
        cost_bps=cost_bps,
    )
    period_summary = summarize_period_backtest(
        result,
        target_col=target_col,
        group_cols=GROUP_COLS,
    )
    return row_summary, period_summary


def save_bar_svg(path, labels, values, title, y_label):
    width = 940
    height = 560
    left = 90
    right = 28
    top = 64
    bottom = 148
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
    gap = 10
    bar_w = (plot_w - gap * (len(values) - 1)) / len(values)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" '
        'viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="36" font-family="Arial, sans-serif" '
        'font-size="22" font-weight="700" fill="#111827">{}</text>'.format(
            left, html.escape(title)),
        '<line x1="{0}" y1="{1:.2f}" x2="{2}" y2="{1:.2f}" '
        'stroke="#6b7280" stroke-width="1.2"/>'.format(left, zero_y, width - right),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" '
        'stroke="#9ca3af" stroke-width="1"/>'.format(left, top, height - bottom),
        '<text x="24" y="{}" font-family="Arial, sans-serif" font-size="13" '
        'fill="#4b5563" transform="rotate(-90 24,{})">{}</text>'.format(
            top + plot_h / 2, top + plot_h / 2, html.escape(y_label)),
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * (bar_w + gap)
        y = min(y_pos(value), zero_y)
        h = abs(y_pos(value) - zero_y)
        color = "#2563eb" if value >= 0 else "#dc2626"
        svg.append(
            '<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" '
            'rx="2" fill="{}"/>'.format(x, y, bar_w, h, color))
        svg.append(
            '<text x="{:.2f}" y="{:.2f}" text-anchor="end" '
            'font-family="Arial, sans-serif" font-size="11" fill="#374151" '
            'transform="rotate(-40 {:.2f},{:.2f})">{}</text>'.format(
                x + bar_w / 2, height - 80, x + bar_w / 2, height - 80,
                html.escape(str(label))))
        svg.append(
            '<text x="{:.2f}" y="{:.2f}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="11" fill="#111827">{:.2f}</text>'.format(
                x + bar_w / 2, y - 8 if value >= 0 else y + h + 16, value))
    svg.append(
        '<text x="{}" y="{}" font-family="Arial, sans-serif" '
        'font-size="12" fill="#6b7280">Unit: bps, ranked by Agent factor score</text>'.format(
            left, height - 18))
    svg.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def write_report(path, metrics, family_summary, factor_agent_summary, args):
    top = metrics.head(args.top_k)
    priority = metrics[metrics["agent_action"].str.startswith("priority")].head(args.top_k)
    lines = [
        "# Agent Factor Mining Report",
        "",
        "## Scope",
        "",
        "- target: `{}`".format(args.target_col),
        "- group: `date + interval`",
        "- top/bottom fraction: {:.2f}/{:.2f}".format(
            args.top_frac, args.bottom_frac),
    ]
    if args.feature_csv:
        lines.append("- source: `{}`".format(args.feature_csv))
    else:
        lines.append("- source: h5 data `{}` from {} to {}".format(
            args.data_dir or "data", args.start_date, args.end_date))

    lines.extend([
        "",
        "## Top Factors",
        "",
        "| rank | feature | family | direction | mean_ic | ICIR | spread_bps | action |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ])
    for i, row in top.iterrows():
        lines.append(
            "| {} | `{}` | {} | {} | {:.6f} | {:.4f} | {:.2f} | {} |".format(
                i + 1,
                row["feature"],
                row["family"],
                row["direction"],
                row["mean_ic"],
                row["ic_ir"],
                row["top_bottom_spread_bps"],
                row["agent_action"],
            )
        )

    lines.extend([
        "",
        "## Family Summary",
        "",
        "| family | count | best_feature | max_score | max_abs_spread_bps | mean_abs_ic |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ])
    for _, row in family_summary.iterrows():
        lines.append(
            "| {} | {} | `{}` | {:.4f} | {:.2f} | {:.6f} |".format(
                row["family"],
                int(row["factor_count"]),
                row["best_feature"],
                row["max_agent_score"],
                row["max_abs_spread_bps"],
                row["mean_abs_ic"],
            )
        )

    lines.extend([
        "",
        "## Agent Suggestions",
        "",
    ])
    if len(priority) == 0:
        lines.append("- No factor crossed the priority threshold. Use the top-score list for manual review.")
    else:
        for _, row in priority.iterrows():
            if row["direction"] == "reverse":
                direction_text = "high values predict lower future return; use as a short/risk signal"
            else:
                direction_text = "high values predict higher future return; use as a long signal"
            lines.append(
                "- `{}`: {}. Spread is {:.2f} bps, mean IC is {:.6f}.".format(
                    row["feature"],
                    direction_text,
                    row["top_bottom_spread_bps"],
                    row["mean_ic"],
                )
            )

    if factor_agent_summary:
        row = factor_agent_summary["row"]
        period = factor_agent_summary["period"]
        lines.extend([
            "",
            "## Factor-Aware Agent Diagnostic",
            "",
            "The script also builds an in-sample factor-aware Agent score from the mined factors.",
            "This is a practical hypothesis prototype, not final out-of-sample proof.",
            "",
            "```text",
            "factor_agent_top_n = {}".format(args.factor_agent_top_n),
            "long_short_spread = {:.8f} ({:.2f} bps)".format(
                row["long_short_spread"], row["long_short_spread"] * 10000.0),
            "mean_net_return = {:.8f}".format(row["mean_net_return"]),
            "trade_mean_net_return = {:.8f}".format(row["trade_mean_net_return"]),
            "period_long_short_spread = {:.8f} ({:.2f} bps)".format(
                period["period_long_short_spread"],
                period["period_long_short_spread"] * 10000.0),
            "period_positive_spread_rate = {:.4f}".format(
                period["period_positive_spread_rate"]),
            "```",
        ])

    lines.extend([
        "",
        "## How To Use This",
        "",
        "- Treat this as an automatic hypothesis generator, not final proof.",
        "- Use priority factors to explain B's forecast and design follow-up ablations.",
        "- For deployment-style discussion, combine factor score with `forecast` as a risk filter before action selection.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic Agent-assisted factor mining for MEOW features.")
    parser.add_argument("--feature-csv", default=None,
                        help="Optional CSV with feature columns and target.")
    parser.add_argument("--data-dir", default=None,
                        help="H5 data directory. Defaults to project data/.")
    parser.add_argument("--start-date", type=int, default=20231201)
    parser.add_argument("--end-date", type=int, default=20231229)
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--features", nargs="*", default=None,
                        help="Optional subset of feature names.")
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--bottom-frac", type=float, default=0.10)
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--factor-agent-top-n", type=int, default=8,
                        help="Number of mined factors in factor-aware Agent config.")
    parser.add_argument("--cost-bps", type=float, default=1.0,
                        help="Cost for factor-aware Agent diagnostic backtest.")
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_feature_frame(args)
    features = choose_features(df, args.features)
    log.inf("Running Agent factor mining on {} rows and {} factors".format(
        len(df), len(features)))

    metrics = mine_factors(
        df,
        features,
        target_col=args.target_col,
        top_frac=args.top_frac,
        bottom_frac=args.bottom_frac,
        min_group_size=args.min_group_size,
    )
    family_summary = summarize_families(metrics)
    config = build_factor_agent_config(metrics, top_n=args.factor_agent_top_n)
    row_summary, period_summary = run_factor_agent_diagnostic(
        df,
        config,
        target_col=args.target_col,
        top_frac=args.top_frac,
        bottom_frac=args.bottom_frac,
        cost_bps=args.cost_bps,
    )
    factor_agent_summary = {
        "row": row_summary,
        "period": period_summary,
    }

    metrics_path = os.path.join(args.output_dir, "C_factor_mining_summary.csv")
    family_path = os.path.join(args.output_dir, "C_factor_family_summary.csv")
    report_path = os.path.join(args.output_dir, "C_factor_mining_report.md")
    svg_path = os.path.join(args.output_dir, "C_top_factor_spread.svg")
    config_path = os.path.join(args.output_dir, "C_factor_agent_config.json")
    factor_agent_path = os.path.join(args.output_dir, "C_factor_agent_summary.csv")

    metrics.to_csv(metrics_path, index=False)
    family_summary.to_csv(family_path, index=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    pd.DataFrame([{**row_summary, **period_summary}]).to_csv(
        factor_agent_path, index=False)
    write_report(report_path, metrics, family_summary, factor_agent_summary, args)
    top = metrics.head(min(args.top_k, 12))
    save_bar_svg(
        svg_path,
        labels=top["feature"].tolist(),
        values=top["top_bottom_spread_bps"].tolist(),
        title="Agent-Mined Factor Top/Bottom Spread",
        y_label="Top-bottom spread (bps)",
    )

    log.inf("Saved factor mining summary to {}".format(metrics_path))
    log.inf("Saved family summary to {}".format(family_path))
    log.inf("Saved factor mining report to {}".format(report_path))
    log.inf("Saved top factor SVG to {}".format(svg_path))
    log.inf("Saved factor-aware Agent config to {}".format(config_path))
    log.inf("Saved factor-aware Agent diagnostic to {}".format(factor_agent_path))
    best = metrics.iloc[0]
    log.inf("Best factor: {} ({}, spread={:.2f} bps, mean_ic={:.6f})".format(
        best["feature"],
        best["agent_action"],
        best["top_bottom_spread_bps"],
        best["mean_ic"],
    ))


if __name__ == "__main__":
    main()
