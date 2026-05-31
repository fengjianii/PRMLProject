"""
Generate all figures for the project report.
Saves PNG files to report_figures/ directory.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))
from dl import MeowDataLoader
from tradingcalendar import Calendar

# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), "report_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
def load_data(h5dir, n_days=10):
    cal = Calendar()
    dates = cal.range(20230601, 20231231)[:n_days]
    loader = MeowDataLoader(h5dir=h5dir)
    return loader.loadDates(dates)


# ---------------------------------------------------------------------------
# Figure 1: fret12 distribution
# ---------------------------------------------------------------------------
def fig_fret12_distribution(df):
    y = df["fret12"].dropna()
    # clamp for better visualization
    lo, hi = y.quantile(0.001), y.quantile(0.999)
    yc = y[(y >= lo) & (y <= hi)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram
    ax = axes[0]
    ax.hist(yc * 100, bins=120, color="#4472C4", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="zero")
    ax.set_xlabel("fret12 (%)")
    ax.set_ylabel("样本数")
    ax.set_title("fret12 分布（12分钟 forward return）")
    ax.legend()

    # Q-Q compare with normal
    from scipy import stats as sp_stats
    ax = axes[1]
    sp_stats.probplot(yc, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor("#4472C4")
    ax.get_lines()[0].set_markeredgecolor("#4472C4")
    ax.get_lines()[0].set_markersize(2)
    ax.get_lines()[0].set_alpha(0.3)
    ax.get_lines()[1].set_color("red")
    ax.get_lines()[1].set_linewidth(1.2)
    ax.set_title("Q-Q Plot（vs 正态分布）")
    ax.set_xlabel("理论分位数")
    ax.set_ylabel("样本分位数")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig01_fret12_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 2: fret12 vs key market variables (scatter + bin means)
# ---------------------------------------------------------------------------
def fig_fret12_vs_variables(df):
    dfc = df.dropna(subset=["fret12"]).copy()
    lo, hi = dfc["fret12"].quantile(0.001), dfc["fret12"].quantile(0.999)
    dfc = dfc[(dfc["fret12"] >= lo) & (dfc["fret12"] <= hi)]

    # Compute variables
    dfc["imb0"] = (dfc["asize0"] - dfc["bsize0"]) / (dfc["asize0"] + dfc["bsize0"]).replace(0, np.nan)
    dfc["rel_spread"] = (dfc["ask0"] - dfc["bid0"]) / dfc["midpx"]
    dfc["buy_intensity"] = dfc["tradeBuyQty"] / (dfc["bsize0_4"] + dfc["asize0_4"]).replace(0, np.nan)
    dfc["trade_imb"] = (dfc["tradeBuyQty"] - dfc["tradeSellQty"]) / (dfc["tradeBuyQty"] + dfc["tradeSellQty"]).replace(0, np.nan)

    pairs = [
        ("imb0", "盘口不平衡 (imb0)", "买盘占比大 → 价格可能上行"),
        ("rel_spread", "相对价差 (relative_spread)", "价差大 → 流动性差，波动大"),
        ("buy_intensity", "买方成交强度", "买方越主动 → 短期价格越可能涨"),
        ("trade_imb", "成交不平衡 (trade_imb)", "主动买多于主动卖 → 看涨信号"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for idx, (col, xlabel, note) in enumerate(pairs):
        ax = axes[idx // 2][idx % 2]
        sub = dfc[[col, "fret12"]].dropna()
        if len(sub) > 5000:
            sub = sub.sample(5000, random_state=42)

        # scatter
        ax.scatter(sub[col], sub["fret12"] * 100, s=1, alpha=0.15, color="#4472C4")

        # bin means
        try:
            bins = pd.qcut(dfc[col].dropna(), 20, duplicates="drop")
            bin_means = dfc.groupby(bins).agg({col: "mean", "fret12": "mean"})
            ax.plot(bin_means[col], bin_means["fret12"] * 100, "o-",
                    color="red", markersize=3, linewidth=1.2, label="各分位均值")
        except Exception:
            pass

        ax.set_xlabel(xlabel)
        ax.set_ylabel("fret12 (%)")
        ax.set_title(f"{xlabel}\n({note})", fontsize=9)
        ax.legend(fontsize=7)

    fig.suptitle("fret12 与关键市场变量的关系", fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig02_fret12_vs_variables.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 3: Order book depth profile
# ---------------------------------------------------------------------------
def fig_orderbook_profile(df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) Buy vs Sell depth at level 0
    ax = axes[0]
    ax.scatter(df["bsize0"] / 1000, df["asize0"] / 1000, s=0.5, alpha=0.15, color="#4472C4")
    ax.plot([0, df["bsize0"].max()/1000], [0, df["asize0"].max()/1000],
            "r--", linewidth=1, label="y=x (买卖相等)")
    ax.set_xlabel("买一量 (千)")
    ax.set_ylabel("卖一量 (千)")
    ax.set_title("Level 0: 买卖盘深度对比")
    ax.legend(fontsize=7)

    # (b) Depth by level
    ax = axes[1]
    levels = ["Level 0", "Level 0-4", "Level 5-9", "Level 10-19"]
    buy_cols = ["bsize0", "bsize0_4", "bsize5_9", "bsize10_19"]
    sell_cols = ["asize0", "asize0_4", "asize5_9", "asize10_19"]
    buy_means = [df[c].mean() / 1000 for c in buy_cols]
    sell_means = [df[c].mean() / 1000 for c in sell_cols]

    x = np.arange(len(levels))
    w = 0.35
    ax.bar(x - w/2, buy_means, w, label="买盘", color="#4472C4", alpha=0.85)
    ax.bar(x + w/2, sell_means, w, label="卖盘", color="#ED7D31", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("平均数量 (千)")
    ax.set_title("各档位平均挂单量")
    ax.legend(fontsize=7)

    # (c) Spread distribution
    ax = axes[2]
    spread = (df["ask0"] - df["bid0"]) / df["midpx"] * 10000  # basis points
    spread_c = spread[(spread > 0) & (spread < spread.quantile(0.99))]
    ax.hist(spread_c, bins=80, color="#4472C4", edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.set_xlabel("相对价差 (bps, 1bp=0.01%)")
    ax.set_ylabel("样本数")
    ax.set_title("相对价差分布 (bps)")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig03_orderbook_profile.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 4: Trade activity over a trading day
# ---------------------------------------------------------------------------
def fig_intraday_trade(df):
    # Pick one day for intraday pattern
    one_day = df[df["date"] == df["date"].iloc[0]].copy()
    if len(one_day) == 0:
        one_day = df[df["date"] == df["date"].unique()[0]].copy()

    # Group by interval, aggregate across all stocks
    intraday = one_day.groupby("interval").agg(
        total_trade_qty=("tradeBuyQty", "sum"),
        total_depth=("bsize0_4", lambda x: x.sum()),
        avg_spread=("midpx", lambda x: ((one_day.loc[x.index, "ask0"] - one_day.loc[x.index, "bid0"]) / one_day.loc[x.index, "midpx"]).mean()),
        n_stocks=("symbol", "nunique"),
    ).reset_index()

    # Convert interval (ms) to minutes from market open
    t0 = intraday["interval"].min()
    intraday["minutes"] = (intraday["interval"] - t0) / 60000

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # (a) Trade quantity and depth
    ax = axes[0]
    ax.fill_between(intraday["minutes"], intraday["total_trade_qty"] / 1e6, alpha=0.5,
                     color="#4472C4", label="成交总量 (百万)")
    ax.set_ylabel("成交量 (百万)")
    ax.set_title("日内成交活跃度变化（单日，全部股票汇总）")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(intraday["minutes"], intraday["total_depth"] / 1e6, "o-",
             color="#ED7D31", markersize=2, linewidth=0.8, label="盘口深度")
    ax2.set_ylabel("盘口深度 (百万)", color="#ED7D31")
    ax2.legend(loc="upper right", fontsize=8)

    # (b) Average spread
    ax = axes[1]
    ax.plot(intraday["minutes"], intraday["avg_spread"] * 10000, "-",
            color="#4472C4", linewidth=0.8)
    ax.set_xlabel("开盘后分钟数")
    ax.set_ylabel("平均相对价差 (bps)")
    ax.set_title("日内价差变化（开盘/收盘价差较大 → U型曲线）")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig04_intraday_trade.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 5: Feature categories overview diagram
# ---------------------------------------------------------------------------
def fig_feature_overview():
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    categories = [
        ("价格动量/反转\n(12个特征)", 1.5, 4,
         "ret_1, ret_3, ret_6, ret_12, ret_24\n"
         "rolling_mean_ret_6/12\n"
         "rolling_vol_6/12\n"
         "high_low_range, price_position\n"
         "overnight_gap",
         "#4472C4"),
        ("盘口压力\n(16个特征)", 5, 4,
         "spread, relative_spread, weighted_spread_4\n"
         "amount_imb_4/9/19\n"
         "depth_sum_4/9, depth_delta_4/9\n"
         "buy_pressure_0/4/9\n"
         "micro_price_dev, bid_ask_bias_0/4",
         "#ED7D31"),
        ("成交主动性\n(15个特征)", 8.5, 4,
         "trade_buy/sell/net_intensity\n"
         "trade_buy/sell/net_turnover_ratio\n"
         "rolling_trade_buy/sell_qty_6/12\n"
         "trade_count_imb/imbema5\n"
         "buy/sell_trade_size, trade_size_ratio",
         "#70AD47"),
        ("交互特征\n(5个特征)", 12, 4,
         "ret_3_x_imb0, ret_6_x_imb0\n"
         "ret_3_x_buy_intensity\n"
         "spread_x_vol\n"
         "highlow_x_imb0",
         "#9B59B6"),
    ]

    # Arrow from left to right
    ax.annotate("", xy=(13.8, 2.2), xytext=(0.5, 2.2),
                arrowprops=dict(arrowstyle="->", color="gray", lw=2))

    for title, x, y, features, color in categories:
        # Box
        rect = plt.Rectangle((x-1.3, y-0.8), 2.6, 3.2, facecolor=color, alpha=0.12,
                              edgecolor=color, linewidth=1.5, linestyle="--")
        ax.add_patch(rect)
        # Title
        ax.text(x, y + 1.6, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color=color)
        # Feature list
        ax.text(x, y - 0.3, features, ha="center", va="top", fontsize=7.5,
                color="#333333", linespacing=1.4)

    # Data flow annotations
    ax.text(0.2, 5.5, "原始数据\n62列", fontsize=9, ha="center", color="#666",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#ccc"))
    ax.text(13.7, 5.5, "模型输入\n54个特征", fontsize=9, ha="center", color="#666",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="#ccc"))

    ax.set_title("特征工程总览：4大类 → 54个特征", fontsize=14, fontweight="bold", pad=20)

    path = os.path.join(OUT_DIR, "fig05_feature_overview.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    h5dir = os.path.join(os.path.dirname(__file__), "data")
    print("Loading data (10 trading days)...")
    df = load_data(h5dir, n_days=10)
    print(f"  {len(df):,} rows loaded\n")

    print("Generating figures...")
    fig_fret12_distribution(df)
    fig_fret12_vs_variables(df)
    fig_orderbook_profile(df)
    fig_intraday_trade(df)
    fig_feature_overview()

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
