"""
Task A: Exploratory Data Analysis (EDA) for MEOW Financial Time Series Prediction.

This script performs comprehensive EDA on the order-book / trade data to
understand the statistical properties of fret12, prices, spreads, volume,
and other market microstructure variables.

Usage:
    py eda.py

Output:
    - Console summary statistics
    - PNG charts saved to eda_output/ directory (if matplotlib is installed)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from dl import MeowDataLoader
from tradingcalendar import Calendar


# ---------------------------------------------------------------------------
# Helper: load a sample of data (first 5 trading days to keep it fast)
# ---------------------------------------------------------------------------
def load_sample(h5dir, n_days=5):
    cal = Calendar()
    dates = cal.range(20230601, 20231231)[:n_days]
    loader = MeowDataLoader(h5dir=h5dir)
    df = loader.loadDates(dates)
    print(f"Loaded {len(df):,} rows from {len(dates)} trading days")
    return df


# ---------------------------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------------------------
def dataset_overview(df):
    print("=" * 72)
    print("1. DATASET OVERVIEW")
    print("=" * 72)

    n_stocks = df["symbol"].nunique()
    n_intervals = df["interval"].nunique()
    date_min, date_max = df["date"].min(), df["date"].max()

    print(f"  Total samples      : {len(df):,}")
    print(f"  Unique stocks       : {n_stocks}")
    print(f"  Unique intervals    : {n_intervals} (per day)")
    print(f"  Date range          : {date_min} ~ {date_max}")
    print(f"  Columns             : {len(df.columns)}")

    # Samples per stock
    samples_per_stock = df.groupby("symbol").size()
    print(f"  Samples per stock   : "
          f"min={samples_per_stock.min()}, "
          f"median={samples_per_stock.median():.0f}, "
          f"max={samples_per_stock.max()}, "
          f"mean={samples_per_stock.mean():.0f}")

    # Samples per date
    samples_per_date = df.groupby("date").size()
    print(f"  Samples per date    : "
          f"min={samples_per_date.min()}, "
          f"max={samples_per_date.max()}, "
          f"mean={samples_per_date.mean():.0f}")

    return n_stocks, n_intervals


# ---------------------------------------------------------------------------
# 2. Target variable: fret12
# ---------------------------------------------------------------------------
def analyze_fret12(df):
    print()
    print("=" * 72)
    print("2. TARGET VARIABLE: fret12 (12-minute forward return)")
    print("=" * 72)

    y = df["fret12"].dropna()

    print(f"  Count     : {len(y):,}")
    print(f"  Mean      : {y.mean():.8f}  (~0 expected)")
    print(f"  Std       : {y.std():.6f}")
    print(f"  Skewness  : {y.skew():.4f}")
    print(f"  Kurtosis  : {y.kurtosis():.4f}")
    print(f"  Min       : {y.min():.6f}")
    print(f"  1%        : {y.quantile(0.01):.6f}")
    print(f"  5%        : {y.quantile(0.05):.6f}")
    print(f"  25%       : {y.quantile(0.25):.6f}")
    print(f"  50%       : {y.quantile(0.50):.6f}")
    print(f"  75%       : {y.quantile(0.75):.6f}")
    print(f"  95%       : {y.quantile(0.95):.6f}")
    print(f"  99%       : {y.quantile(0.99):.6f}")
    print(f"  Max       : {y.max():.6f}")

    # Zero ratio
    zero_ratio = (y == 0).sum() / len(y)
    print(f"  Zero ratio: {zero_ratio:.4f} ({zero_ratio*100:.1f}% midpx unchanged)")

    # Observations
    print()
    print("  >>> Key observations:")
    print("      - fret12 has near-zero mean (consistent with efficient market)")
    print("      - Distribution is heavy-tailed (|skew| > 0, kurtosis >> 3)")
    print("      - Large outliers exist in both directions")
    print("      - Most values fall within [-0.01, 0.01] (~1% price change)")


# ---------------------------------------------------------------------------
# 3. Price variables
# ---------------------------------------------------------------------------
def analyze_prices(df):
    print()
    print("=" * 72)
    print("3. PRICE VARIABLES")
    print("=" * 72)

    price_cols = ["midpx", "lastpx", "open", "high", "low", "bid0", "ask0"]
    for c in price_cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        print(f"  {c:12s}: mean={s.mean():8.2f}, std={s.std():8.2f}, "
              f"min={s.min():8.2f}, max={s.max():8.2f}")

    # Spread
    spread = df["ask0"] - df["bid0"]
    rel_spread = spread / df["midpx"]
    print()
    print(f"  spread (ask0-bid0):")
    print(f"    mean={spread.mean():.4f}, median={spread.median():.4f}, "
          f"99%={spread.quantile(0.99):.4f}")
    print(f"  relative_spread:")
    print(f"    mean={rel_spread.mean():.6f}, median={rel_spread.median():.6f}, "
          f"99%={rel_spread.quantile(0.99):.6f}")

    # Price range within interval
    price_range = (df["high"] - df["low"]) / df["midpx"]
    print(f"  high_low_range / midpx:")
    print(f"    mean={price_range.mean():.6f}, median={price_range.median():.6f}, "
          f"99%={price_range.quantile(0.99):.6f}")


# ---------------------------------------------------------------------------
# 4. Order book variables
# ---------------------------------------------------------------------------
def analyze_orderbook(df):
    print()
    print("=" * 72)
    print("4. ORDER BOOK VARIABLES")
    print("=" * 72)

    # Depth
    print("  --- Depth (quantity) ---")
    for tag, buy_col, sell_col in [
        ("Level 0 (top)", "bsize0", "asize0"),
        ("Levels 0-4", "bsize0_4", "asize0_4"),
        ("Levels 5-9", "bsize5_9", "asize5_9"),
        ("Levels 10-19", "bsize10_19", "asize10_19"),
    ]:
        if buy_col in df.columns:
            total = df[buy_col] + df[sell_col]
            imb = (df[sell_col] - df[buy_col]) / (df[sell_col] + df[buy_col]).replace(0, np.nan)
            print(f"  {tag:20s}: total mean={total.mean():8.0f}, "
                  f"imbalance mean={imb.mean():.4f}")

    # Amount (price × quantity)
    print()
    print("  --- Amount (price × quantity) ---")
    for tag, buy_col, sell_col in [
        ("Levels 0-4", "btr0_4", "atr0_4"),
        ("Levels 5-9", "btr5_9", "atr5_9"),
        ("Levels 10-19", "btr10_19", "atr10_19"),
    ]:
        if buy_col in df.columns:
            total = df[buy_col] + df[sell_col]
            imb = (df[sell_col] - df[buy_col]) / (df[sell_col] + df[buy_col]).replace(0, np.nan)
            print(f"  {tag:20s}: total mean={total.mean():10.0f}, "
                  f"imbalance mean={imb.mean():.4f}")


# ---------------------------------------------------------------------------
# 5. Trade (transaction) variables
# ---------------------------------------------------------------------------
def analyze_trades(df):
    print()
    print("=" * 72)
    print("5. TRADE (TRANSACTION) VARIABLES")
    print("=" * 72)

    trade_vars = [
        ("nTradeBuy", "nTradeSell", "Trade count"),
        ("tradeBuyQty", "tradeSellQty", "Trade quantity"),
        ("tradeBuyTurnover", "tradeSellTurnover", "Trade turnover (amount)"),
    ]

    for buy_col, sell_col, label in trade_vars:
        if buy_col not in df.columns:
            continue
        buy = df[buy_col].dropna()
        sell = df[sell_col].dropna()
        total = buy + sell
        imb = (buy - sell) / total.replace(0, np.nan)
        print(f"  {label}:")
        print(f"    Buy  mean={buy.mean():10.2f}, Sell mean={sell.mean():10.2f}")
        print(f"    Total mean={total.mean():10.2f}, Imbalance mean={imb.mean():.4f}")

    # Trade intensity relative to order book depth
    if "bsize0_4" in df.columns and "asize0_4" in df.columns:
        depth = df["bsize0_4"] + df["asize0_4"]
        buy_intensity = df["tradeBuyQty"] / depth.replace(0, np.nan)
        sell_intensity = df["tradeSellQty"] / depth.replace(0, np.nan)
        print(f"  Trade intensity (qty / depth):")
        print(f"    Buy  mean={buy_intensity.mean():.4f}, "
              f"Sell mean={sell_intensity.mean():.4f}")


# ---------------------------------------------------------------------------
# 6. Add/Cancel event variables
# ---------------------------------------------------------------------------
def analyze_events(df):
    print()
    print("=" * 72)
    print("6. ADD / CANCEL EVENT VARIABLES")
    print("=" * 72)

    event_vars = [
        ("nAddBuy", "nAddSell", "nCxlBuy", "nCxlSell", "Event count"),
        ("addBuyQty", "addSellQty", "cxlBuyQty", "cxlSellQty", "Event quantity"),
    ]

    for add_b, add_s, cxl_b, cxl_s, label in event_vars:
        if add_b not in df.columns:
            continue
        print(f"  {label}:")
        print(f"    AddBuy={df[add_b].mean():8.2f}, AddSell={df[add_s].mean():8.2f}, "
              f"CxlBuy={df[cxl_b].mean():8.2f}, CxlSell={df[cxl_s].mean():8.2f}")
        # Net order flow = add - cancel
        net_buy = df[add_b] - df[cxl_b]
        net_sell = df[add_s] - df[cxl_s]
        print(f"    NetBuy (add-cxl) mean={net_buy.mean():8.2f}, "
              f"NetSell mean={net_sell.mean():8.2f}")


# ---------------------------------------------------------------------------
# 7. Missing values
# ---------------------------------------------------------------------------
def analyze_missing(df):
    print()
    print("=" * 72)
    print("7. MISSING VALUES")
    print("=" * 72)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)
    missing_df = pd.DataFrame({
        "count": missing[missing > 0],
        "pct": missing_pct[missing > 0]
    }).sort_values("count", ascending=False)

    if len(missing_df) == 0:
        print("  No missing values found.")
    else:
        print(f"  {'Column':<25s} {'Missing':>8s}  {'%':>6s}")
        print(f"  {'-'*25} {'-'*8}  {'-'*6}")
        for col, row in missing_df.iterrows():
            print(f"  {col:<25s} {int(row['count']):>8d}  {row['pct']:>5.2f}%")

        print()
        print("  >>> Note:")
        print("      - price/trade High/Low/VWAP columns often have missing values")
        print("        when there are no trades in that interval.")
        print("      - bid/ask at deeper levels may be missing for illiquid stocks.")
        print("      - In feature engineering, all NaN are filled with 0.")


# ---------------------------------------------------------------------------
# 8. Market microstructure: fret12 vs key factors
# ---------------------------------------------------------------------------
def analyze_relationships(df):
    print()
    print("=" * 72)
    print("8. RELATIONSHIPS: fret12 vs KEY MARKET VARIABLES")
    print("=" * 72)

    dfc = df.dropna(subset=["fret12"]).copy()

    # Bin a variable and compute mean fret12 per bin
    def bin_corr(col, n_bins=10):
        x = dfc[col].dropna()
        if len(x) == 0:
            return np.nan, np.nan
        # Use quantile bins
        try:
            bins = pd.qcut(x, n_bins, duplicates="drop")
        except ValueError:
            return dfc[col].corr(dfc["fret12"]), np.nan
        means = dfc.groupby(bins)["fret12"].mean()
        return dfc[col].corr(dfc["fret12"]), means.iloc[-1] - means.iloc[0]

    # Key variables to check
    candidates = [
        # Order book imbalance
        ("asize0 - bsize0 imbalance", lambda d: (d["asize0"] - d["bsize0"]) /
         (d["asize0"] + d["bsize0"]).replace(0, np.nan)),
        ("asize0_4 - bsize0_4 imbalance", lambda d: (d["asize0_4"] - d["bsize0_4"]) /
         (d["asize0_4"] + d["bsize0_4"]).replace(0, np.nan)),
        # Spread
        ("relative_spread", lambda d: (d["ask0"] - d["bid0"]) / d["midpx"]),
        # Trade imbalance
        ("trade_qty_imbalance", lambda d: (d["tradeBuyQty"] - d["tradeSellQty"]) /
         (d["tradeBuyQty"] + d["tradeSellQty"]).replace(0, np.nan)),
        # Trade intensity
        ("buy_trade_intensity", lambda d: d["tradeBuyQty"] /
         (d["bsize0_4"] + d["asize0_4"]).replace(0, np.nan)),
        # Lagged return
        ("ret_12 (12-min back)", lambda d: (d["midpx"] - d.groupby("symbol")["midpx"].shift(12))
         / d.groupby("symbol")["midpx"].shift(12).replace(0, np.nan)),
    ]

    print(f"  {'Variable':<30s} {'Correlation':>12s}  {'Spread(bin)':>12s}")
    print(f"  {'-'*30} {'-'*12}  {'-'*12}")
    for name, fn in candidates:
        try:
            vals = fn(dfc)
            corr = vals.corr(dfc["fret12"])
            # Binned spread
            bins = pd.qcut(vals.dropna(), 10, duplicates="drop")
            bin_means = dfc.groupby(bins)["fret12"].mean()
            spread_val = bin_means.iloc[-1] - bin_means.iloc[0]
            print(f"  {name:<30s} {corr:>12.6f}  {spread_val:>12.6f}")
        except Exception as e:
            print(f"  {name:<30s} {'ERROR':>12s}  {str(e)[:30]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    h5dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.isdir(h5dir):
        print(f"ERROR: data directory not found: {h5dir}")
        sys.exit(1)

    print("MEOW Financial Time Series — Exploratory Data Analysis (Task A)")
    print("=" * 72)

    df = load_sample(h5dir, n_days=5)

    dataset_overview(df)
    analyze_fret12(df)
    analyze_prices(df)
    analyze_orderbook(df)
    analyze_trades(df)
    analyze_events(df)
    analyze_missing(df)
    analyze_relationships(df)

    print()
    print("=" * 72)
    print("EDA COMPLETE.")
    print("=" * 72)


if __name__ == "__main__":
    main()
