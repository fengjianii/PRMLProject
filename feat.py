"""
特征生成模块 (feat.py) — Task A v2 (与 GBT-final/V5 对齐)

经消融实验验证的最优方案（Pearson 0.0714）：
  基础特征(54)：全局 shift/rolling/ewm + cs_rank by interval
  P0 滚动(14) ：sort_values + groupby(symbol) + rolling/diff
  P1 横截面(8) ：groupby(interval).rank + sort+groupby(symbol).diff

关键教训：
  - shift/rolling/ewm 不能按 (symbol,date) 分组 → 每天重置丢失跨天连续性
  - 全局操作虽有 ~5% 边界错误，但实际效果远好于每天重置
  - cs_rank 按 interval 混合不同日期 → 样本量更大，排序更稳定

feature_set 参数：
  'v4'   = 54 特征（仅基础）
  'no_p1'= 68 特征（基础 + P0）
  'no_p0'= 62 特征（基础 + P1）
  'full' = 76 特征（全部）
"""

import os
import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):

    # ------------------------------------------------------------------
    # Feature name definitions
    # ------------------------------------------------------------------
    @classmethod
    def _base_features(cls):
        """V4 原版 54 个基础特征"""
        return [
            "cs_rank_ob_imb0", "cs_rank_ob_imb4", "cs_rank_ob_imb9",
            "cs_rank_trade_imb", "cs_rank_trade_imbema5", "lagret12",
            "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
            "rolling_mean_ret_6", "rolling_mean_ret_12",
            "rolling_vol_6", "rolling_vol_12",
            "high_low_range", "price_position", "overnight_gap",
            "cs_rank_spread", "cs_rank_relative_spread", "cs_rank_weighted_spread_4",
            "cs_rank_amount_imb_4",
            "amount_imb_9", "amount_imb_19",
            "cs_rank_depth_sum_4", "depth_sum_9",
            "cs_rank_depth_delta_4", "depth_delta_9",
            "cs_rank_buy_pressure_0", "cs_rank_buy_pressure_4", "buy_pressure_9",
            "cs_rank_micro_price_dev",
            "cs_rank_bid_ask_bias_0", "bid_ask_bias_4",
            "cs_rank_trade_buy_intensity", "trade_sell_intensity",
            "cs_rank_trade_net_intensity",
            "cs_rank_trade_buy_turnover_ratio",
            "trade_sell_turnover_ratio", "trade_net_turnover_ratio",
            "rolling_trade_buy_qty_6", "rolling_trade_buy_qty_12",
            "rolling_trade_sell_qty_6", "rolling_trade_sell_qty_12",
            "cs_rank_trade_count_imb", "trade_count_imbema5",
            "cs_rank_buy_trade_size", "sell_trade_size", "trade_size_ratio",
            "ret_3_x_imb0", "ret_6_x_imb0",
            "ret_3_x_buy_intensity", "spread_x_vol", "highlow_x_imb0",
        ]

    @classmethod
    def _p0_features(cls):
        """P0 滚动统计特征（14 个）"""
        return [
            "ob_imb0_roll_mean_12", "ob_imb0_roll_std_12", "ob_imb0_roll_skew_12",
            "ob_imb4_roll_mean_12", "ob_imb4_roll_std_12",
            "trade_imb_roll_mean_12", "trade_imb_roll_std_12", "trade_imb_roll_skew_12",
            "trade_imbema5_roll_mean_12", "trade_imbema5_roll_std_12",
            "ob_imb0_change_6", "ob_imb4_change_6",
            "trade_imb_change_6", "trade_imbema5_change_6",
        ]

    @classmethod
    def _p1_features(cls):
        """P1 横截面特征（8 个）"""
        return [
            "cs_rank_ret_3", "cs_rank_ret_6",
            "cs_rank_rolling_vol_12", "cs_rank_rolling_mean_ret_12",
            "cs_rank_ret_3_change", "cs_rank_ret_6_change",
            "cs_rank_rolling_vol_12_change", "cs_rank_rolling_mean_ret_12_change",
        ]

    @classmethod
    def featureNames(cls, feature_set="full"):
        if feature_set == "v4":
            return cls._base_features()
        elif feature_set == "no_p1":
            return cls._base_features() + cls._p0_features()
        elif feature_set == "no_p0":
            return cls._base_features() + cls._p1_features()
        else:
            return cls._base_features() + cls._p0_features() + cls._p1_features()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, cacheDir, feature_set="full"):
        self.cacheDir = cacheDir
        self.feature_set = feature_set
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def genFeatures(self, df):
        fset = self.feature_set
        log.inf("Generating {} features (set={})...".format(
            len(self.featureNames(fset)), fset))

        # ================================================================
        # 1. 基础特征 — V4 原版逻辑：全局 shift/rolling/ewm
        # ================================================================
        df.loc[:, "ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"])
        df.loc[:, "ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"])
        df.loc[:, "ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"])

        df.loc[:, "trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])
        df.loc[:, "trade_imbema5"] = df["trade_imb"].ewm(halflife=5).mean()

        # lagret12 — 全局 shift
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12)
        cx = df.groupby("interval")[["bret12"]].mean().reset_index()
        cx.rename(columns={"bret12": "cx_bret12"}, inplace=True)
        df = df.merge(cx, on="interval", how="left")
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]

        # ================================================================
        # 2. 价格动量/反转 — 全局 shift + rolling
        # ================================================================
        log.inf("  Price momentum/reversal features...")
        for lag in (1, 3, 6, 12, 24):
            df.loc[:, f"ret_{lag}"] = (
                (df["midpx"] - df["midpx"].shift(lag)) / df["midpx"].shift(lag))

        df.loc[:, "rolling_mean_ret_6"] = df["ret_1"].rolling(6, min_periods=1).mean()
        df.loc[:, "rolling_mean_ret_12"] = df["ret_1"].rolling(12, min_periods=1).mean()
        df.loc[:, "rolling_vol_6"] = df["ret_1"].rolling(6, min_periods=1).std()
        df.loc[:, "rolling_vol_12"] = df["ret_1"].rolling(12, min_periods=1).std()

        df.loc[:, "high_low_range"] = (df["high"] - df["low"]) / df["midpx"]
        pr = (df["high"] - df["low"]).replace(0, 1e-10)
        df.loc[:, "price_position"] = (df["lastpx"] - df["low"]) / pr

        # overnight_gap — sort+groupby(symbol)
        dfs = df.sort_values(["symbol", "interval"])
        dfs.loc[:, "prev_close"] = dfs.groupby("symbol")["lastpx"].shift(1)
        dfs.loc[:, "overnight_gap"] = (dfs["open"] - dfs["prev_close"]) / dfs["prev_close"]
        df = dfs.sort_index()

        # ================================================================
        # 3. 盘口压力
        # ================================================================
        log.inf("  Order book pressure features...")
        df.loc[:, "spread"] = df["ask0"] - df["bid0"]
        df.loc[:, "relative_spread"] = (df["ask0"] - df["bid0"]) / df["midpx"]

        vwap_b = df["btr0_4"] / df["bsize0_4"].replace(0, 1e-10)
        vwap_a = df["atr0_4"] / df["asize0_4"].replace(0, 1e-10)
        df.loc[:, "weighted_spread_4"] = (vwap_a - vwap_b) / df["midpx"]

        df.loc[:, "amount_imb_4"] = (df["atr0_4"] - df["btr0_4"]) / (df["atr0_4"] + df["btr0_4"]).replace(0, 1e-10)
        df.loc[:, "amount_imb_9"] = df["amount_imb_4"]   # 代理
        df.loc[:, "amount_imb_19"] = df["amount_imb_4"]

        df.loc[:, "depth_sum_4"] = df["bsize0_4"] + df["asize0_4"]
        df.loc[:, "depth_sum_9"] = df["depth_sum_4"]     # 代理

        # depth_delta — sort+groupby(symbol)
        dfs = df.sort_values(["symbol", "interval"])
        dfs.loc[:, "depth_delta_4"] = dfs.groupby("symbol")["depth_sum_4"].diff()
        dfs.loc[:, "depth_delta_9"] = dfs.groupby("symbol")["depth_sum_9"].diff()
        df = dfs.sort_index()

        df.loc[:, "buy_pressure_0"] = df["bsize0"] / (df["bsize0"] + df["asize0"]).replace(0, 1e-10)
        df.loc[:, "buy_pressure_4"] = df["bsize0_4"] / (df["bsize0_4"] + df["asize0_4"]).replace(0, 1e-10)
        df.loc[:, "buy_pressure_9"] = df["buy_pressure_4"]  # 代理

        mp = (df["bid0"] * df["bsize0"] + df["ask0"] * df["asize0"]) / (df["bsize0"] + df["asize0"]).replace(0, 1e-10)
        df.loc[:, "micro_price_dev"] = (mp - df["midpx"]) / df["midpx"]

        df.loc[:, "bid_ask_bias_0"] = (df["bid0"] * df["bsize0"] - df["ask0"] * df["asize0"]) / (df["bid0"] * df["bsize0"] + df["ask0"] * df["asize0"]).replace(0, 1e-10)
        df.loc[:, "bid_ask_bias_4"] = (df["btr0_4"] - df["atr0_4"]) / (df["btr0_4"] + df["atr0_4"]).replace(0, 1e-10)

        # ================================================================
        # 4. 成交主动性
        # ================================================================
        log.inf("  Trade aggressiveness features...")

        df.loc[:, "trade_buy_intensity"] = df["tradeBuyQty"] / df["depth_sum_4"].replace(0, 1e-10)
        df.loc[:, "trade_sell_intensity"] = df["tradeSellQty"] / df["depth_sum_4"].replace(0, 1e-10)
        df.loc[:, "trade_net_intensity"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / df["depth_sum_4"].replace(0, 1e-10)

        total_amt = df["btr0_4"] + df["atr0_4"]
        df.loc[:, "trade_buy_turnover_ratio"] = df["tradeBuyTurnover"] / total_amt.replace(0, 1e-10)
        df.loc[:, "trade_sell_turnover_ratio"] = df["tradeSellTurnover"] / total_amt.replace(0, 1e-10)
        df.loc[:, "trade_net_turnover_ratio"] = (df["tradeBuyTurnover"] - df["tradeSellTurnover"]) / total_amt.replace(0, 1e-10)

        # rolling trade qty — sort+groupby(symbol)
        dfs = df.sort_values(["symbol", "interval"])
        dfs.loc[:, "rolling_trade_buy_qty_6"] = (
            dfs.groupby("symbol")["tradeBuyQty"].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True))
        dfs.loc[:, "rolling_trade_buy_qty_12"] = (
            dfs.groupby("symbol")["tradeBuyQty"].rolling(12, min_periods=1).sum().reset_index(level=0, drop=True))
        dfs.loc[:, "rolling_trade_sell_qty_6"] = (
            dfs.groupby("symbol")["tradeSellQty"].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True))
        dfs.loc[:, "rolling_trade_sell_qty_12"] = (
            dfs.groupby("symbol")["tradeSellQty"].rolling(12, min_periods=1).sum().reset_index(level=0, drop=True))
        df = dfs.sort_index()

        tc = df["nTradeBuy"] + df["nTradeSell"]
        df.loc[:, "trade_count_imb"] = (df["nTradeBuy"] - df["nTradeSell"]) / tc.replace(0, 1e-10)
        df.loc[:, "trade_count_imbema5"] = df["trade_count_imb"].ewm(halflife=5).mean()

        df.loc[:, "buy_trade_size"] = df["tradeBuyQty"] / df["nTradeBuy"].replace(0, 1e-10)
        df.loc[:, "sell_trade_size"] = df["tradeSellQty"] / df["nTradeSell"].replace(0, 1e-10)
        df.loc[:, "trade_size_ratio"] = df["buy_trade_size"] / df["sell_trade_size"].replace(0, 1e-10)

        # ================================================================
        # 5. 横截面 cs_rank（20 个）— groupby(interval)
        # ================================================================
        log.inf("  Cross-sectional rank (20 features)...")
        cs_rank_cols = [
            "ob_imb0", "ob_imb4", "ob_imb9",
            "trade_imb", "trade_imbema5",
            "spread", "relative_spread", "weighted_spread_4",
            "amount_imb_4", "depth_sum_4", "depth_delta_4",
            "buy_pressure_0", "buy_pressure_4",
            "micro_price_dev", "bid_ask_bias_0",
            "trade_buy_intensity", "trade_net_intensity",
            "trade_buy_turnover_ratio",
            "trade_count_imb", "buy_trade_size",
        ]
        for col in cs_rank_cols:
            df.loc[:, f"cs_rank_{col}"] = df.groupby("interval")[col].rank(pct=True)

        # ================================================================
        # 6. 交互特征（5 个）
        # ================================================================
        log.inf("  Interaction features...")
        df.loc[:, "ret_3_x_imb0"] = df["ret_3"] * df["cs_rank_ob_imb0"]
        df.loc[:, "ret_6_x_imb0"] = df["ret_6"] * df["cs_rank_ob_imb0"]
        df.loc[:, "ret_3_x_buy_intensity"] = df["ret_3"] * df["cs_rank_trade_buy_intensity"]
        df.loc[:, "spread_x_vol"] = df["cs_rank_relative_spread"] * df["rolling_vol_12"]
        df.loc[:, "highlow_x_imb0"] = df["high_low_range"] * df["cs_rank_ob_imb0"]

        # ================================================================
        # 7. P0 滚动统计（14 个）— sort_values + groupby(symbol)
        # ================================================================
        if fset in ("full", "no_p1"):
            log.inf("  P0 rolling statistics (14 features)...")
            dfs = df.sort_values(["symbol", "interval"])

            dfs.loc[:, "ob_imb0_roll_mean_12"] = dfs.groupby("symbol")["ob_imb0"].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
            dfs.loc[:, "ob_imb0_roll_std_12"] = dfs.groupby("symbol")["ob_imb0"].rolling(12, min_periods=1).std().reset_index(level=0, drop=True)
            dfs.loc[:, "ob_imb0_roll_skew_12"] = dfs.groupby("symbol")["ob_imb0"].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True) ** 3

            dfs.loc[:, "ob_imb4_roll_mean_12"] = dfs.groupby("symbol")["ob_imb4"].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
            dfs.loc[:, "ob_imb4_roll_std_12"] = dfs.groupby("symbol")["ob_imb4"].rolling(12, min_periods=1).std().reset_index(level=0, drop=True)

            dfs.loc[:, "trade_imb_roll_mean_12"] = dfs.groupby("symbol")["trade_imb"].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
            dfs.loc[:, "trade_imb_roll_std_12"] = dfs.groupby("symbol")["trade_imb"].rolling(12, min_periods=1).std().reset_index(level=0, drop=True)
            dfs.loc[:, "trade_imb_roll_skew_12"] = dfs.groupby("symbol")["trade_imb"].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True) ** 3

            dfs.loc[:, "trade_imbema5_roll_mean_12"] = dfs.groupby("symbol")["trade_imbema5"].rolling(12, min_periods=1).mean().reset_index(level=0, drop=True)
            dfs.loc[:, "trade_imbema5_roll_std_12"] = dfs.groupby("symbol")["trade_imbema5"].rolling(12, min_periods=1).std().reset_index(level=0, drop=True)

            dfs.loc[:, "ob_imb0_change_6"] = dfs.groupby("symbol")["ob_imb0"].diff(6)
            dfs.loc[:, "ob_imb4_change_6"] = dfs.groupby("symbol")["ob_imb4"].diff(6)
            dfs.loc[:, "trade_imb_change_6"] = dfs.groupby("symbol")["trade_imb"].diff(6)
            dfs.loc[:, "trade_imbema5_change_6"] = dfs.groupby("symbol")["trade_imbema5"].diff(6)

            df = dfs.sort_index()

        # ================================================================
        # 8. P1 横截面（8 个）— groupby(interval) + sort+groupby(symbol)
        # ================================================================
        if fset in ("full", "no_p0"):
            log.inf("  P1 cross-sectional features (8 features)...")

            df.loc[:, "cs_rank_ret_3"] = df.groupby("interval")["ret_3"].rank(pct=True)
            df.loc[:, "cs_rank_ret_6"] = df.groupby("interval")["ret_6"].rank(pct=True)
            df.loc[:, "cs_rank_rolling_vol_12"] = df.groupby("interval")["rolling_vol_12"].rank(pct=True)
            df.loc[:, "cs_rank_rolling_mean_ret_12"] = df.groupby("interval")["rolling_mean_ret_12"].rank(pct=True)

            dfs = df.sort_values(["symbol", "interval"])
            dfs.loc[:, "cs_rank_ret_3_change"] = dfs.groupby("symbol")["cs_rank_ret_3"].diff()
            dfs.loc[:, "cs_rank_ret_6_change"] = dfs.groupby("symbol")["cs_rank_ret_6"].diff()
            dfs.loc[:, "cs_rank_rolling_vol_12_change"] = dfs.groupby("symbol")["cs_rank_rolling_vol_12"].diff()
            dfs.loc[:, "cs_rank_rolling_mean_ret_12_change"] = dfs.groupby("symbol")["cs_rank_rolling_mean_ret_12"].diff()
            df = dfs.sort_index()

        # ================================================================
        # 输出
        # ================================================================
        selected = self.featureNames(fset)
        xdf = df[self.mcols + selected].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        return xdf.fillna(0), ydf.fillna(0)
