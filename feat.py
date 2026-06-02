"""
特征生成模块 (feat.py) — Task A v2

合并方案：
  - A：per-symbol / per-session 分组（修复 shift 跨天 bug）
  - B：横截面 cs_rank 标准化（20 个特征）
  - P0：滚动统计特征（14 个）
  - P1：多股票横截面特征（8 个）

总计：76 个特征（原始 6 + 价格 12 + 盘口 16 + 成交 15 + 交互 5 + P0 14 + P1 8）
"""

import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    @classmethod
    def featureNames(cls):
        return [
            # === 基础（6个：5 cs_rank + 1 原始）===
            "cs_rank_ob_imb0", "cs_rank_ob_imb4", "cs_rank_ob_imb9",
            "cs_rank_trade_imb", "cs_rank_trade_imbema5", "lagret12",

            # === 价格动量/反转（12个，无量纲保留原始值）===
            "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
            "rolling_mean_ret_6", "rolling_mean_ret_12",
            "rolling_vol_6", "rolling_vol_12",
            "high_low_range", "price_position", "overnight_gap",

            # === 盘口压力（16个：8 cs_rank + 8 原始）===
            "cs_rank_spread", "cs_rank_relative_spread",
            "cs_rank_weighted_spread_4", "cs_rank_amount_imb_4",
            "amount_imb_9", "amount_imb_19",
            "cs_rank_depth_sum_4", "depth_sum_9",
            "cs_rank_depth_delta_4", "depth_delta_9",
            "cs_rank_buy_pressure_0", "cs_rank_buy_pressure_4", "buy_pressure_9",
            "cs_rank_micro_price_dev",
            "cs_rank_bid_ask_bias_0", "bid_ask_bias_4",

            # === 成交主动性（15个：5 cs_rank + 10 原始）===
            "cs_rank_trade_buy_intensity", "trade_sell_intensity",
            "cs_rank_trade_net_intensity",
            "cs_rank_trade_buy_turnover_ratio",
            "trade_sell_turnover_ratio", "trade_net_turnover_ratio",
            "rolling_trade_buy_qty_6", "rolling_trade_sell_qty_6",
            "rolling_trade_buy_qty_12", "rolling_trade_sell_qty_12",
            "cs_rank_trade_count_imb", "trade_count_imbema5",
            "cs_rank_buy_trade_size", "sell_trade_size", "trade_size_ratio",

            # === 交互（5个，使用 cs_rank 版本）===
            "ret_3_x_imb0", "ret_6_x_imb0", "ret_3_x_buy_intensity",
            "spread_x_vol", "highlow_x_imb0",

            # === P0 滚动统计（14个）===
            "ob_imb0_roll_mean_12", "ob_imb0_roll_std_12",
            "ob_imb4_roll_mean_12",
            "trade_imb_roll_mean_12", "trade_imb_roll_std_12",
            "buy_intensity_roll_mean_12",
            "rel_spread_roll_mean_12", "depth_sum_4_roll_mean_12",
            "ret_1_roll_skew_12", "rolling_vol_roll_mean_12",
            "ob_imb0_change_6", "trade_imb_change_6",
            "buy_intensity_change_6", "depth_sum_4_change_6",

            # === P1 横截面（8个）===
            "cs_rank_ret_3", "cs_rank_ret_12",
            "cs_rank_rolling_mean_ret_12", "cs_rank_rolling_vol_12",
            "cs_rank_high_low_range", "cs_rank_overnight_gap",
            "cs_rank_ret_3_change", "cs_rank_ob_imb0_change",
        ]

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ratio(a, b):
        r = a / b.replace(0, np.nan)
        return r.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def _per_session(df, col, fn):
        return df.groupby("_sid")[col].transform(fn)

    @staticmethod
    def _per_symbol(df, col, fn):
        return df.groupby("symbol")[col].transform(fn)

    def _cs_rank(self, df, col):
        """横截面 rank：同一日期+时刻，所有股票排名 → [0, 1]"""
        return df.groupby(["date", "interval"])[col].rank(pct=True)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def genFeatures(self, df):
        log.inf("Generating {} features from raw data...".format(
            len(self.featureNames())))

        # 排序 + session 标记
        df = df.sort_values(["symbol", "date", "interval"]).reset_index(drop=True)
        df["_sid"] = df["symbol"].astype(str) + "_" + df["date"].astype(str)

        # ================================================================
        # 1. 基础特征（6个）→ 存为 _raw_* 临时列，后面统一做 cs_rank
        # ================================================================
        log.inf("  [1/7] Base features...")

        df["_raw_ob_imb0"] = self._ratio(df["asize0"] - df["bsize0"],
                                         df["asize0"] + df["bsize0"])
        df["_raw_ob_imb4"] = self._ratio(df["asize0_4"] - df["bsize0_4"],
                                         df["asize0_4"] + df["bsize0_4"])
        df["_raw_ob_imb9"] = self._ratio(df["asize5_9"] - df["bsize5_9"],
                                         df["asize5_9"] + df["bsize5_9"])
        df["_raw_trade_imb"] = self._ratio(df["tradeBuyQty"] - df["tradeSellQty"],
                                           df["tradeBuyQty"] + df["tradeSellQty"])
        df["_raw_trade_imbema5"] = self._per_symbol(
            df, "_raw_trade_imb",
            lambda x: x.ewm(halflife=5, min_periods=1).mean())

        # lagret12（保留原始值，不做 cs_rank）
        df["_bret12"] = self._per_session(
            df, "midpx",
            lambda x: (x - x.shift(12)) / x.shift(12).replace(0, np.nan))
        cx = df.groupby("interval")["_bret12"].mean().reset_index()
        cx.rename(columns={"_bret12": "_cx_bret12"}, inplace=True)
        df = df.merge(cx, on="interval", how="left")
        df["lagret12"] = df["_bret12"].fillna(0) - df["_cx_bret12"].fillna(0)

        # ================================================================
        # 2. 价格动量/反转（12个，无量纲 → 保留原始值）
        # ================================================================
        log.inf("  [2/7] Price momentum/reversal...")

        for lag in (1, 3, 6, 12, 24):
            df[f"ret_{lag}"] = self._per_session(
                df, "midpx",
                lambda x, l=lag: (x - x.shift(l)) / x.shift(l).replace(0, np.nan))

        df["rolling_mean_ret_6"] = self._per_session(
            df, "ret_1", lambda x: x.rolling(6, min_periods=2).mean())
        df["rolling_mean_ret_12"] = self._per_session(
            df, "ret_1", lambda x: x.rolling(12, min_periods=3).mean())
        df["rolling_vol_6"] = self._per_session(
            df, "ret_1", lambda x: x.rolling(6, min_periods=3).std())
        df["rolling_vol_12"] = self._per_session(
            df, "ret_1", lambda x: x.rolling(12, min_periods=5).std())

        df["high_low_range"] = self._ratio(df["high"] - df["low"], df["midpx"])
        df["price_position"] = self._ratio(df["lastpx"] - df["low"],
                                           df["high"] - df["low"])

        df["_prev_close"] = df.groupby("_sid")["lastpx"].shift(1)
        df["overnight_gap"] = self._ratio(df["open"] - df["_prev_close"],
                                          df["_prev_close"].abs())
        df["_first_tick"] = df.groupby("_sid").cumcount() == 0
        df.loc[~df["_first_tick"], "overnight_gap"] = 0.0

        # ================================================================
        # 3. 盘口压力（16个）→ 部分存 _raw_* 待 cs_rank
        # ================================================================
        log.inf("  [3/7] Order book pressure...")

        df["_raw_spread"] = df["ask0"] - df["bid0"]
        df["_raw_relative_spread"] = self._ratio(df["ask0"] - df["bid0"], df["midpx"])

        df["_vwap_bid_4"] = self._ratio(df["btr0_4"], df["bsize0_4"])
        df["_vwap_ask_4"] = self._ratio(df["atr0_4"], df["asize0_4"])
        df["_raw_weighted_spread_4"] = self._ratio(
            df["_vwap_ask_4"] - df["_vwap_bid_4"], df["midpx"])

        df["_raw_amount_imb_4"] = self._ratio(df["atr0_4"] - df["btr0_4"],
                                              df["atr0_4"] + df["btr0_4"])
        df["amount_imb_9"] = self._ratio(df["atr5_9"] - df["btr5_9"],
                                         df["atr5_9"] + df["btr5_9"])
        df["amount_imb_19"] = self._ratio(df["atr10_19"] - df["btr10_19"],
                                          df["atr10_19"] + df["btr10_19"])

        df["_raw_depth_sum_4"] = df["bsize0_4"] + df["asize0_4"]
        df["depth_sum_9"] = df["bsize5_9"] + df["asize5_9"]

        df["_raw_depth_delta_4"] = self._per_session(
            df, "_raw_depth_sum_4", lambda x: x.diff(1))
        df["depth_delta_9"] = self._per_session(
            df, "depth_sum_9", lambda x: x.diff(1))

        df["_raw_buy_pressure_0"] = self._ratio(df["bsize0"],
                                                df["bsize0"] + df["asize0"])
        df["_raw_buy_pressure_4"] = self._ratio(df["bsize0_4"],
                                                df["bsize0_4"] + df["asize0_4"])
        df["buy_pressure_9"] = self._ratio(df["bsize5_9"],
                                           df["bsize5_9"] + df["asize5_9"])

        df["_micro_price"] = self._ratio(
            df["bid0"] * df["asize0"] + df["ask0"] * df["bsize0"],
            df["bsize0"] + df["asize0"])
        df["_raw_micro_price_dev"] = self._ratio(
            df["_micro_price"] - df["midpx"], df["midpx"])

        df["_raw_bid_ask_bias_0"] = self._ratio(
            df["bid0"] * df["bsize0"] - df["ask0"] * df["asize0"],
            df["bid0"] * df["bsize0"] + df["ask0"] * df["asize0"])
        df["bid_ask_bias_4"] = self._ratio(df["btr0_4"] - df["atr0_4"],
                                           df["btr0_4"] + df["atr0_4"])

        # ================================================================
        # 4. 成交主动性（15个）→ 部分存 _raw_* 待 cs_rank
        # ================================================================
        log.inf("  [4/7] Trade aggressiveness...")

        depth4 = df["bsize0_4"] + df["asize0_4"]
        amt4 = df["btr0_4"] + df["atr0_4"]

        df["_raw_trade_buy_intensity"] = self._ratio(df["tradeBuyQty"], depth4)
        df["trade_sell_intensity"] = self._ratio(df["tradeSellQty"], depth4)
        df["_raw_trade_net_intensity"] = self._ratio(
            df["tradeBuyQty"] - df["tradeSellQty"], depth4)

        df["_raw_trade_buy_turnover_ratio"] = self._ratio(df["tradeBuyTurnover"], amt4)
        df["trade_sell_turnover_ratio"] = self._ratio(df["tradeSellTurnover"], amt4)
        df["trade_net_turnover_ratio"] = self._ratio(
            df["tradeBuyTurnover"] - df["tradeSellTurnover"], amt4)

        for w in (6, 12):
            df[f"rolling_trade_buy_qty_{w}"] = self._per_session(
                df, "tradeBuyQty",
                lambda x, win=w: x.rolling(win, min_periods=1).sum())
            df[f"rolling_trade_sell_qty_{w}"] = self._per_session(
                df, "tradeSellQty",
                lambda x, win=w: x.rolling(win, min_periods=1).sum())

        df["_raw_trade_count_imb"] = self._ratio(
            df["nTradeBuy"] - df["nTradeSell"],
            df["nTradeBuy"] + df["nTradeSell"])
        df["trade_count_imbema5"] = self._per_symbol(
            df, "_raw_trade_count_imb",
            lambda x: x.ewm(halflife=5, min_periods=1).mean())

        df["_raw_buy_trade_size"] = self._ratio(df["tradeBuyQty"], df["nTradeBuy"])
        df["sell_trade_size"] = self._ratio(df["tradeSellQty"], df["nTradeSell"])
        df["trade_size_ratio"] = self._ratio(df["_raw_buy_trade_size"],
                                             df["sell_trade_size"])

        # ================================================================
        # 5. 横截面 cs_rank 标准化（20个特征）
        # ================================================================
        log.inf("  [5/7] Cross-sectional rank (20 features)...")

        rank_pairs = [
            ("_raw_ob_imb0", "cs_rank_ob_imb0"),
            ("_raw_ob_imb4", "cs_rank_ob_imb4"),
            ("_raw_ob_imb9", "cs_rank_ob_imb9"),
            ("_raw_trade_imb", "cs_rank_trade_imb"),
            ("_raw_trade_imbema5", "cs_rank_trade_imbema5"),
            ("_raw_spread", "cs_rank_spread"),
            ("_raw_relative_spread", "cs_rank_relative_spread"),
            ("_raw_weighted_spread_4", "cs_rank_weighted_spread_4"),
            ("_raw_amount_imb_4", "cs_rank_amount_imb_4"),
            ("_raw_depth_sum_4", "cs_rank_depth_sum_4"),
            ("_raw_depth_delta_4", "cs_rank_depth_delta_4"),
            ("_raw_buy_pressure_0", "cs_rank_buy_pressure_0"),
            ("_raw_buy_pressure_4", "cs_rank_buy_pressure_4"),
            ("_raw_micro_price_dev", "cs_rank_micro_price_dev"),
            ("_raw_bid_ask_bias_0", "cs_rank_bid_ask_bias_0"),
            ("_raw_trade_buy_intensity", "cs_rank_trade_buy_intensity"),
            ("_raw_trade_net_intensity", "cs_rank_trade_net_intensity"),
            ("_raw_trade_buy_turnover_ratio", "cs_rank_trade_buy_turnover_ratio"),
            ("_raw_trade_count_imb", "cs_rank_trade_count_imb"),
            ("_raw_buy_trade_size", "cs_rank_buy_trade_size"),
        ]
        for raw_col, out_col in rank_pairs:
            df[out_col] = self._cs_rank(df, raw_col)

        # ================================================================
        # 6. 交互特征（5个）+ P0 滚动（14个）+ P1 截面（8个）
        # ================================================================
        log.inf("  [6/7] Interaction + P0 rolling + P1 cross-sectional...")

        # --- 交互 ---
        df["ret_3_x_imb0"] = df["ret_3"] * df["cs_rank_ob_imb0"]
        df["ret_6_x_imb0"] = df["ret_6"] * df["cs_rank_ob_imb0"]
        df["ret_3_x_buy_intensity"] = df["ret_3"] * df["cs_rank_trade_buy_intensity"]
        df["spread_x_vol"] = df["cs_rank_relative_spread"] * df["rolling_vol_12"]
        df["highlow_x_imb0"] = df["high_low_range"] * df["cs_rank_ob_imb0"]

        # --- P0: 滚动统计 ---
        df["ob_imb0_roll_mean_12"] = self._per_session(
            df, "cs_rank_ob_imb0",
            lambda x: x.rolling(12, min_periods=3).mean())
        df["ob_imb0_roll_std_12"] = self._per_session(
            df, "cs_rank_ob_imb0",
            lambda x: x.rolling(12, min_periods=5).std())
        df["ob_imb4_roll_mean_12"] = self._per_session(
            df, "cs_rank_ob_imb4",
            lambda x: x.rolling(12, min_periods=3).mean())
        df["trade_imb_roll_mean_12"] = self._per_session(
            df, "cs_rank_trade_imb",
            lambda x: x.rolling(12, min_periods=3).mean())
        df["trade_imb_roll_std_12"] = self._per_session(
            df, "cs_rank_trade_imb",
            lambda x: x.rolling(12, min_periods=5).std())
        df["buy_intensity_roll_mean_12"] = self._per_session(
            df, "cs_rank_trade_buy_intensity",
            lambda x: x.rolling(12, min_periods=3).mean())
        df["rel_spread_roll_mean_12"] = self._per_session(
            df, "cs_rank_relative_spread",
            lambda x: x.rolling(12, min_periods=3).mean())
        df["depth_sum_4_roll_mean_12"] = self._per_session(
            df, "cs_rank_depth_sum_4",
            lambda x: x.rolling(12, min_periods=3).mean())
        df["ret_1_roll_skew_12"] = self._per_session(
            df, "ret_1",
            lambda x: x.rolling(12, min_periods=8).skew())
        df["rolling_vol_roll_mean_12"] = self._per_session(
            df, "rolling_vol_12",
            lambda x: x.rolling(12, min_periods=3).mean())

        # 变化率（加速度）
        df["ob_imb0_change_6"] = self._per_session(
            df, "cs_rank_ob_imb0", lambda x: x - x.shift(6))
        df["trade_imb_change_6"] = self._per_session(
            df, "cs_rank_trade_imb", lambda x: x - x.shift(6))
        df["buy_intensity_change_6"] = self._per_session(
            df, "cs_rank_trade_buy_intensity", lambda x: x - x.shift(6))
        df["depth_sum_4_change_6"] = self._per_session(
            df, "cs_rank_depth_sum_4", lambda x: x - x.shift(6))

        # --- P1: 横截面 ---
        df["cs_rank_ret_3"] = self._cs_rank(df, "ret_3")
        df["cs_rank_ret_12"] = self._cs_rank(df, "ret_12")
        df["cs_rank_rolling_mean_ret_12"] = self._cs_rank(df, "rolling_mean_ret_12")
        df["cs_rank_rolling_vol_12"] = self._cs_rank(df, "rolling_vol_12")
        df["cs_rank_high_low_range"] = self._cs_rank(df, "high_low_range")
        df["cs_rank_overnight_gap"] = self._cs_rank(df, "overnight_gap")
        df["cs_rank_ret_3_change"] = self._per_session(
            df, "cs_rank_ret_3", lambda x: x - x.shift(6))
        df["cs_rank_ob_imb0_change"] = self._per_session(
            df, "cs_rank_ob_imb0", lambda x: x - x.shift(6))

        # ================================================================
        # 7. 输出
        # ================================================================
        log.inf("  [7/7] Building output...")

        tmp_cols = [c for c in df.columns
                    if c.startswith("_raw_") or c.startswith("_tmp_")
                    or c in ("_sid", "_bret12", "_cx_bret12", "_prev_close",
                             "_first_tick", "_micro_price", "_vwap_bid_4",
                             "_vwap_ask_4")]
        df.drop(columns=tmp_cols, inplace=True, errors="ignore")

        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)

        xdf = xdf.replace([np.inf, -np.inf], np.nan).fillna(0)
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)

        log.inf("Done: {} features, {} samples".format(
            len(self.featureNames()), xdf.shape[0]))
        return xdf, ydf
