import os
import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    @classmethod
    def featureNames(cls):
        """Return all feature names, organized by category.

        Categories:
          1. Price Momentum/Reversal (12 features)
          2. Order Book Pressure (16 features)
          3. Trade Aggressiveness (15 features)
          4. Interaction (5 features)
        Total: 48 features + 6 original = 54 features
        """
        return [
            # === Original baseline (6) ===
            "ob_imb0",
            "ob_imb4",
            "ob_imb9",
            "trade_imb",
            "trade_imbema5",
            "lagret12",

            # === Category 1: Price Momentum/Reversal (12) ===
            # Historical returns at different horizons
            "ret_1",
            "ret_3",
            "ret_6",
            "ret_12",
            "ret_24",
            # Rolling statistics of returns
            "rolling_mean_ret_6",
            "rolling_mean_ret_12",
            "rolling_vol_6",
            "rolling_vol_12",
            # Price range features
            "high_low_range",
            "price_position",
            "overnight_gap",

            # === Category 2: Order Book Pressure (16) ===
            # Spread features
            "spread",
            "relative_spread",
            "weighted_spread_4",
            # Amount imbalance (monetary value of orders)
            "amount_imb_4",
            "amount_imb_9",
            "amount_imb_19",
            # Depth features
            "depth_sum_4",
            "depth_sum_9",
            "depth_delta_4",
            "depth_delta_9",
            # Buy pressure (buy side proportion of total)
            "buy_pressure_0",
            "buy_pressure_4",
            "buy_pressure_9",
            # Micro price and weighted mid
            "micro_price_dev",
            "bid_ask_bias_0",
            "bid_ask_bias_4",

            # === Category 3: Trade Aggressiveness (15) ===
            # Trade intensity (volume normalized by depth)
            "trade_buy_intensity",
            "trade_sell_intensity",
            "trade_net_intensity",
            # Trade turnover ratio
            "trade_buy_turnover_ratio",
            "trade_sell_turnover_ratio",
            "trade_net_turnover_ratio",
            # Rolling trade quantities
            "rolling_trade_buy_qty_6",
            "rolling_trade_sell_qty_6",
            "rolling_trade_buy_qty_12",
            "rolling_trade_sell_qty_12",
            # Trade count imbalance
            "trade_count_imb",
            "trade_count_imbema5",
            # Average trade size
            "buy_trade_size",
            "sell_trade_size",
            "trade_size_ratio",

            # === Category 4: Interaction (5) ===
            "ret_3_x_imb0",
            "ret_6_x_imb0",
            "ret_3_x_buy_intensity",
            "spread_x_vol",
            "highlow_x_imb0",
        ]

    def __init__(self, cacheDir):
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _ratio(a, b):
        """Compute a / b safely, returning 0 where b is near-zero or NaN."""
        result = a / b.replace(0, np.nan)
        return result.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def _per_symbol(df, col, fn):
        """Apply a function per-symbol group."""
        return df.groupby("symbol")[col].transform(fn)

    @staticmethod
    def _per_session(df, col, fn):
        """Apply a function per (symbol, date) session."""
        return df.groupby("_sid")[col].transform(fn)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def genFeatures(self, df):
        n_features = len(self.featureNames())
        log.inf("Generating {} features from raw data...".format(n_features))

        # --- Sort and tag sessions ---
        df = df.sort_values(["symbol", "date", "interval"]).reset_index(drop=True)
        df["_sid"] = df["symbol"].astype(str) + "_" + df["date"].astype(str)

        # ================================================================
        # [1/5] Original Baseline Features (6)
        # ================================================================
        log.inf("  [1/5] Original baseline features...")

        df["ob_imb0"] = self._ratio(df["asize0"] - df["bsize0"],
                                    df["asize0"] + df["bsize0"])
        df["ob_imb4"] = self._ratio(df["asize0_4"] - df["bsize0_4"],
                                    df["asize0_4"] + df["bsize0_4"])
        df["ob_imb9"] = self._ratio(df["asize5_9"] - df["bsize5_9"],
                                    df["asize5_9"] + df["bsize5_9"])
        df["trade_imb"] = self._ratio(df["tradeBuyQty"] - df["tradeSellQty"],
                                      df["tradeBuyQty"] + df["tradeSellQty"])
        df["trade_imbema5"] = self._per_symbol(
            df, "trade_imb", lambda x: x.ewm(halflife=5, min_periods=1).mean())

        # lagret12: 12-period backward return, cross-sectionally demeaned
        df["bret12"] = self._per_session(
            df, "midpx", lambda x: (x - x.shift(12)) / x.shift(12).replace(0, np.nan))
        cx_bret = (df.groupby("interval")["bret12"].mean()
                   .reset_index().rename(columns={"bret12": "cx_bret12"}))
        df = df.merge(cx_bret, on="interval", how="left")
        df["lagret12"] = df["bret12"].fillna(0) - df["cx_bret12"].fillna(0)

        # ================================================================
        # [2/5] Price Momentum / Reversal Features (12)
        # ================================================================
        log.inf("  [2/5] Price momentum/reversal features...")

        # Historical returns at multiple horizons (within session)
        for lag in (1, 3, 6, 12, 24):
            df[f"ret_{lag}"] = self._per_session(
                df, "midpx",
                lambda x, l=lag: (x - x.shift(l)) / x.shift(l).replace(0, np.nan)
            )

        # Rolling mean / volatility of 1-min returns (within session)
        df["rolling_mean_ret_6"] = self._per_session(
            df, "ret_1", lambda x: x.rolling(6, min_periods=2).mean())
        df["rolling_mean_ret_12"] = self._per_session(
            df, "ret_1", lambda x: x.rolling(12, min_periods=3).mean())
        df["rolling_vol_6"] = self._per_session(
            df, "ret_1", lambda x: x.rolling(6, min_periods=3).std())
        df["rolling_vol_12"] = self._per_session(
            df, "ret_1", lambda x: x.rolling(12, min_periods=5).std())

        # Intra-interval price range
        df["high_low_range"] = self._ratio(df["high"] - df["low"], df["midpx"])
        df["price_position"] = self._ratio(df["lastpx"] - df["low"],
                                           df["high"] - df["low"])

        # Overnight gap (only first interval of each session is meaningful)
        df["_prev_close"] = df.groupby("_sid")["lastpx"].shift(1)
        df["overnight_gap"] = self._ratio(df["open"] - df["_prev_close"],
                                          df["_prev_close"].abs())
        df["_first_tick"] = df.groupby("_sid").cumcount() == 0
        df.loc[~df["_first_tick"], "overnight_gap"] = 0.0

        # ================================================================
        # [3/5] Order Book Pressure Features (16)
        # ================================================================
        log.inf("  [3/5] Order book pressure features...")

        # Spread
        df["spread"] = df["ask0"] - df["bid0"]
        df["relative_spread"] = self._ratio(df["ask0"] - df["bid0"], df["midpx"])

        # Weighted spread (vwap spread across 5 levels)
        df["_vwap_bid_4"] = self._ratio(df["btr0_4"], df["bsize0_4"])
        df["_vwap_ask_4"] = self._ratio(df["atr0_4"], df["asize0_4"])
        df["weighted_spread_4"] = self._ratio(
            df["_vwap_ask_4"] - df["_vwap_bid_4"], df["midpx"])

        # Amount imbalance (monetary value)
        df["amount_imb_4"] = self._ratio(df["atr0_4"] - df["btr0_4"],
                                         df["atr0_4"] + df["btr0_4"])
        df["amount_imb_9"] = self._ratio(df["atr5_9"] - df["btr5_9"],
                                         df["atr5_9"] + df["btr5_9"])
        df["amount_imb_19"] = self._ratio(df["atr10_19"] - df["btr10_19"],
                                          df["atr10_19"] + df["btr10_19"])

        # Total depth
        df["depth_sum_4"] = df["bsize0_4"] + df["asize0_4"]
        df["depth_sum_9"] = df["bsize5_9"] + df["asize5_9"]

        # Depth delta
        df["depth_delta_4"] = self._per_session(
            df, "depth_sum_4", lambda x: x.diff(1))
        df["depth_delta_9"] = self._per_session(
            df, "depth_sum_9", lambda x: x.diff(1))

        # Buy pressure (buy proportion)
        df["buy_pressure_0"] = self._ratio(df["bsize0"],
                                           df["bsize0"] + df["asize0"])
        df["buy_pressure_4"] = self._ratio(df["bsize0_4"],
                                           df["bsize0_4"] + df["asize0_4"])
        df["buy_pressure_9"] = self._ratio(df["bsize5_9"],
                                           df["bsize5_9"] + df["asize5_9"])

        # Micro price deviation
        df["_micro_price"] = self._ratio(
            df["bid0"] * df["asize0"] + df["ask0"] * df["bsize0"],
            df["bsize0"] + df["asize0"])
        df["micro_price_dev"] = self._ratio(
            df["_micro_price"] - df["midpx"], df["midpx"])

        # Bid-ask bias
        df["bid_ask_bias_0"] = self._ratio(
            df["bid0"] * df["bsize0"] - df["ask0"] * df["asize0"],
            df["bid0"] * df["bsize0"] + df["ask0"] * df["asize0"])
        df["bid_ask_bias_4"] = self._ratio(df["btr0_4"] - df["atr0_4"],
                                           df["btr0_4"] + df["atr0_4"])

        # ================================================================
        # [4/5] Trade Aggressiveness Features (15)
        # ================================================================
        log.inf("  [4/5] Trade aggressiveness features...")

        total_depth_4 = df["bsize0_4"] + df["asize0_4"]

        # Trade intensity
        df["trade_buy_intensity"] = self._ratio(df["tradeBuyQty"], total_depth_4)
        df["trade_sell_intensity"] = self._ratio(df["tradeSellQty"], total_depth_4)
        df["trade_net_intensity"] = self._ratio(
            df["tradeBuyQty"] - df["tradeSellQty"], total_depth_4)

        # Trade turnover ratio
        total_amt_4 = df["btr0_4"] + df["atr0_4"]
        df["trade_buy_turnover_ratio"] = self._ratio(df["tradeBuyTurnover"], total_amt_4)
        df["trade_sell_turnover_ratio"] = self._ratio(df["tradeSellTurnover"], total_amt_4)
        df["trade_net_turnover_ratio"] = self._ratio(
            df["tradeBuyTurnover"] - df["tradeSellTurnover"], total_amt_4)

        # Rolling trade quantities
        for w in (6, 12):
            df[f"rolling_trade_buy_qty_{w}"] = self._per_session(
                df, "tradeBuyQty",
                lambda x, win=w: x.rolling(win, min_periods=1).sum())
            df[f"rolling_trade_sell_qty_{w}"] = self._per_session(
                df, "tradeSellQty",
                lambda x, win=w: x.rolling(win, min_periods=1).sum())

        # Trade count imbalance
        df["trade_count_imb"] = self._ratio(
            df["nTradeBuy"] - df["nTradeSell"],
            df["nTradeBuy"] + df["nTradeSell"])
        df["trade_count_imbema5"] = self._per_symbol(
            df, "trade_count_imb",
            lambda x: x.ewm(halflife=5, min_periods=1).mean())

        # Average trade size
        df["buy_trade_size"] = self._ratio(df["tradeBuyQty"], df["nTradeBuy"])
        df["sell_trade_size"] = self._ratio(df["tradeSellQty"], df["nTradeSell"])
        df["trade_size_ratio"] = self._ratio(df["buy_trade_size"],
                                             df["sell_trade_size"])

        # ================================================================
        # [5/5] Interaction Features (5)
        # ================================================================
        log.inf("  [5/5] Interaction features...")

        df["ret_3_x_imb0"] = df["ret_3"] * df["ob_imb0"]
        df["ret_6_x_imb0"] = df["ret_6"] * df["ob_imb0"]
        df["ret_3_x_buy_intensity"] = df["ret_3"] * df["trade_buy_intensity"]
        df["spread_x_vol"] = df["relative_spread"] * df["rolling_vol_12"]
        df["highlow_x_imb0"] = df["high_low_range"] * df["ob_imb0"]

        # ================================================================
        # Build output
        # ================================================================
        log.inf("Building output DataFrames...")

        # Remove temporary columns
        tmp = ["_sid", "bret12", "cx_bret12", "_prev_close", "_first_tick",
               "_micro_price", "_vwap_bid_4", "_vwap_ask_4"]
        df.drop(columns=[c for c in tmp if c in df.columns], inplace=True)

        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)

        # Final NaN/Inf clean
        xdf = xdf.replace([np.inf, -np.inf], np.nan).fillna(0)
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)

        log.inf("Feature generation complete: {} features, {} samples".format(
            len(self.featureNames()), xdf.shape[0]))
        return xdf, ydf
