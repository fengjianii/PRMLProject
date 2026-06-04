import os
import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    
    @classmethod
    def _base_features(cls):
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
            "cs_rank_buy_pressure_0", "cs_rank_buy_pressure_4",
            "buy_pressure_9",
            "cs_rank_micro_price_dev", "cs_rank_bid_ask_bias_0",
            "bid_ask_bias_4",
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
        return [
            "cs_rank_ret_3", "cs_rank_ret_6",
            "cs_rank_rolling_vol_12", "cs_rank_rolling_mean_ret_12",
            "cs_rank_ret_3_change", "cs_rank_ret_6_change",
            "cs_rank_rolling_vol_12_change", "cs_rank_rolling_mean_ret_12_change",
        ]
    
    @classmethod
    def featureNames(cls, feature_set='full'):
        if feature_set == 'v4':
            return cls._base_features()
        elif feature_set == 'no_p1':
            return cls._base_features() + cls._p0_features()
        elif feature_set == 'no_p0':
            return cls._base_features() + cls._p1_features()
        else:
            return cls._base_features() + cls._p0_features() + cls._p1_features()

    def __init__(self, cacheDir, feature_set='full'):
        self.cacheDir = cacheDir
        self.feature_set = feature_set
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]

    def genFeatures(self, df):
        log.inf("Generating {} features (set={})...".format(
            len(self.featureNames(self.feature_set)), self.feature_set))
        
        df.loc[:, "ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"])
        df.loc[:, "ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"])
        df.loc[:, "ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"])
        
        df.loc[:, "trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])
        df.loc[:, "trade_imbema5"] = df["trade_imb"].ewm(halflife=5).mean()
        
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12)
        cxbret = df.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        df = df.merge(cxbret, on="interval", how="left")
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]
        
        log.inf("Generating price momentum/reversal features...")
        df.loc[:, "ret_1"] = (df["midpx"] - df["midpx"].shift(1)) / df["midpx"].shift(1)
        df.loc[:, "ret_3"] = (df["midpx"] - df["midpx"].shift(3)) / df["midpx"].shift(3)
        df.loc[:, "ret_6"] = (df["midpx"] - df["midpx"].shift(6)) / df["midpx"].shift(6)
        df.loc[:, "ret_12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12)
        df.loc[:, "ret_24"] = (df["midpx"] - df["midpx"].shift(24)) / df["midpx"].shift(24)
        
        df.loc[:, "rolling_mean_ret_6"] = df["ret_1"].rolling(window=6, min_periods=1).mean()
        df.loc[:, "rolling_mean_ret_12"] = df["ret_1"].rolling(window=12, min_periods=1).mean()
        df.loc[:, "rolling_vol_6"] = df["ret_1"].rolling(window=6, min_periods=1).std()
        df.loc[:, "rolling_vol_12"] = df["ret_1"].rolling(window=12, min_periods=1).std()
        
        df.loc[:, "high_low_range"] = (df["high"] - df["low"]) / df["midpx"]
        price_range = df["high"] - df["low"]
        price_range = price_range.replace(0, 1e-10)
        df.loc[:, "price_position"] = (df["lastpx"] - df["low"]) / price_range
        
        df_sorted = df.sort_values(["symbol", "interval"])
        df_sorted.loc[:, "prev_close"] = df_sorted.groupby("symbol")["lastpx"].shift(1)
        df_sorted.loc[:, "overnight_gap"] = (df_sorted["open"] - df_sorted["prev_close"]) / df_sorted["prev_close"]
        df = df_sorted.sort_index()
        
        log.inf("Generating order book pressure features...")
        df.loc[:, "spread"] = df["ask0"] - df["bid0"]
        df.loc[:, "relative_spread"] = (df["ask0"] - df["bid0"]) / df["midpx"]
        
        vwap_bid_4 = df["btr0_4"] / df["bsize0_4"].replace(0, 1e-10)
        vwap_ask_4 = df["atr0_4"] / df["asize0_4"].replace(0, 1e-10)
        df.loc[:, "weighted_spread_4"] = (vwap_ask_4 - vwap_bid_4) / df["midpx"]
        
        df.loc[:, "amount_imb_4"] = (df["atr0_4"] - df["btr0_4"]) / (df["atr0_4"] + df["btr0_4"]).replace(0, 1e-10)
        df.loc[:, "amount_imb_9"] = df["amount_imb_4"]
        df.loc[:, "amount_imb_19"] = df["amount_imb_4"]
        
        df.loc[:, "depth_sum_4"] = df["bsize0_4"] + df["asize0_4"]
        df.loc[:, "depth_sum_9"] = df["depth_sum_4"]
        
        df_sorted = df.sort_values(["symbol", "interval"])
        df_sorted.loc[:, "depth_delta_4"] = df_sorted.groupby("symbol")["depth_sum_4"].diff()
        df_sorted.loc[:, "depth_delta_9"] = df_sorted.groupby("symbol")["depth_sum_9"].diff()
        df = df_sorted.sort_index()
        
        df.loc[:, "buy_pressure_0"] = df["bsize0"] / (df["bsize0"] + df["asize0"]).replace(0, 1e-10)
        df.loc[:, "buy_pressure_4"] = df["bsize0_4"] / (df["bsize0_4"] + df["asize0_4"]).replace(0, 1e-10)
        df.loc[:, "buy_pressure_9"] = df["buy_pressure_4"]
        
        micro_price = (df["bid0"] * df["bsize0"] + df["ask0"] * df["asize0"]) / (df["bsize0"] + df["asize0"]).replace(0, 1e-10)
        df.loc[:, "micro_price_dev"] = (micro_price - df["midpx"]) / df["midpx"]
        df.loc[:, "bid_ask_bias_0"] = (df["bid0"] * df["bsize0"] - df["ask0"] * df["asize0"]) / (df["bid0"] * df["bsize0"] + df["ask0"] * df["asize0"]).replace(0, 1e-10)
        df.loc[:, "bid_ask_bias_4"] = (df["btr0_4"] - df["atr0_4"]) / (df["btr0_4"] + df["atr0_4"]).replace(0, 1e-10)
        
        log.inf("Generating trade activity features...")
        df.loc[:, "trade_buy_intensity"] = df["tradeBuyQty"] / df["depth_sum_4"].replace(0, 1e-10)
        
        has_trade_sell = "tradeSellQty" in df.columns
        has_nTradeSell = "nTradeSell" in df.columns
        has_tradeSellTurnover = "tradeSellTurnover" in df.columns
        
        if has_trade_sell:
            df.loc[:, "trade_sell_intensity"] = df["tradeSellQty"] / df["depth_sum_4"].replace(0, 1e-10)
            df.loc[:, "trade_net_intensity"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / df["depth_sum_4"].replace(0, 1e-10)
        else:
            df.loc[:, "trade_sell_intensity"] = 0
            df.loc[:, "trade_net_intensity"] = df["trade_buy_intensity"]
        
        total_turnover = df["btr0_4"] + df["atr0_4"]
        df.loc[:, "trade_buy_turnover_ratio"] = df["tradeBuyTurnover"] / total_turnover.replace(0, 1e-10)
        
        if has_tradeSellTurnover:
            df.loc[:, "trade_sell_turnover_ratio"] = df["tradeSellTurnover"] / total_turnover.replace(0, 1e-10)
            df.loc[:, "trade_net_turnover_ratio"] = (df["tradeBuyTurnover"] - df["tradeSellTurnover"]) / total_turnover.replace(0, 1e-10)
        else:
            df.loc[:, "trade_sell_turnover_ratio"] = 0
            df.loc[:, "trade_net_turnover_ratio"] = df["trade_buy_turnover_ratio"]
        
        df_sorted = df.sort_values(["symbol", "interval"])
        df_sorted.loc[:, "rolling_trade_buy_qty_6"] = df_sorted.groupby("symbol")["tradeBuyQty"].rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True)
        df_sorted.loc[:, "rolling_trade_buy_qty_12"] = df_sorted.groupby("symbol")["tradeBuyQty"].rolling(window=12, min_periods=1).sum().reset_index(level=0, drop=True)
        if has_trade_sell:
            df_sorted.loc[:, "rolling_trade_sell_qty_6"] = df_sorted.groupby("symbol")["tradeSellQty"].rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True)
            df_sorted.loc[:, "rolling_trade_sell_qty_12"] = df_sorted.groupby("symbol")["tradeSellQty"].rolling(window=12, min_periods=1).sum().reset_index(level=0, drop=True)
        else:
            df_sorted.loc[:, "rolling_trade_sell_qty_6"] = 0
            df_sorted.loc[:, "rolling_trade_sell_qty_12"] = 0
        df = df_sorted.sort_index()
        
        if has_nTradeSell:
            total_trade_count = df["nTradeBuy"] + df["nTradeSell"]
            df.loc[:, "trade_count_imb"] = (df["nTradeBuy"] - df["nTradeSell"]) / total_trade_count.replace(0, 1e-10)
        else:
            df.loc[:, "trade_count_imb"] = df["nTradeBuy"] / (df["nTradeBuy"] + 1).replace(0, 1e-10)
        
        df.loc[:, "trade_count_imbema5"] = df["trade_count_imb"].ewm(halflife=5).mean()
        df.loc[:, "buy_trade_size"] = df["tradeBuyQty"] / df["nTradeBuy"].replace(0, 1e-10)
        
        if has_trade_sell and has_nTradeSell:
            df.loc[:, "sell_trade_size"] = df["tradeSellQty"] / df["nTradeSell"].replace(0, 1e-10)
            df.loc[:, "trade_size_ratio"] = df["buy_trade_size"] / df["sell_trade_size"].replace(0, 1e-10)
        else:
            df.loc[:, "sell_trade_size"] = df["buy_trade_size"]
            df.loc[:, "trade_size_ratio"] = 1.0
        
        log.inf("Generating cross-sectional rank features...")
        cs_rank_cols = [
            "ob_imb0", "ob_imb4", "ob_imb9", "trade_imb", "trade_imbema5",
            "spread", "relative_spread", "weighted_spread_4", "amount_imb_4",
            "depth_sum_4", "depth_delta_4", "buy_pressure_0", "buy_pressure_4",
            "micro_price_dev", "bid_ask_bias_0", "trade_buy_intensity", "trade_net_intensity",
            "trade_buy_turnover_ratio", "trade_count_imb", "buy_trade_size",
        ]
        for col in cs_rank_cols:
            df.loc[:, f"cs_rank_{col}"] = df.groupby("interval")[col].rank(pct=True)
        
        log.inf("Generating interaction features...")
        df.loc[:, "ret_3_x_imb0"] = df["ret_3"] * df["cs_rank_ob_imb0"]
        df.loc[:, "ret_6_x_imb0"] = df["ret_6"] * df["cs_rank_ob_imb0"]
        df.loc[:, "ret_3_x_buy_intensity"] = df["ret_3"] * df["cs_rank_trade_buy_intensity"]
        df.loc[:, "spread_x_vol"] = df["cs_rank_relative_spread"] * df["rolling_vol_12"]
        df.loc[:, "highlow_x_imb0"] = df["high_low_range"] * df["cs_rank_ob_imb0"]
        
        log.inf("Generating P0 rolling statistics features...")
        df_sorted = df.sort_values(["symbol", "interval"])
        df_sorted.loc[:, "ob_imb0_roll_mean_12"] = df_sorted.groupby("symbol")["ob_imb0"].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True)
        df_sorted.loc[:, "ob_imb0_roll_std_12"] = df_sorted.groupby("symbol")["ob_imb0"].rolling(window=12, min_periods=1).std().reset_index(level=0, drop=True)
        df_sorted.loc[:, "ob_imb0_roll_skew_12"] = df_sorted.groupby("symbol")["ob_imb0"].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True) ** 3
        df_sorted.loc[:, "ob_imb4_roll_mean_12"] = df_sorted.groupby("symbol")["ob_imb4"].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True)
        df_sorted.loc[:, "ob_imb4_roll_std_12"] = df_sorted.groupby("symbol")["ob_imb4"].rolling(window=12, min_periods=1).std().reset_index(level=0, drop=True)
        df_sorted.loc[:, "trade_imb_roll_mean_12"] = df_sorted.groupby("symbol")["trade_imb"].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True)
        df_sorted.loc[:, "trade_imb_roll_std_12"] = df_sorted.groupby("symbol")["trade_imb"].rolling(window=12, min_periods=1).std().reset_index(level=0, drop=True)
        df_sorted.loc[:, "trade_imb_roll_skew_12"] = df_sorted.groupby("symbol")["trade_imb"].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True) ** 3
        df_sorted.loc[:, "trade_imbema5_roll_mean_12"] = df_sorted.groupby("symbol")["trade_imbema5"].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True)
        df_sorted.loc[:, "trade_imbema5_roll_std_12"] = df_sorted.groupby("symbol")["trade_imbema5"].rolling(window=12, min_periods=1).std().reset_index(level=0, drop=True)
        df_sorted.loc[:, "ob_imb0_change_6"] = df_sorted.groupby("symbol")["ob_imb0"].diff(6)
        df_sorted.loc[:, "ob_imb4_change_6"] = df_sorted.groupby("symbol")["ob_imb4"].diff(6)
        df_sorted.loc[:, "trade_imb_change_6"] = df_sorted.groupby("symbol")["trade_imb"].diff(6)
        df_sorted.loc[:, "trade_imbema5_change_6"] = df_sorted.groupby("symbol")["trade_imbema5"].diff(6)
        df = df_sorted.sort_index()
        
        log.inf("Generating P1 cross-sectional features...")
        df.loc[:, "cs_rank_ret_3"] = df.groupby("interval")["ret_3"].rank(pct=True)
        df.loc[:, "cs_rank_ret_6"] = df.groupby("interval")["ret_6"].rank(pct=True)
        df.loc[:, "cs_rank_rolling_vol_12"] = df.groupby("interval")["rolling_vol_12"].rank(pct=True)
        df.loc[:, "cs_rank_rolling_mean_ret_12"] = df.groupby("interval")["rolling_mean_ret_12"].rank(pct=True)
        
        df_sorted = df.sort_values(["symbol", "interval"])
        df_sorted.loc[:, "cs_rank_ret_3_change"] = df_sorted.groupby("symbol")["cs_rank_ret_3"].diff()
        df_sorted.loc[:, "cs_rank_ret_6_change"] = df_sorted.groupby("symbol")["cs_rank_ret_6"].diff()
        df_sorted.loc[:, "cs_rank_rolling_vol_12_change"] = df_sorted.groupby("symbol")["cs_rank_rolling_vol_12"].diff()
        df_sorted.loc[:, "cs_rank_rolling_mean_ret_12_change"] = df_sorted.groupby("symbol")["cs_rank_rolling_mean_ret_12"].diff()
        df = df_sorted.sort_index()
        
        selected_features = self.featureNames(self.feature_set)
        xdf = df[self.mcols + selected_features].set_index(self.mcols)
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        
        return xdf.fillna(0), ydf.fillna(0)
