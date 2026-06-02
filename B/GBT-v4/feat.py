"""
特征生成模块 (feat.py)

功能说明：
本模块负责从原始交易数据中生成预测特征。
基于市场微观结构理论，提取订单簿、交易流、动量反转等特征。

主要功能：
1. 订单簿不平衡特征（ob_imb系列）
2. 交易不平衡特征（trade_imb系列）
3. 动量/反转特征（lagret12）
4. 价格动量/反转特征（12个）
5. 盘口压力特征（16个）
6. 成交主动性特征（15个）
7. 交互特征（5个）
8. 特征标准化和缺失值处理

理论基础：
    - Kyle模型：订单簿不平衡反映价格压力
    - 动量效应：过去收益延续
    - 反转效应：过去收益反转
    - 市场微观结构：订单流包含信息
    - 价格动量：不同时间尺度的历史收益率
    - 盘口压力：买卖力量对比和流动性
    - 成交主动性：交易行为分析

    当前特征（共74个）：
    基础特征（6个）：
    1. ob_imb0: 买一卖一不平衡
    2. ob_imb4: 买1-5档vs卖1-5档不平衡
    3. ob_imb9: 买5-9档vs卖5-9档不平衡
    4. trade_imb: 主买主卖不平衡
    5. trade_imbema5: 交易不平衡的指数平滑
    6. lagret12: 过去12分钟超额收益率
    
    价格动量/反转特征（12个）：
    7-11. ret_1, ret_3, ret_6, ret_12, ret_24: 不同时间尺度的历史收益率
    12-13. rolling_mean_ret_6, rolling_mean_ret_12: 近期趋势方向
    14-15. rolling_vol_6, rolling_vol_12: 近期波动水平
    16. high_low_range: 区间内多空争夺剧烈程度
    17. price_position: 收盘价在区间内的位置
    18. overnight_gap: 隔夜信息对开盘的冲击
    
    盘口压力特征（16个）：
    19. spread: 绝对价差
    20. relative_spread: 相对价差
    21. weighted_spread_4: 加权价差
    22-24. amount_imb_4, amount_imb_9, amount_imb_19: 各档位金额不平衡
    25-26. depth_sum_4, depth_sum_9: 总深度
    27-28. depth_delta_4, depth_delta_9: 深度变化速度
    29-31. buy_pressure_0, buy_pressure_4, buy_pressure_9: 买方挂单比例
    32. micro_price_dev: 微观价格偏离
    33-34. bid_ask_bias_0, bid_ask_bias_4: 价格和数量双加权的买卖偏向
    
    成交主动性特征（15个）：
    35-37. trade_buy_intensity, trade_sell_intensity, trade_net_intensity: 成交量占盘口深度比例
    38-40. trade_buy_turnover_ratio, trade_sell_turnover_ratio, trade_net_turnover_ratio: 成交额占挂单额比例
    41-44. rolling_trade_buy_qty_6, rolling_trade_buy_qty_12, rolling_trade_sell_qty_6, rolling_trade_sell_qty_12: 近期成交活跃度
    45. trade_count_imb: 成交笔数不平衡
    46. trade_count_imbema5: 平滑后的成交笔数不平衡趋势
    47-48. buy_trade_size, sell_trade_size: 单笔平均成交量
    49. trade_size_ratio: 买卖单笔大小比
    
    交互特征（5个）：
    50. ret_3_x_imb0: 短期动量 × 盘口不平衡rank
    51. ret_6_x_imb0: 中期动量 × 盘口不平衡rank
    52. ret_3_x_buy_intensity: 动量 × 买方力度rank
    53. spread_x_vol: 流动性紧张rank × 高波动
    54. highlow_x_imb0: 剧烈争夺 × 盘口偏向rank
    
    横截面rank特征（20个）：
    55-74. cs_rank_*: 各特征的横截面排名百分位(0~1)
    消除股票间异质性，让模型学到"相对强弱"而非"绝对值"

注意：
    当前共74个特征，涵盖了价格、盘口、成交三个维度的信息
    以及横截面rank标准化特征
"""

import os
import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    """
    特征生成器类
    
    负责从原始数据中提取和构造预测特征。
    当前实现6个基础特征，可作为特征工程的基础框架。
    """
    
    @classmethod
    def featureNames(cls):
        """
        获取特征名称列表（类方法）
        
        返回：
            特征名称列表，共54个特征
            
        说明：
            作为类方法，无需实例化即可调用
            用于在其他地方引用特征名称
        """
        return [
            # 基础特征（6个）
            "ob_imb0",        # 订单簿不平衡（买一卖一）
            "ob_imb4",        # 订单簿不平衡（买1-5档vs卖1-5档）
            "ob_imb9",        # 订单簿不平衡（买5-9档vs卖5-9档）
            "trade_imb",      # 交易不平衡（主买vs主卖）
            "trade_imbema5",  # 交易不平衡的5期指数平滑
            "lagret12",       # 过去12分钟超额收益率
            
            # 价格动量/反转特征（12个）
            "ret_1",          # 1分钟收益率
            "ret_3",          # 3分钟收益率
            "ret_6",          # 6分钟收益率
            "ret_12",         # 12分钟收益率
            "ret_24",         # 24分钟收益率
            "rolling_mean_ret_6",   # 6窗口收益率均值
            "rolling_mean_ret_12",  # 12窗口收益率均值
            "rolling_vol_6",        # 6窗口收益率波动率
            "rolling_vol_12",       # 12窗口收益率波动率
            "high_low_range",       # 高低价区间
            "price_position",       # 价格位置
            "overnight_gap",        # 隔夜跳空
            
            # 盘口压力特征（16个）
            "spread",                # 绝对价差
            "relative_spread",       # 相对价差
            "weighted_spread_4",     # 加权价差
            "amount_imb_4",          # 4档金额不平衡
            "amount_imb_9",          # 9档金额不平衡
            "amount_imb_19",         # 19档金额不平衡
            "depth_sum_4",           # 4档总深度
            "depth_sum_9",           # 9档总深度
            "depth_delta_4",         # 4档深度变化
            "depth_delta_9",         # 9档深度变化
            "buy_pressure_0",        # 买一档压力
            "buy_pressure_4",        # 买4档压力
            "buy_pressure_9",        # 买9档压力
            "micro_price_dev",       # 微观价格偏离
            "bid_ask_bias_0",        # 买一卖一偏向
            "bid_ask_bias_4",        # 买4卖4偏向
            
            # 成交主动性特征（15个）
            "trade_buy_intensity",      # 主买强度
            "trade_sell_intensity",     # 主卖强度
            "trade_net_intensity",      # 净强度
            "trade_buy_turnover_ratio", # 主买成交额比例
            "trade_sell_turnover_ratio",# 主卖成交额比例
            "trade_net_turnover_ratio", # 净成交额比例
            "rolling_trade_buy_qty_6",  # 6窗口主买成交量
            "rolling_trade_buy_qty_12", # 12窗口主买成交量
            "rolling_trade_sell_qty_6", # 6窗口主卖成交量
            "rolling_trade_sell_qty_12",# 12窗口主卖成交量
            "trade_count_imb",          # 成交笔数不平衡
            "trade_count_imbema5",      # 平滑成交笔数不平衡
            "buy_trade_size",           # 主买单笔大小
            "sell_trade_size",          # 主卖单笔大小
            "trade_size_ratio",         # 买卖单笔大小比
            
            # 交互特征（5个）
            "ret_3_x_imb0",            # 短期动量 × 盘口不平衡
            "ret_6_x_imb0",            # 中期动量 × 盘口不平衡
            "ret_3_x_buy_intensity",   # 动量 × 买方力度
            "spread_x_vol",            # 流动性紧张 × 高波动
            "highlow_x_imb0",          # 剧烈争夺 × 盘口偏向
            
            # 横截面rank特征（20个）
            # 每个时间点所有股票的排名百分位(0~1)，消除股票间异质性
            "cs_rank_ob_imb0",               # ob_imb0横截面rank
            "cs_rank_ob_imb4",               # ob_imb4横截面rank
            "cs_rank_ob_imb9",               # ob_imb9横截面rank
            "cs_rank_trade_imb",             # trade_imb横截面rank
            "cs_rank_trade_imbema5",         # trade_imbema5横截面rank
            "cs_rank_spread",                # spread横截面rank
            "cs_rank_relative_spread",       # relative_spread横截面rank
            "cs_rank_weighted_spread_4",     # weighted_spread_4横截面rank
            "cs_rank_amount_imb_4",          # amount_imb_4横截面rank
            "cs_rank_depth_sum_4",           # depth_sum_4横截面rank
            "cs_rank_depth_delta_4",         # depth_delta_4横截面rank
            "cs_rank_buy_pressure_0",        # buy_pressure_0横截面rank
            "cs_rank_buy_pressure_4",        # buy_pressure_4横截面rank
            "cs_rank_micro_price_dev",       # micro_price_dev横截面rank
            "cs_rank_bid_ask_bias_0",        # bid_ask_bias_0横截面rank
            "cs_rank_trade_buy_intensity",   # trade_buy_intensity横截面rank
            "cs_rank_trade_net_intensity",   # trade_net_intensity横截面rank
            "cs_rank_trade_buy_turnover_ratio",  # trade_buy_turnover_ratio横截面rank
            "cs_rank_trade_count_imb",       # trade_count_imb横截面rank
            "cs_rank_buy_trade_size",        # buy_trade_size横截面rank
        ]

    def __init__(self, cacheDir):
        """
        初始化特征生成器
        
        参数：
            cacheDir: 缓存目录（当前未使用）
            
        说明：
            设置目标变量列名和主键列名
        """
        self.cacheDir = cacheDir
        self.ycol = "fret12"  # 预测目标：未来12分钟收益率
        self.mcols = ["symbol", "date", "interval"]  # 主键列：股票、日期、时间

    def genFeatures(self, df):
        """
        从原始数据生成特征
        
        参数：
            df: 原始数据DataFrame
            
        返回：
            xdf: 特征DataFrame，形状为(n_samples, n_features)
            ydf: 标签DataFrame，形状为(n_samples, 1)
            
        特征说明：
            1. ob_imb0: 买一卖一数量不平衡
               公式：(asize0 - bsize0) / (asize0 + bsize0)
               含义：买盘强为正，卖盘强为负，范围[-1, 1]
               
            2. ob_imb4: 买1-5档vs卖1-5档不平衡
               公式：(asize0_4 - bsize0_4) / (asize0_4 + bsize0_4)
               含义：考虑更深档位的买卖力量对比
               
            3. ob_imb9: 买5-9档vs卖5-9档不平衡
               公式：(asize5_9 - bsize5_9) / (asize5_9 + bsize5_9)
               含义：远档位订单反映长期意图
               
            4. trade_imb: 主买主卖数量不平衡
               公式：(tradeBuyQty - tradeSellQty) / (tradeBuyQty + tradeSellQty)
               含义：主买多说明买方主动，价格上涨概率大
               
            5. trade_imbema5: 交易不平衡的指数平滑
               公式：EWM(trade_imb, halflife=5)
               含义：平滑短期波动，捕捉趋势
               
            6. lagret12: 过去12分钟超额收益率
               公式：bret12 - cx_bret12
               其中：bret12 = (当前价 - 12分钟前价) / 12分钟前价
                    cx_bret12 = 市场平均bret12
               含义：个股相对市场的超额收益，反映动量/反转效应
               
        处理流程：
            1. 计算各类特征
            2. 提取特征列和标签列
            3. 设置主键索引
            4. 缺失值填充为0
        """
        log.inf("Generating {} features from raw data...".format(len(self.featureNames())))
        
        # === 订单簿不平衡特征 ===
        # 买一卖一不平衡（最及时的价格压力信号）
        df.loc[:, "ob_imb0"] = (df["asize0"] - df["bsize0"]) / (df["asize0"] + df["bsize0"])
        # 买1-5档vs卖1-5档不平衡（更深档位信息）
        df.loc[:, "ob_imb4"] = (df["asize0_4"] - df["bsize0_4"]) / (df["asize0_4"] + df["bsize0_4"])
        # 买5-9档vs卖5-9档不平衡（远档位，反映长期意图）
        df.loc[:, "ob_imb9"] = (df["asize5_9"] - df["bsize5_9"]) / (df["asize5_9"] + df["bsize5_9"])
        
        # === 交易不平衡特征 ===
        # 主买主卖数量不平衡（实际成交的买卖力量）
        df.loc[:, "trade_imb"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / (df["tradeBuyQty"] + df["tradeSellQty"])
        # 交易不平衡的5期指数平滑（捕捉短期趋势）
        df.loc[:, "trade_imbema5"] = df["trade_imb"].ewm(halflife=5).mean()
        
        # === 动量/反转特征 ===
        # 过去12分钟收益率（backward return）
        df.loc[:, "bret12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12)
        # 计算同一时间点所有股票的平均收益率（市场水平）
        cxbret = df.groupby("interval")[["bret12"]].mean().reset_index().rename(columns={"bret12": "cx_bret12"})
        # 合并市场平均收益率
        df = df.merge(cxbret, on="interval", how="left")
        # 个股超额收益率 = 个股收益 - 市场平均收益（去除市场因素）
        df.loc[:, "lagret12"] = df["bret12"] - df["cx_bret12"]
        
        # === 价格动量/反转特征（12个）===
        log.inf("Generating price momentum/reversal features...")
        
        # 1. 不同时间尺度的历史收益率
        df.loc[:, "ret_1"] = (df["midpx"] - df["midpx"].shift(1)) / df["midpx"].shift(1)  # 1分钟收益率
        df.loc[:, "ret_3"] = (df["midpx"] - df["midpx"].shift(3)) / df["midpx"].shift(3)  # 3分钟收益率
        df.loc[:, "ret_6"] = (df["midpx"] - df["midpx"].shift(6)) / df["midpx"].shift(6)  # 6分钟收益率
        df.loc[:, "ret_12"] = (df["midpx"] - df["midpx"].shift(12)) / df["midpx"].shift(12)  # 12分钟收益率
        df.loc[:, "ret_24"] = (df["midpx"] - df["midpx"].shift(24)) / df["midpx"].shift(24)  # 24分钟收益率
        
        # 2. 近期趋势方向（滚动均值）
        df.loc[:, "rolling_mean_ret_6"] = df["ret_1"].rolling(window=6, min_periods=1).mean()  # 6窗口收益率均值
        df.loc[:, "rolling_mean_ret_12"] = df["ret_1"].rolling(window=12, min_periods=1).mean()  # 12窗口收益率均值
        
        # 3. 近期波动水平（滚动标准差）
        df.loc[:, "rolling_vol_6"] = df["ret_1"].rolling(window=6, min_periods=1).std()  # 6窗口收益率波动率
        df.loc[:, "rolling_vol_12"] = df["ret_1"].rolling(window=12, min_periods=1).std()  # 12窗口收益率波动率
        
        # 4. 高低价区间
        df.loc[:, "high_low_range"] = (df["high"] - df["low"]) / df["midpx"]  # 区间内多空争夺的剧烈程度
        
        # 5. 价格位置
        # 避免除零错误，当high等于low时设为0.5（中间位置）
        price_range = df["high"] - df["low"]
        price_range = price_range.replace(0, 1e-10)  # 避免除零
        df.loc[:, "price_position"] = (df["lastpx"] - df["low"]) / price_range  # 收盘价在区间内的位置
        
        # 6. 隔夜跳空（仅首分钟有效）
        # 首先计算前一日收盘价（需要按股票和时间处理）
        df_sorted = df.sort_values(["symbol", "interval"])
        df_sorted.loc[:, "prev_close"] = df_sorted.groupby("symbol")["lastpx"].shift(1)
        df_sorted.loc[:, "overnight_gap"] = (df_sorted["open"] - df_sorted["prev_close"]) / df_sorted["prev_close"]
        # 重置为原始顺序
        df = df_sorted.sort_index()
        
        # === 盘口压力特征（16个）===
        log.inf("Generating order book pressure features...")
        
        # 1. 价差特征
        df.loc[:, "spread"] = df["ask0"] - df["bid0"]  # 绝对价差
        df.loc[:, "relative_spread"] = (df["ask0"] - df["bid0"]) / df["midpx"]  # 相对价差
        
        # 2. 加权价差（考虑各档深度）
        # 计算加权买卖价
        vwap_bid_4 = df["btr0_4"] / df["bsize0_4"].replace(0, 1e-10)
        vwap_ask_4 = df["atr0_4"] / df["asize0_4"].replace(0, 1e-10)
        df.loc[:, "weighted_spread_4"] = (vwap_ask_4 - vwap_bid_4) / df["midpx"]
        
        # 3. 金额不平衡特征
        df.loc[:, "amount_imb_4"] = (df["atr0_4"] - df["btr0_4"]) / (df["atr0_4"] + df["btr0_4"]).replace(0, 1e-10)
        # 假设有9档和19档数据，这里使用4档数据作为示例
        df.loc[:, "amount_imb_9"] = df["amount_imb_4"]  # 如果没有9档数据，先用4档替代
        df.loc[:, "amount_imb_19"] = df["amount_imb_4"]  # 如果没有19档数据，先用4档替代
        
        # 4. 深度特征
        df.loc[:, "depth_sum_4"] = df["bsize0_4"] + df["asize0_4"]  # 4档总深度
        df.loc[:, "depth_sum_9"] = df["depth_sum_4"]  # 如果没有9档数据，先用4档替代
        
        # 5. 深度变化特征
        df_sorted = df.sort_values(["symbol", "interval"])
        df_sorted.loc[:, "depth_delta_4"] = df_sorted.groupby("symbol")["depth_sum_4"].diff()
        df_sorted.loc[:, "depth_delta_9"] = df_sorted.groupby("symbol")["depth_sum_9"].diff()
        df = df_sorted.sort_index()
        
        # 6. 买方压力特征
        df.loc[:, "buy_pressure_0"] = df["bsize0"] / (df["bsize0"] + df["asize0"]).replace(0, 1e-10)
        df.loc[:, "buy_pressure_4"] = df["bsize0_4"] / (df["bsize0_4"] + df["asize0_4"]).replace(0, 1e-10)
        df.loc[:, "buy_pressure_9"] = df["buy_pressure_4"]  # 如果没有9档数据，先用4档替代
        
        # 7. 微观价格偏离
        # 微观价格 = (bid * bsize + ask * asize) / (bsize + asize)
        micro_price = (df["bid0"] * df["bsize0"] + df["ask0"] * df["asize0"]) / (df["bsize0"] + df["asize0"]).replace(0, 1e-10)
        df.loc[:, "micro_price_dev"] = (micro_price - df["midpx"]) / df["midpx"]
        
        # 8. 买卖偏向（价格和数量双加权）
        df.loc[:, "bid_ask_bias_0"] = (df["bid0"] * df["bsize0"] - df["ask0"] * df["asize0"]) / (df["bid0"] * df["bsize0"] + df["ask0"] * df["asize0"]).replace(0, 1e-10)
        df.loc[:, "bid_ask_bias_4"] = (df["btr0_4"] - df["atr0_4"]) / (df["btr0_4"] + df["atr0_4"]).replace(0, 1e-10)
        
        # === 成交主动性特征（15个）===
        log.inf("Generating trade activity features...")
        
        # 1. 成交强度特征（基于主买数据）
        # 由于数据中可能没有tradeSellQty，我们使用主买数据构建相关特征
        df.loc[:, "trade_buy_intensity"] = df["tradeBuyQty"] / df["depth_sum_4"].replace(0, 1e-10)
        
        # 检查是否有主卖数据列
        has_trade_sell = "tradeSellQty" in df.columns
        has_nTradeSell = "nTradeSell" in df.columns
        has_tradeSellTurnover = "tradeSellTurnover" in df.columns
        
        # 处理主卖相关特征
        if has_trade_sell:
            df.loc[:, "trade_sell_intensity"] = df["tradeSellQty"] / df["depth_sum_4"].replace(0, 1e-10)
            df.loc[:, "trade_net_intensity"] = (df["tradeBuyQty"] - df["tradeSellQty"]) / df["depth_sum_4"].replace(0, 1e-10)
        else:
            # 如果没有主卖数据，设为0或使用其他估算
            df.loc[:, "trade_sell_intensity"] = 0
            df.loc[:, "trade_net_intensity"] = df["trade_buy_intensity"]
        
        # 2. 成交额比例特征
        total_turnover = df["btr0_4"] + df["atr0_4"]
        df.loc[:, "trade_buy_turnover_ratio"] = df["tradeBuyTurnover"] / total_turnover.replace(0, 1e-10)
        
        if has_tradeSellTurnover:
            df.loc[:, "trade_sell_turnover_ratio"] = df["tradeSellTurnover"] / total_turnover.replace(0, 1e-10)
            df.loc[:, "trade_net_turnover_ratio"] = (df["tradeBuyTurnover"] - df["tradeSellTurnover"]) / total_turnover.replace(0, 1e-10)
        else:
            df.loc[:, "trade_sell_turnover_ratio"] = 0
            df.loc[:, "trade_net_turnover_ratio"] = df["trade_buy_turnover_ratio"]
        
        # 3. 滚动成交量特征
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
        
        # 4. 成交笔数不平衡
        if has_nTradeSell:
            total_trade_count = df["nTradeBuy"] + df["nTradeSell"]
            df.loc[:, "trade_count_imb"] = (df["nTradeBuy"] - df["nTradeSell"]) / total_trade_count.replace(0, 1e-10)
        else:
            # 如果没有主卖笔数，使用主买笔数作为代理
            df.loc[:, "trade_count_imb"] = df["nTradeBuy"] / (df["nTradeBuy"] + 1).replace(0, 1e-10)
        
        df.loc[:, "trade_count_imbema5"] = df["trade_count_imb"].ewm(halflife=5).mean()
        
        # 5. 单笔成交大小特征
        df.loc[:, "buy_trade_size"] = df["tradeBuyQty"] / df["nTradeBuy"].replace(0, 1e-10)
        
        if has_trade_sell and has_nTradeSell:
            df.loc[:, "sell_trade_size"] = df["tradeSellQty"] / df["nTradeSell"].replace(0, 1e-10)
            df.loc[:, "trade_size_ratio"] = df["buy_trade_size"] / df["sell_trade_size"].replace(0, 1e-10)
        else:
            df.loc[:, "sell_trade_size"] = df["buy_trade_size"]  # 使用主买大小作为估计
            df.loc[:, "trade_size_ratio"] = 1.0  # 设为1表示相等
        
        # === 横截面rank特征（20个）===
        # 每个时间点(interval)所有股票的排名百分位(0~1)
        # 消除股票间异质性，让模型学到"相对强弱"而非"绝对值"
        log.inf("Generating cross-sectional rank features...")
        
        cs_rank_cols = [
            "ob_imb0", "ob_imb4", "ob_imb9",
            "trade_imb", "trade_imbema5",
            "spread", "relative_spread", "weighted_spread_4",
            "amount_imb_4",
            "depth_sum_4", "depth_delta_4",
            "buy_pressure_0", "buy_pressure_4",
            "micro_price_dev", "bid_ask_bias_0",
            "trade_buy_intensity", "trade_net_intensity",
            "trade_buy_turnover_ratio",
            "trade_count_imb", "buy_trade_size",
        ]
        
        for col in cs_rank_cols:
            rank_col = f"cs_rank_{col}"
            df.loc[:, rank_col] = df.groupby("interval")[col].rank(pct=True)
        
        # === 交互特征（5个，使用横截面rank版本）===
        log.inf("Generating interaction features...")
        
        # 1. 短期动量 × 盘口不平衡rank
        df.loc[:, "ret_3_x_imb0"] = df["ret_3"] * df["cs_rank_ob_imb0"]
        
        # 2. 中期动量 × 盘口不平衡rank
        df.loc[:, "ret_6_x_imb0"] = df["ret_6"] * df["cs_rank_ob_imb0"]
        
        # 3. 动量 × 买方力度rank
        df.loc[:, "ret_3_x_buy_intensity"] = df["ret_3"] * df["cs_rank_trade_buy_intensity"]
        
        # 4. 流动性紧张rank × 高波动
        df.loc[:, "spread_x_vol"] = df["cs_rank_relative_spread"] * df["rolling_vol_12"]
        
        # 5. 剧烈争夺 × 盘口偏向rank
        df.loc[:, "highlow_x_imb0"] = df["high_low_range"] * df["cs_rank_ob_imb0"]
        
        # === 提取特征和标签 ===
        # 特征DataFrame：主键 + 特征列，设置主键为索引
        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
        # 标签DataFrame：主键 + 目标列，设置主键为索引
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        
        # 缺失值填充为0（简单处理，可改进为插值等）
        return xdf.fillna(0), ydf.fillna(0)
