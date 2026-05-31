"""
特征生成模块 (feat.py)

功能说明：
本模块负责从原始交易数据中生成预测特征。
基于市场微观结构理论，提取订单簿、交易流、动量反转等特征。

主要功能：
1. 订单簿不平衡特征（ob_imb系列）
2. 交易不平衡特征（trade_imb系列）
3. 动量/反转特征（lagret12）
4. 特征标准化和缺失值处理

理论基础：
    - Kyle模型：订单簿不平衡反映价格压力
    - 动量效应：过去收益延续
    - 反转效应：过去收益反转
    - 市场微观结构：订单流包含信息

当前特征（共6个）：
    1. ob_imb0: 买一卖一不平衡
    2. ob_imb4: 买1-5档vs卖1-5档不平衡
    3. ob_imb9: 买5-9档vs卖5-9档不平衡
    4. trade_imb: 主买主卖不平衡
    5. trade_imbema5: 交易不平衡的指数平滑
    6. lagret12: 过去12分钟超额收益率

注意：
    当前仅基线特征，需要扩展至50-100+个特征以提升性能
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
            特征名称列表，共6个特征
            
        说明：
            作为类方法，无需实例化即可调用
            用于在其他地方引用特征名称
        """
        return [
            "ob_imb0",        # 订单簿不平衡（买一卖一）
            "ob_imb4",        # 订单簿不平衡（买1-5档vs卖1-5档）
            "ob_imb9",        # 订单簿不平衡（买5-9档vs卖5-9档）
            "trade_imb",      # 交易不平衡（主买vs主卖）
            "trade_imbema5",  # 交易不平衡的5期指数平滑
            "lagret12",       # 过去12分钟超额收益率
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
        
        # === 提取特征和标签 ===
        # 特征DataFrame：主键 + 特征列，设置主键为索引
        xdf = df[self.mcols + self.featureNames()].set_index(self.mcols)
        # 标签DataFrame：主键 + 目标列，设置主键为索引
        ydf = df[self.mcols + [self.ycol]].set_index(self.mcols)
        
        # 缺失值填充为0（简单处理，可改进为插值等）
        return xdf.fillna(0), ydf.fillna(0)
