"""
评估模块 (eval.py)

功能说明：
本模块负责评估模型预测结果，计算多个性能指标。
用于量化预测效果，指导模型优化。

主要功能：
1. 计算Pearson相关系数
2. 计算R²决定系数
3. 计算均方误差MSE

评估指标：
    1. Pearson相关系数（主要指标）
       - 衡量预测值与真实值的线性相关性
       - 范围：[-1, 1]，越大越好
       - 业内参考：2-3%（基线），5-10%（良好），20-30%（优秀）
       
    2. R²决定系数
       - 衡量模型解释目标变量方差的比例
       - 范围：(-∞, 1]，越接近1越好
       - 0表示模型与均值预测相当
       
    3. 均方误差MSE
       - 衡量预测误差的平均大小
       - 范围：[0, +∞)，越小越好
       - 对异常值敏感

使用示例：
    evaluator = MeowEvaluator(cacheDir=None)
    ydf['forecast'] = predictions
    evaluator.eval(ydf)
"""

import os
import numpy as np
import pandas as pd
from log import log


class MeowEvaluator(object):
    """
    评估器类
    
    负责计算模型预测性能的各种指标。
    """
    
    def __init__(self, cacheDir):
        """
        初始化评估器
        
        参数：
            cacheDir: 缓存目录（当前未使用）
            
        说明：
            设置预测列名和真实标签列名
        """
        self.cacheDir = cacheDir
        self.predictionCol = "forecast"  # 预测值列名
        self.ycol = "fret12"            # 真实值列名

    def eval(self, ydf):
        """
        评估模型预测结果
        
        参数：
            ydf: DataFrame，包含预测值和真实值
                - forecast列：模型预测值
                - fret12列：真实值（未来12分钟收益率）
                
        计算的指标：
            1. Pearson相关系数
               公式：cov(pred, y) / (std(pred) * std(y))
               含义：预测值与真实值的线性相关程度
               重要性：*** 主要评估指标 ***
               
            2. R²决定系数
               公式：1 - Σ(pred - y)² / (n * var(y))
               含义：模型解释方差的比例
               重要性：** 辅助评估指标 **
               
            3. 均方误差MSE
               公式：Σ(pred - y)² / n
               含义：预测误差的平方均值
               重要性：** 辅助评估指标 **
               
        输出格式：
            Meow evaluation summary: Pearson correlation=0.0222, R2=0.00046, MSE=0.00
            
        说明：
            先处理无穷值和缺失值，再计算指标
            MSE可能很小（收益率本身数值小），主要看Pearson和R²
        """
        # 处理无穷值和缺失值（替换为NaN再填充为0）
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # === Pearson相关系数 ===
        # 计算预测值和真实值的相关系数矩阵，取(0,1)位置
        pcor = ydf[[self.predictionCol, self.ycol]].corr().to_numpy()[0, 1]
        
        # === R²决定系数 ===
        # R² = 1 - SSE / (n * var(y))
        # SSE: 残差平方和
        r2 = 1 - ((ydf[self.predictionCol] - ydf[self.ycol]) ** 2).sum() / ydf[self.ycol].var() / ydf.shape[0]
        
        # === 均方误差MSE ===
        # MSE = Σ(pred - y)² / n
        mse = ((ydf[self.predictionCol] - ydf[self.ycol]) ** 2).sum() / ydf.shape[0]
        
        # 输出评估结果（格式化保留适当小数位）
        log.inf("Meow evaluation summary: Pearson correlation={:.4f}, R2={:.5f}, MSE={:.2f}".format(pcor, r2, mse))
