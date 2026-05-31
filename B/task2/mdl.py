"""
模型模块 (mdl.py)

功能说明：
本模块定义预测模型，负责模型的训练和预测。
当前使用Ridge线性回归作为基线模型。

主要功能：
1. 模型初始化和参数配置
2. 模型训练（fit）
3. 模型预测（predict）

当前模型：
    Ridge回归（L2正则化的线性回归）
    - alpha=0.5：正则化强度
    - fit_intercept=False：不拟合截距
    - tol=1e-8：收敛精度

模型选择建议：
    基线：Ridge回归（当前）
    改进：LightGBM、XGBoost、神经网络等非线性模型
    
性能参考：
    当前Ridge模型：Pearson相关系数约2-3%
    目标：提升至5-10%以上
"""

import os
from sklearn.linear_model import Ridge
from log import log


class MeowModel(object):
    """
    模型类
    
    封装机器学习模型，提供统一的训练和预测接口。
    当前使用sklearn的Ridge回归作为基线模型。
    """
    
    def __init__(self, cacheDir):
        """
        初始化模型
        
        参数：
            cacheDir: 缓存目录（当前未使用）
            
        模型参数说明：
            alpha=0.5: L2正则化强度
                - 越大正则化越强，防止过拟合
                - 当前0.5为经验值，可调优
                
            fit_intercept=False: 是否拟合截距
                - False：假设数据已中心化或特征包含偏置
                - 金融数据通常已标准化，可不拟合截距
                
            tol=1e-8: 优化收敛精度
                - 精度越高，训练越准确但越慢
                
            random_state=None: 随机种子
                - None：不固定随机性
        """
        self.estimator = Ridge(
            alpha=0.5,           # L2正则化强度
            random_state=None,   # 随机种子
            fit_intercept=False, # 不拟合截距
            tol=1e-8            # 收敛精度
        )

    def fit(self, xdf, ydf):
        """
        训练模型
        
        参数：
            xdf: 特征DataFrame，形状为(n_samples, n_features)
            ydf: 标签DataFrame，形状为(n_samples, 1)
            
        训练过程：
            Ridge回归求解：min ||y - Xw||² + alpha * ||w||²
            其中w为权重向量，alpha为正则化强度
            
        说明：
            将DataFrame转换为numpy数组后训练
            Ridge回归有解析解，训练速度快
        """
        self.estimator.fit(
            X=xdf.to_numpy(),  # 特征矩阵
            y=ydf.to_numpy(),  # 标签向量
        )
        log.inf("Done fitting")

    def predict(self, xdf):
        """
        模型预测
        
        参数：
            xdf: 特征DataFrame，形状为(n_samples, n_features)
            
        返回：
            预测值数组，形状为(n_samples,)
            
        说明：
            使用训练好的模型进行预测
            预测值 = X @ w（线性组合）
        """
        return self.estimator.predict(xdf.to_numpy())
