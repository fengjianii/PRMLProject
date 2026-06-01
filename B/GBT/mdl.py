"""
模型模块 (mdl.py)

功能说明：
本模块定义预测模型，负责模型的训练和预测。
使用LightGBM GBT（Gradient Boosting Tree）作为主力模型。

主要功能：
1. 模型初始化和参数配置
2. 模型训练（fit）带早停和交叉验证
3. 模型预测（predict）
4. 防过拟合措施

当前模型：
    LightGBM GBT回归
    - 树数量：100（带早停）
    - 学习率：0.1
    - 最大深度：3（防止过拟合）
    - 特征采样：0.8
    - 样本采样：0.8
    - L2正则化：1.0

防过拟合措施：
    1. 限制树深度（max_depth=3）
    2. 添加L2正则化（reg_lambda=1.0）
    3. 特征采样（feature_fraction=0.8）
    4. 样本采样（bagging_fraction=0.8）
    5. 早停机制（early_stopping_rounds=10）
    6. 交叉验证验证集

性能参考：
    基线Ridge模型：Pearson相关系数约2-3%
    目标GBT模型：Pearson相关系数目标5-10%以上
"""

import os
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from log import log


class MeowModel(object):
    """
    模型类
    
    封装机器学习模型，提供统一的训练和预测接口。
    使用LightGBM GBT回归模型，带防过拟合措施。
    """
    
    def __init__(self, cacheDir):
        """
        初始化GBT模型
        
        参数：
            cacheDir: 缓存目录（当前未使用）
            
        模型参数说明（防过拟合设计）：
            n_estimators=100: 树的数量（带早停机制）
            learning_rate=0.1: 学习率，较小值防止过拟合
            max_depth=3: 最大树深度，限制模型复杂度
            num_leaves=8: 叶子节点数（2^max_depth）
            min_child_samples=20: 叶子节点最小样本数
            subsample=0.8: 样本采样比例
            colsample_bytree=0.8: 特征采样比例
            reg_alpha=0.0: L1正则化
            reg_lambda=1.0: L2正则化，防止过拟合
            random_state=42: 固定随机种子保证可复现
            
        早停机制：
            early_stopping_rounds=10: 验证集性能10轮不提升则停止
            verbose_eval=False: 不打印训练过程
        """
        self.cacheDir = cacheDir
        self.model = None
        self.feature_names = None
        
        # LightGBM参数配置（防过拟合重点）
        self.params = {
            # 基础参数
            'objective': 'regression',
            'metric': 'l2',
            'boosting_type': 'gbdt',
            'verbose': -1,  # 不输出训练信息
            
            # GPU加速参数
            'device': 'gpu',           # 使用GPU(OpenCL)加速
            'gpu_device_id': 0,        # 使用第0号GPU（独显）
            'gpu_use_dp': False,       # 使用单精度浮点（GPU更快）
            'max_bin': 63,             # GPU模式下限制直方图bin数
            
            # 防过拟合参数
            'max_depth': 3,           # 限制树深度
            'num_leaves': 8,          # 叶子节点数（2^max_depth）
            'min_child_samples': 20,  # 叶子节点最小样本数
            'min_child_weight': 1e-3, # 叶子节点最小权重和
            
            # 采样参数（防止过拟合）
            'subsample': 0.8,         # 样本采样比例
            'colsample_bytree': 0.8,  # 特征采样比例
            'subsample_freq': 1,      # 采样频率
            
            # 正则化参数
            'reg_alpha': 0.0,         # L1正则化
            'reg_lambda': 1.0,        # L2正则化
            
            # 学习参数
            'learning_rate': 0.1,     # 较小学习率防止过拟合
            'n_estimators': 100,      # 树的数量（带早停）
            
            # 其他参数
            'random_state': 42,       # 固定随机种子
            'n_jobs': -1,             # 使用所有CPU核心
        }
        
        # 早停参数
        self.early_stopping_rounds = 10
        self.verbose_eval = False

    def fit(self, xdf, ydf):
        """
        训练GBT模型（带防过拟合措施）
        
        参数：
            xdf: 特征DataFrame，形状为(n_samples, n_features)
            ydf: 标签DataFrame，形状为(n_samples, 1)
            
        训练过程：
            1. 分割训练集和验证集（80%-20%）
            2. 创建LightGBM数据集
            3. 训练模型（带早停机制）
            4. 记录特征重要性
            5. 记录训练历史（用于可视化）
            
        防过拟合措施：
            1. 验证集早停：防止在训练集上过拟合
            2. 限制树深度：max_depth=3
            3. 正则化：L2正则化reg_lambda=1.0
            4. 采样：特征和样本采样
            5. 最小叶子样本：min_child_samples=20
        """
        # 保存特征名称（用于特征重要性分析）
        self.feature_names = list(xdf.columns)
        
        # 转换为numpy数组
        X = xdf.to_numpy()
        y = ydf.to_numpy().ravel()  # LightGBM需要1D数组
        
        # 分割训练集和验证集（防过拟合关键）
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2,      # 20%作为验证集
            random_state=42,    # 固定随机种子
            shuffle=False       # 时间序列数据不随机打乱
        )
        
        log.inf(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        
        # 创建LightGBM数据集
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names)
        
        # 训练模型（带早停机制）
        log.inf("Training GBT model with early stopping...")
        
        # 准备callbacks（避免None值）
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        if self.verbose_eval:
            callbacks.append(lgb.log_evaluation(period=100))
        
        # 添加训练历史记录回调
        self.train_history = {}
        callbacks.append(lgb.record_evaluation(self.train_history))
        
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            valid_sets=[train_data, val_data],
            valid_names=['training', 'validation'],
            callbacks=callbacks
        )
        
        # 记录训练信息
        best_iteration = self.model.best_iteration
        best_score = self.model.best_score['validation']['l2']
        log.inf(f"Training completed. Best iteration: {best_iteration}, Best validation L2: {best_score:.6f}")
        
        # 输出训练历史统计
        if 'validation' in self.train_history and 'l2' in self.train_history['validation']:
            train_losses = self.train_history.get('training', {}).get('l2', [])
            val_losses = self.train_history['validation']['l2']
            
            if train_losses and val_losses:
                log.inf(f"训练历史记录: {len(train_losses)} 轮迭代")
                log.inf(f"最终训练损失: {train_losses[-1]:.6f}, 最终验证损失: {val_losses[-1]:.6f}")
                
                # 检查过拟合迹象
                if len(val_losses) > 10:
                    last_10_train = train_losses[-10:]
                    last_10_val = val_losses[-10:]
                    
                    train_trend = np.mean(np.diff(last_10_train))
                    val_trend = np.mean(np.diff(last_10_val))
                    
                    if train_trend < 0 and val_trend > 0:
                        log.yellow("⚠️ 检测到过拟合风险: 训练损失下降但验证损失上升")
                    elif val_trend > 0:
                        log.yellow("⚠️ 验证损失有上升趋势")
        
        # 输出特征重要性（前10个）
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        log.inf("Top 10 feature importance (gain):")
        for feature, imp in feature_importance:
            log.inf(f"  {feature}: {imp:.4f}")
        
        log.inf("Done fitting")

    def predict(self, xdf):
        """
        GBT模型预测
        
        参数：
            xdf: 特征DataFrame，形状为(n_samples, n_features)
            
        返回：
            预测值数组，形状为(n_samples,)
            
        说明：
            使用训练好的LightGBM模型进行预测
            返回所有树的加权预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X = xdf.to_numpy()
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def get_feature_importance(self, top_n=20):
        """
        获取特征重要性
        
        参数：
            top_n: 返回前N个最重要的特征
            
        返回：
            特征重要性列表，按重要性降序排序
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = sorted(
            zip(self.feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        return feature_importance[:top_n]
    
    def save_model(self, filepath):
        """
        保存模型到文件
        
        参数：
            filepath: 模型保存路径
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        self.model.save_model(filepath)
        log.inf(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        从文件加载模型
        
        参数：
            filepath: 模型文件路径
        """
        self.model = lgb.Booster(model_file=filepath)
        log.inf(f"Model loaded from {filepath}")
    
    def get_training_history(self):
        """
        获取训练历史数据
        
        返回：
            训练历史字典，包含训练集和验证集的损失曲线
        """
        if not hasattr(self, 'train_history') or not self.train_history:
            log.yellow("训练历史未记录，请确保在fit()方法中启用了历史记录")
            return None
        
        return self.train_history
    
    def plot_training_curves(self, output_dir='figures'):
        """
        绘制训练和验证损失曲线
        
        参数：
            output_dir: 图表保存目录
            
        返回：
            图表文件路径
        """
        try:
            from visualization import MeowVisualizer
        except ImportError:
            log.red("可视化模块未找到，请确保visualization.py在相同目录")
            return None
        
        if not hasattr(self, 'train_history') or not self.train_history:
            log.yellow("训练历史未记录，无法绘制曲线")
            return None
        
        # 提取训练和验证损失
        train_losses = self.train_history.get('training', {}).get('l2', [])
        val_losses = self.train_history.get('validation', {}).get('l2', [])
        
        if not train_losses or not val_losses:
            log.yellow("训练历史中缺少损失数据")
            return None
        
        # 创建可视化器并绘制曲线
        visualizer = MeowVisualizer(output_dir=output_dir)
        save_path = visualizer.plot_training_curve(train_losses, val_losses)
        
        return save_path
    
    def create_training_report(self, y_true=None, y_pred=None, output_dir='figures'):
        """
        创建完整的训练报告
        
        参数：
            y_true: 真实值（可选，用于预测分析）
            y_pred: 预测值（可选，用于预测分析）
            output_dir: 输出目录
            
        返回：
            生成的图表文件路径列表
        """
        try:
            from visualization import MeowVisualizer
        except ImportError:
            log.red("可视化模块未找到，请确保visualization.py在相同目录")
            return []
        
        if not hasattr(self, 'train_history') or not self.train_history:
            log.yellow("训练历史未记录，无法生成报告")
            return []
        
        # 获取训练历史
        train_history = self.train_history
        
        # 获取特征重要性
        feature_importance = []
        if self.model is not None and self.feature_names is not None:
            importance = self.model.feature_importance(importance_type='gain')
            feature_importance = list(zip(self.feature_names, importance))
            # 按重要性排序
            feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 创建可视化器
        visualizer = MeowVisualizer(output_dir=output_dir)
        
        # 生成报告
        generated_files = visualizer.create_training_report(
            train_history=train_history,
            feature_importance=feature_importance,
            y_true=y_true,
            y_pred=y_pred
        )
        
        return generated_files
