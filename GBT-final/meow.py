
"""
主引擎模块 (meow.py)

功能说明：
本模块是项目的核心入口，负责协调各模块完成完整的训练和评估流程。
整合数据加载、特征生成、模型训练、结果评估等所有步骤。

主要功能：
1. 初始化各个模块（数据加载器、特征生成器、模型、评估器）
2. 训练流程：加载数据 → 生成特征 → 训练模型
3. 评估流程：加载数据 → 生成特征 → 模型预测 → 评估结果

工作流程：
    训练阶段（fit）：
        1. 获取交易日列表
        2. 加载原始数据
        3. 生成特征和标签
        4. 训练模型
        
    评估阶段（eval）：
        1. 获取交易日列表
        2. 加载原始数据
        3. 生成特征和标签
        4. 模型预测
        5. 计算评估指标

使用示例：
    # 创建引擎
    engine = MeowEngine(h5dir="archive", cacheDir=None)
    
    # 训练（使用6月-11月数据）
    engine.fit(20230601, 20231130)
    
    # 评估（使用12月数据）
    prediction_path = os.environ.get("MEOW_PREDICTION_OUTPUT")
    engine.eval(20231201, 20231229, predictionPath=prediction_path)

项目结构：
    meow.py（主引擎）
        ├── log.py（日志）
        ├── tradingcalendar.py（交易日历）
        ├── dl.py（数据加载）
        ├── feat.py（特征生成）
        ├── mdl.py（模型）
        └── eval.py（评估）
"""

import os
import numpy as np
from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar


class MeowEngine(object):
    """
    主引擎类
    
    整合所有模块，提供统一的训练和评估接口。
    是项目的核心控制类，负责协调各模块的工作流程。
    """
    
    def __init__(self, h5dir, cacheDir, feature_set='full'):
        """
        参数：
            h5dir: H5数据文件所在目录
            cacheDir: 缓存目录
            feature_set: 特征集选择（'full','v4','no_p1','no_p0'），用于消融实验
        """
        # 加载交易日历
        self.calendar = Calendar()
        
        # 验证数据目录
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        
        self.cacheDir = cacheDir  # 缓存目录（当前未使用）
        
        # 创建各模块实例
        self.dloader = MeowDataLoader(h5dir=h5dir)           # 数据加载器
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir, feature_set=feature_set)
        self.model = MeowModel(cacheDir=cacheDir)            # 模型
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)    # 评估器

    def fit(self, startDate, endDate):
        """
        训练模型
        
        参数：
            startDate: 训练起始日期（包含），格式YYYYMMDD
            endDate: 训练结束日期（包含），格式YYYYMMDD
            
        训练流程：
            1. 获取日期范围内的交易日列表
            2. 加载所有交易日的原始数据
            3. 从原始数据生成特征和标签
            4. 使用特征和标签训练模型
            
        示例：
            engine.fit(20230601, 20231130)
            # 使用2023年6月1日至11月30日的数据训练
            
        时间参考：
            123个交易日，约880万样本，耗时约1分钟
        """
        # 获取交易日列表
        dates = self.calendar.range(startDate, endDate)
        # 加载原始数据
        rawData = self.dloader.loadDates(dates)
        log.inf("Running model fitting...")
        # 生成特征和标签
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        # 训练模型
        self.model.fit(xdf, ydf)
        
        # 生成训练曲线
        try:
            log.inf("生成训练曲线...")
            figure_path = self.model.plot_training_curves(output_dir='figures')
            if figure_path:
                log.inf(f"训练曲线已保存到: {figure_path}")
            else:
                log.yellow("无法生成训练曲线")
        except Exception as e:
            log.yellow(f"生成训练曲线时出错: {e}")

    def predict(self, xdf):
        """
        模型预测
        
        参数：
            xdf: 特征DataFrame
            
        返回：
            预测值数组
            
        说明：
            单独提供预测接口，可在评估之外使用
        """
        return self.model.predict(xdf)

    def eval(self, startDate, endDate, predictionPath=None):
        """
        评估模型
        
        参数：
            startDate: 评估起始日期（包含），格式YYYYMMDD
            endDate: 评估结束日期（包含），格式YYYYMMDD
            
        评估流程：
            1. 获取日期范围内的交易日列表
            2. 加载所有交易日的原始数据
            3. 从原始数据生成特征和标签
            4. 使用训练好的模型进行预测
            5. 计算评估指标（Pearson、R²、MSE）
            
        示例：
            engine.eval(20231201, 20231229)
            # 使用2023年12月1日至12月29日的数据评估
            
        时间参考：
            21个交易日，约144万样本，耗时约6秒
            
        输出示例：
            Meow evaluation summary: Pearson correlation=0.0222, R2=0.00046, MSE=0.00
        """
        log.inf("Running model evaluation...")
        # 获取交易日列表
        dates = self.calendar.range(startDate, endDate)
        # 加载原始数据
        rawData = self.dloader.loadDates(dates)
        # 生成特征和标签
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        # 模型预测，将预测结果添加到ydf
        predictions = self.predict(xdf)
        ydf.loc[:, "forecast"] = predictions
        # 计算评估指标
        self.evaluator.eval(ydf)
        if predictionPath is not None:
            self.savePredictions(ydf, predictionPath)
        
        # 生成预测可视化图表
        try:
            log.inf("生成预测可视化图表...")
            # 提取真实值和预测值
            y_true = ydf["fret12"].to_numpy()
            y_pred = predictions
            
            # 创建完整的训练报告
            generated_files = self.model.create_training_report(
                y_true=y_true,
                y_pred=y_pred,
                output_dir='figures'
            )
            
            if generated_files:
                log.inf(f"生成了 {len(generated_files)} 个可视化图表:")
                for file_path in generated_files:
                    log.inf(f"  - {file_path}")
            else:
                log.yellow("无法生成可视化图表")
        except Exception as e:
            log.yellow(f"生成可视化图表时出错: {e}")

    def savePredictions(self, ydf, predictionPath):
        os.makedirs(os.path.dirname(predictionPath) or ".", exist_ok=True)
        ydf.reset_index().to_csv(predictionPath, index=False)
        log.inf("Saved predictions to {}".format(predictionPath))

    def tune(self, startDate, endDate, n_trials=100, algorithm='bayesian',
             n_splits=5, early_stopping_rounds=50, timeout_per_trial=1800,
             output_dir='tuning_output', apply_best=True, sample_ratio=0.3,
             generate_plots=False):
        """
        自动调参：基于项目模型fit->predict->eval闭环，逐步推导最优参数
        
        参数：
            startDate: 训练数据起始日期
            endDate: 训练数据结束日期
            n_trials: 最大试验次数
            algorithm: 'bayesian' 或 'random'
            n_splits: 时序交叉验证折数
            early_stopping_rounds: 早停轮数
            timeout_per_trial: 单次试验超时秒数
            output_dir: 调参输出目录
            apply_best: 是否自动将最优参数应用到当前模型
            sample_ratio: 调参时使用的数据比例（0-1），默认0.3
                调参用子集找参数方向，最终训练用全量数据
                超参数的最优值在不同数据量下趋势基本一致
                
        返回：
            最优超参数字典
        """
        from tuner import MeowTuner
        
        dates = self.calendar.range(startDate, endDate)
        rawData = self.dloader.loadDates(dates)
        log.inf("Preparing data for auto-tuning...")
        xdf, ydf = self.featGenerator.genFeatures(rawData)
        
        X = xdf.to_numpy().astype(np.float32)
        y = ydf.to_numpy().ravel().astype(np.float32)
        
        log.inf(f"Tuning data shape: X={X.shape}, y={y.shape}")
        
        tuner = MeowTuner(output_dir=output_dir)
        best_params = tuner.tune(
            X, y,
            n_trials=n_trials,
            algorithm=algorithm,
            n_splits=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            timeout_per_trial=timeout_per_trial,
            generate_plots=generate_plots,
            sample_ratio=sample_ratio,
        )
        
        if apply_best:
            log.inf("Applying best params to current model...")
            self.model.update_params(best_params)
            self.model.update_params({'early_stopping_rounds': early_stopping_rounds})
            log.inf("Best params applied. Call engine.fit() to train with optimized params.")
        
        return best_params


if __name__ == "__main__":
    """
    主程序入口
    
    执行流程：
        1. 创建引擎实例
        2. 训练模型（使用6月-11月数据）
        3. 评估模型（使用12月数据）
        
    数据划分：
        训练集：20230601 - 20231130（约123个交易日）
        测试集：20231201 - 20231229（约21个交易日）
        
    注意：
        运行前需要修改h5dir为实际数据目录路径
        当前使用相对路径"archive"，需要确保archive目录存在且包含数据文件
    """
    # 创建引擎（修改h5dir为你的数据目录）
    engine = MeowEngine(h5dir="../../archive", cacheDir=None)
    
    # 训练模型（使用6月至11月数据）
    engine.fit(20230601, 20231130)
    
    # 评估模型（使用12月数据）
    engine.eval(20231201, 20231229)
