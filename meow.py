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
    engine.eval(20231201, 20231229)

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
    
    def __init__(self, h5dir, cacheDir):
        """
        初始化引擎，创建所有模块实例
        
        参数：
            h5dir: H5数据文件所在目录
            cacheDir: 缓存目录（当前未使用，可扩展用于缓存特征、模型等）
            
        初始化流程：
            1. 加载交易日历
            2. 验证数据目录有效性
            3. 创建数据加载器
            4. 创建特征生成器
            5. 创建模型
            6. 创建评估器
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
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)  # 特征生成器
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

    def eval(self, startDate, endDate):
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
        ydf.loc[:, "forecast"] = self.predict(xdf)
        # 计算评估指标
        self.evaluator.eval(ydf)


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
    engine = MeowEngine(h5dir="archive", cacheDir=None)
    
    # 训练模型（使用6月至11月数据）
    engine.fit(20230601, 20231130)
    
    # 评估模型（使用12月数据）
    engine.eval(20231201, 20231229)
