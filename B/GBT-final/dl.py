"""
数据加载模块 (dl.py)

功能说明：
本模块负责从H5文件中加载股票交易数据。
每个交易日对应一个独立的H5文件，包含当日所有股票的分钟级数据。

主要功能：
1. 加载单个交易日的数据
2. 加载多个交易日的数据并拼接
3. 验证交易日有效性
4. 数据列重排列

数据格式：
    输入：H5文件（Pandas HDFStore格式）
    输出：DataFrame，包含股票代码、时间、价格、订单簿等字段
    
使用示例：
    loader = MeowDataLoader(h5dir="archive")
    data = loader.loadDates([20230601, 20230602])
"""

import os
import pandas as pd
from tradingcalendar import Calendar
from log import log


class MeowDataLoader(object):
    """
    数据加载器类
    
    负责从H5文件加载股票交易数据，提供单日和多日数据加载功能。
    """
    
    def __init__(self, h5dir):
        """
        初始化数据加载器
        
        参数：
            h5dir: H5文件所在目录路径
            
        说明：
            同时初始化交易日历对象，用于验证日期有效性
        """
        self.h5dir = h5dir
        self.calendar = Calendar()

    def loadDates(self, dates):
        """
        加载多个交易日的数据
        
        参数：
            dates: 交易日列表，如 [20230601, 20230602, 20230605]
            
        返回：
            DataFrame，包含所有指定日期的数据，按日期拼接
            
        说明：
            使用pd.concat拼接多个DataFrame，效率较高
            如果dates为空，抛出ValueError异常
            
        示例：
            data = loader.loadDates([20230601, 20230602])
            # 返回约480万条记录（假设每天约240分钟，300只股票）
        """
        if len(dates) == 0:
            raise ValueError("Dates empty")
        log.inf("Loading data of {} dates from {} to {}...".format(len(dates), min(dates), max(dates)))
        # 逐日加载并拼接
        return pd.concat(self.loadDate(x) for x in dates)

    def loadDate(self, date):
        """
        加载单个交易日的数据
        
        参数：
            date: 交易日，格式为YYYYMMDD的整数
            
        返回：
            DataFrame，包含该日所有股票的分钟级数据
            
        数据字段（部分）：
            - symbol: 股票代码
            - interval: 时间（毫秒）
            - date: 日期
            - midpx: 中间价
            - fret12: 未来12分钟收益率（预测目标）
            - bid0~bid4: 买一至买五价格
            - ask0~ask4: 卖一至卖五价格
            - bsize0~bsize4: 买一至买五数量
            - asize0~asize4: 卖一至卖五数量
            - tradeBuyQty/tradeSellQty: 主买/主卖数量
            
        说明：
            如果date不是交易日，抛出ValueError异常
            加载后自动添加date列，并重排列使主键在前
        """
        if not self.calendar.isTradingDay(date):
            raise ValueError("Not a trading day: {}".format(date))
        # 构建H5文件路径
        h5File = os.path.join(self.h5dir, "{}.h5".format(date))
        # 读取H5文件
        df = pd.read_hdf(h5File)
        # 添加日期列
        df.loc[:, "date"] = date
        # 重排列顺序，主键列在前
        precols = ["symbol", "interval", "date"]
        df = df[precols + [x for x in df.columns if x not in precols]]
        return df
