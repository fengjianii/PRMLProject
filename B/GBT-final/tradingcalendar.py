"""
交易日历模块 (tradingcalendar.py)

功能说明：
本模块提供交易日历相关功能，用于判断和管理交易日。
从资源文件中加载交易日历，提供日期查询、转换等功能。

主要功能：
1. 判断某日期是否为交易日
2. 查找下一个/上一个交易日
3. 获取指定范围的交易日列表
4. 日期偏移计算

数据来源：
    resources/calendar 文件，每行一个交易日（格式：YYYYMMDD）
"""

import os
import bisect
from log import log


class Calendar(object):
    """
    交易日历类
    
    管理交易日历数据，提供交易日相关的查询和计算功能。
    使用bisect模块实现高效的日期查找。
    """
    
    def __init__(self):
        """
        初始化交易日历，从文件加载数据
        
        说明：
            从 resources/calendar 文件读取交易日列表
            文件格式：每行一个日期，格式为YYYYMMDD的整数
        """
        calendarFile = os.path.join(os.path.dirname(__file__), "resources/calendar")
        with open(calendarFile) as f:
            tokens = f.read().splitlines()
            # 排序后的交易日列表，用于二分查找
            self.tradingDays = sorted([int(x) for x in tokens])
            # 交易日集合，用于快速判断
            self.tradingDaySet = set(self.tradingDays)

    def isTradingDay(self, date):
        """
        判断某日期是否为交易日
        
        参数：
            date: 日期（整数或可转换为整数的类型）
            
        返回：
            True: 是交易日
            False: 不是交易日
        """
        if not isinstance(date, int):
            date = int(date)
        return date in self.tradingDaySet

    def toTradingDay(self, date):
        """
        将日期转换为最近的不早于该日期的交易日
        
        参数：
            date: 日期
            
        返回：
            最近的不早于输入日期的交易日
            
        示例：
            toTradingDay(20230101) -> 20230103 (假设1月1日、2日非交易日)
        """
        if not isinstance(date, int):
            date = int(date)
        # 使用二分查找找到第一个>=date的交易日
        index = bisect.bisect_left(self.tradingDays, date)
        return self.tradingDays[index]

    def next(self, date):
        """
        获取下一个交易日
        
        参数：
            date: 当前日期
            
        返回：
            下一个交易日，如果已到最后则返回None
        """
        if not isinstance(date, int):
            date = int(date)
        # 找到第一个>date的交易日
        index = bisect.bisect_right(self.tradingDays, date)
        if index >= len(self.tradingDays):
            return None
        return self.tradingDays[index]

    def prev(self, date):
        """
        获取上一个交易日
        
        参数：
            date: 当前日期
            
        返回：
            上一个交易日，如果已在最前则返回None
        """
        if not isinstance(date, int):
            date = int(date)
        # 找到第一个<date的交易日
        index = bisect.bisect_left(self.tradingDays, date)
        if index == 0:
            return None
        return self.tradingDays[index - 1]

    def shift(self, date, n):
        """
        从指定日期偏移n个交易日
        
        参数：
            date: 起始日期
            n: 偏移量（可正可负）
            
        返回：
            偏移后的交易日
            
        示例：
            shift(20230103, 5) -> 从20230103往后数第5个交易日
        """
        if not isinstance(date, int):
            date = int(date)
        if not isinstance(n, int):
            log.red("Invalid shift n: {}".format(n))
            return None

        index = bisect.bisect_left(self.tradingDays, date)
        if index == 0:
            log.red("Failed to shift for date {}, n={}".format(date, n))
            return None
        return self.tradingDays[index + n]

    def prevn(self, date, n):
        """
        获取指定日期之前的n个交易日
        
        参数：
            date: 日期
            n: 数量
            
        返回：
            交易日列表（不包含date本身），按时间升序排列
            
        示例：
            prevn(20230110, 3) -> [20230105, 20230106, 20230109]
        """
        if not isinstance(date, int):
            date = int(date)
        if not isinstance(n, int) or n < 1:
            log.red("Invalid prevn: date={},n={}".format(date, n))
            return None

        index = bisect.bisect_left(self.tradingDays, date)
        if index == 0:
            log.red("Failed to find prev trading day for date {}".format(date))
            return None
        if index < n:
            log.yellow("Not enough days for prevn: date={},n={},index={}".format(date, n, index))

        # 返回前n个交易日（不包含当前日期）
        return self.tradingDays[max(index - n, 0) : index]

    def nextn(self, date, n):
        """
        获取指定日期之后的n个交易日
        
        参数：
            date: 日期
            n: 数量
            
        返回：
            交易日列表（不包含date本身），按时间升序排列
            
        示例：
            nextn(20230110, 3) -> [20230111, 20230112, 20230113]
        """
        if not isinstance(date, int):
            date = int(date)
        if not isinstance(n, int) or n < 1:
            log.red("Invalid nextn: date={},n={}".format(date, n))
            return None

        index = bisect.bisect_right(self.tradingDays, date)
        if index >= len(self.tradingDays):
            log.red("Failed to find next trading day for date {}".format(date))
            return None
        if index + n > len(self.tradingDays):
            log.yellow("Not enough days for next: date={},n={},index={}".format(date, n, index))

        # 返回后n个交易日（不包含当前日期）
        return self.tradingDays[index: min(index + n, len(self.tradingDays))]

    def range(self, startDate, endDate):
        """
        获取指定日期范围内的所有交易日
        
        参数：
            startDate: 起始日期（包含）
            endDate: 结束日期（包含）
            
        返回：
            交易日列表，按时间升序排列
            
        示例：
            range(20230601, 20230610) -> [20230601, 20230602, ..., 20230609]
            
        注意：
            如果startDate > endDate，返回None并输出错误信息
        """
        if not isinstance(startDate, int):
            startDate = int(startDate)
        if not isinstance(endDate, int):
            endDate = int(endDate)
        if startDate > endDate:
            log.red("Invalid range - startDate is larger than endDate: startDate={},endDate={}".format(startDate, endDate))
            return None

        # 找到>=startDate的第一个交易日
        startIndex = bisect.bisect_left(self.tradingDays, startDate)
        if (startIndex == len(self.tradingDays)):
            log.red("No valid trading days found within the range [{}, {})".format(startDate, endDate))
            return None

        # 找到<=endDate的最后一个交易日（用bisect_right再取前一个）
        endIndex = bisect.bisect_right(self.tradingDays, endDate)
        # 返回[startIndex, endIndex)范围内的交易日
        return self.tradingDays[startIndex : endIndex]
