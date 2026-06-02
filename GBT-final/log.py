"""
日志模块 (log.py)

功能说明：
本模块提供统一的日志记录功能，用于在项目运行过程中输出带时间戳、文件名和行号的日志信息。
支持多种颜色输出，便于区分不同级别的信息。

主要功能：
1. 带时间戳和位置信息的日志输出
2. 多种颜色支持（红、绿、黄、蓝等）
3. 支持输出到文件或控制台
4. 进程ID显示（可选）

使用示例：
    from log import log
    log.inf("这是一条普通信息")
    log.red("这是一条错误信息")
    log.green("这是一条成功信息")
"""

import os
from datetime import datetime
from inspect import currentframe, getframeinfo


class MeowLogger(object):
    """
    日志记录器类
    
    负责统一管理项目中的日志输出，支持控制台输出和文件输出。
    日志格式：[时间|文件名:行号|进程ID] 消息内容
    """
    
    def __init__(self):
        """初始化日志记录器，默认不输出到文件"""
        self.logf = None

    def __del__(self):
        """析构函数，确保文件句柄正确关闭"""
        if self.logf is not None:
            self.logf.close()

    def __header(self, pid):
        """
        生成日志头部信息（私有方法）
        
        参数：
            pid: 是否显示进程ID
            
        返回：
            格式化的日志头部字符串，包含时间、文件名、行号和可选的进程ID
            
        示例输出：
            [2026-05-27T20:30:47.587327|meow.py:27|12345]
        """
        now = datetime.now()
        # 获取调用者的文件名和行号（向上回溯两层调用栈）
        frameInfo = getframeinfo(currentframe().f_back.f_back)
        if pid:
            # 包含进程ID的格式
            return "[\033[90m{}|\033[0m{}:{}|{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno, os.getpid())
        # 不包含进程ID的格式
        return "[\033[90m{}|\033[0m{}:{}] ".format(now.strftime("%Y-%m-%dT%H:%M:%S.%f"), os.path.basename(frameInfo.filename), frameInfo.lineno)

    def setLogFile(self, filename):
        """
        设置日志输出文件
        
        参数：
            filename: 日志文件路径
            
        说明：
            如果已有日志文件在写入，会先关闭旧文件再打开新文件
        """
        if self.logf is not None:
            self.logf.close()
        self.logf = open(filename, "w")

    def log(self, content, muted=False):
        """
        核心日志输出方法
        
        参数：
            content: 日志内容
            muted: 是否静默（True则不输出）
            
        说明：
            如果设置了日志文件，则写入文件；否则输出到控制台
        """
        if muted:
            return
        if self.logf is not None:
            # 写入文件并立即刷新缓冲区
            self.logf.write(content + "\n")
            self.logf.flush()
            return
        # 输出到控制台
        print(content)

    def inf(self, line, pid=False, muted=False):
        """
        输出普通信息日志（白色）
        
        参数：
            line: 日志消息
            pid: 是否显示进程ID
            muted: 是否静默
        """
        self.log(self.__header(pid) + line, muted)

    def grey(self, line, pid=False, muted=False):
        """输出灰色日志（用于次要信息）"""
        self.log("{}\033[90m{}\033[0m".format(self.__header(pid), line), muted)

    def red(self, line, pid=False, muted=False):
        """输出红色日志（用于错误和警告）"""
        self.log("{}\033[91m{}\033[0m".format(self.__header(pid), line), muted)

    def green(self, line, pid=False, muted=False):
        """输出绿色日志（用于成功信息）"""
        self.log("{}\033[92m{}\033[0m".format(self.__header(pid), line), muted)

    def yellow(self, line, pid=False, muted=False):
        """输出黄色日志（用于提示信息）"""
        self.log("{}\033[93m{}\033[0m".format(self.__header(pid), line), muted)

    def blue(self, line, pid=False, muted=False):
        """输出蓝色日志（用于调试信息）"""
        self.log("{}\033[94m{}\033[0m".format(self.__header(pid), line), muted)

    def pink(self, line, pid=False, muted=False):
        """输出粉色日志（用于特殊标记）"""
        self.log("{}\033[95m{}\033[0m".format(self.__header(pid), line), muted)

    def cyan(self, line, pid=False, muted=False):
        """输出青色日志（用于信息提示）"""
        self.log("{}\033[96m{}\033[0m".format(self.__header(pid), line), muted)


# 创建全局日志对象，供整个项目使用
log = MeowLogger()
