from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseStrategy(ABC):
    """策略基类，约定策略应实现的接口。"""

    def __init__(self):
        self.trade_records = []

    @abstractmethod
    def preprocess_hourly_signals(self, df_1h: pd.DataFrame) -> None:
        """预处理小时级数据（如生成信号缓存）。"""
        raise NotImplementedError

    @abstractmethod
    def run_backtest(self, df_1m: pd.DataFrame) -> None:
        """在 1 分钟级数据上执行回测。"""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """返回统计指标。"""
        raise NotImplementedError

    def save_trade_records(self, filename: str = 'trade_records.csv') -> None:
        """可由具体策略实现；如果基类实现需要用到 DataFrame，可在子类中覆盖。"""
        raise NotImplementedError
