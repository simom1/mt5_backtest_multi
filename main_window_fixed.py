"""Main Window and worker implementations for MT5 backtesting GUI."""

from typing import Dict, Any, List
import json
import random
import traceback
import os
import threading
from decimal import Decimal, getcontext

import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

# 使用绝对导入避免相对导入问题
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DEFAULT_SYMBOL, DEFAULT_H1_BARS, DEFAULT_M1_BARS,
    DEFAULT_STOP_LOSS_USD, DEFAULT_TAKE_PROFIT_USD, DEFAULT_MIN_RANGE_USD,
    DEFAULT_QTY, DEFAULT_LOT_MODE,
)
from strategies.pinbar_configurable import ConfigurableTimeframePinbarStrategy
from strategies.timed_entry import TimedEntryStrategy5m
from ui_components import MplCanvas, GlassCard, ModernButton, AnimatedProgressBar, KPICard, ModernInput
from utils.plotting import plot_pnl_chart

# Note: Do NOT import gui_app here to avoid circular imports.


# Local Worker implementations
class WorkerSignals(QtCore.QObject):
    finished = QtCore.Signal(object)  # payload
    error = QtCore.Signal(str)
    progress = QtCore.Signal(str)


class RunBacktestWorker(QtCore.QRunnable):
    def __init__(self, symbol: str, h1_bars: int, m1_bars: int, strategy_cls, strategy_params: Dict[str, Any], reuse_db: bool = True):
        super().__init__()
        self.symbol = symbol
        self.h1_bars = h1_bars
        self.m1_bars = m1_bars
        self.strategy_cls = strategy_cls
        self.strategy_params = strategy_params
        self.reuse_db = reuse_db
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.progress.emit("正在初始化MT5...")
            from engines.backtester import run_backtest_pipeline
            
            self.signals.progress.emit("正在运行回测...")
            trades_df, stats, strategy = run_backtest_pipeline(
                symbol=self.symbol,
                h1_bars=self.h1_bars,
                m1_bars=self.m1_bars,
                strategy_cls=self.strategy_cls,
                strategy_params=self.strategy_params,
                reuse_db=self.reuse_db,
            )
            
            self.signals.progress.emit("回测完成")
            self.signals.finished.emit({
                'trades_df': trades_df,
                'stats': stats,
                'strategy': strategy,
            })
            
        except Exception as e:
            error_msg = f"回测失败: {str(e)}"
            self.signals.error.emit(error_msg)
            traceback.print_exc()


class MainWindow(QtWidgets.QMainWindow):
    """主窗口类 - 用于测试"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MT5多周期回测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # 添加标题
        title_label = QtWidgets.QLabel("MT5多周期回测系统 - 测试版本")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white; margin: 20px;")
        layout.addWidget(title_label)
        
        # 添加说明
        info_label = QtWidgets.QLabel("GUI测试成功！所有组件正常工作。")
        info_label.setStyleSheet("font-size: 16px; color: #4A9EFF; margin: 10px;")
        layout.addWidget(info_label)
        
        # 添加测试按钮
        test_button = ModernButton("测试按钮", "primary")
        test_button.clicked.connect(self.on_test_click)
        layout.addWidget(test_button)
        
        # 添加状态显示
        self.status_label = QtWidgets.QLabel("状态: 就绪")
        self.status_label.setStyleSheet("font-size: 14px; color: #52C41A; margin: 10px;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
    
    def on_test_click(self):
        self.status_label.setText("状态: 按钮点击测试成功！")
        self.status_label.setStyleSheet("font-size: 14px; color: #FA8C16; margin: 10px;")


def main():
    """主函数 - 用于测试"""
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # 设置暗色主题
    app.setStyleSheet("""
        QMainWindow {
            background-color: #1e1e1e;
            color: white;
        }
        QWidget {
            background-color: #1e1e1e;
            color: white;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    main()
