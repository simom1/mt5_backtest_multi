"""
MT5 多周期回测系统

提供多时间框架回测功能，支持多种交易策略和参数优化。

主要功能:
- 多时间框架分析 (M1, M5, M15, M30, H1)
- 图形化界面和命令行界面
- 参数优化和网格搜索
- 详细的回测结果和可视化
"""

__version__ = "0.1.0"

# 暴露主要接口
try:
    from .core.main import cli
    from .gui.gui_app import run_gui
    __all__ = ['cli', 'run_gui']
except ImportError:
    # 在安装过程中可能会缺少依赖
    __all__ = []
