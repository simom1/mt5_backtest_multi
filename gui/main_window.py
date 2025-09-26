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

import sys
import os

# Add project root to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now use absolute imports
try:
    from config import (
        DEFAULT_SYMBOL, DEFAULT_H1_BARS, DEFAULT_M1_BARS,
        DEFAULT_STOP_LOSS_USD, DEFAULT_TAKE_PROFIT_USD, DEFAULT_MIN_RANGE_USD,
        DEFAULT_QTY, DEFAULT_LOT_MODE,
    )
    from strategies.pinbar_configurable import ConfigurableTimeframePinbarStrategy
    from strategies.timed_entry import TimedEntryStrategy5m
    from gui.ui_components import MplCanvas, GlassCard, ModernButton, AnimatedProgressBar, KPICard, ModernInput
    from utils.plotting import plot_pnl_chart
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current sys.path: {sys.path}")
    raise

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

    @QtCore.Slot()
    def run(self):
        try:
            # Progress diagnostics
            try:
                self.signals.progress.emit("[RUN] 初始化回测环境…")
            except Exception:
                pass
            # Add project root to path if not already there
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from engines.backtester import run_backtest_pipeline
            try:
                self.signals.progress.emit("[RUN] 启动回测管线…")
            except Exception:
                pass
            trades_df, stats, strategy = run_backtest_pipeline(
                symbol=self.symbol,
                h1_bars=self.h1_bars,
                m1_bars=self.m1_bars,
                strategy_params=self.strategy_params,
                strategy_cls=self.strategy_cls,
                reuse_db=self.reuse_db,
            )
            try:
                self.signals.progress.emit("[RUN] 回测完成，正在返回结果…")
            except Exception:
                pass
            self.signals.finished.emit({
                "trades_df": trades_df,
                "stats": stats,
                "strategy": strategy,
            })
        except Exception as e:
            tb = traceback.format_exc()
            self.signals.error.emit(f"运行回测失败: {e}\n{tb}")


class SaveTuningWorker(QtCore.QRunnable):
    """Background saver for tuning results to avoid blocking UI.
    
    This worker saves results in chunks to prevent UI freezing and provides
    progress updates during the save process.
    """
    def __init__(self, results: List[Dict[str, Any]], *, symbol: str, strategy_cls, objective: str, chunk_size: int = 10):
        super().__init__()
        self.results = results
        self.symbol = symbol
        self.strategy_cls = strategy_cls
        self.objective = objective
        self.chunk_size = chunk_size
        self.signals = WorkerSignals()
        self._stop_requested = False

    def stop(self):
        """Request the worker to stop at the next opportunity."""
        self._stop_requested = True

    @QtCore.Slot()
    def run(self):
        """Save results in chunks to avoid blocking the UI."""
        from utils import db as dbutil
        
        total = len(self.results)
        saved_count = 0
        last_error = None
        
        try:
            # Process results in chunks
            for i in range(0, total, self.chunk_size):
                if self._stop_requested:
                    self.signals.progress.emit(f"保存已取消，已保存 {saved_count}/{total} 条记录")
                    return
                    
                chunk = self.results[i:i + self.chunk_size]
                try:
                    # Save current chunk
                    dbutil.save_tuning_results(
                        results=chunk,
                        symbol=self.symbol,
                        strategy_name=getattr(self.strategy_cls, '__name__', str(self.strategy_cls)),
                        objective=self.objective,
                        code_version=None,
                    )
                    saved_count += len(chunk)
                    
                    # Emit progress
                    progress = (i + len(chunk)) / total * 100
                    self.signals.progress.emit(
                        f"保存进度: {saved_count}/{total} "
                        f"({progress:.1f}%)"
                    )
                    
                    # Small delay to keep UI responsive
                    time.sleep(0.01)
                    
                except Exception as e:
                    last_error = str(e)
                    self.signals.error.emit(f"保存部分结果时出错: {e}")
                    # Continue with next chunk even if one fails
            
            # Emit final status
            if last_error:
                self.signals.finished.emit({
                    "ok": False,
                    "saved_count": saved_count,
                    "total": total,
                    "last_error": last_error
                })
            else:
                self.signals.finished.emit({
                    "ok": True,
                    "saved_count": saved_count,
                    "total": total
                })
                
        except Exception as e:
            self.signals.error.emit(f"保存结果时发生错误: {e}")
            self.signals.finished.emit({
                "ok": False,
                "saved_count": saved_count,
                "total": total,
                "last_error": str(e)
            })


class TuneWorker(QtCore.QRunnable):
    def __init__(self, symbol: str, h1_bars: int, m1_bars: int,
                 sl_range: tuple[float, float, float],
                 tp_range: tuple[float, float, float],
                 mr_range: tuple[float, float, float],
                 qty_range: tuple[int, int, int],
                 lot_modes: List[str],
                 max_workers: int,
                 objective: str,
                 search_mode: str = "Grid",
                 random_iter: int = 50,
                 strategy_cls=None,
                 strategy_kwargs: Dict[str, Any] | None = None,
                 entry_min_range: tuple[int, int, int] | None = None,
                 max_positions_range: tuple[int, int, int] | None = None,
                 enable_time_filter_values: List[bool] | None = None,
                 start_hour_range: tuple[int, int, int] | None = None,
                 end_hour_range: tuple[int, int, int] | None = None,
                 trade_direction_values: List[str] | None = None,
                 save_callback=None):
        super().__init__()
        self.symbol = symbol
        self.h1_bars = h1_bars
        self.m1_bars = m1_bars
        self.sl_range = sl_range
        self.tp_range = tp_range
        self.mr_range = mr_range
        self.qty_range = qty_range
        self.lot_modes = lot_modes
        self.max_workers = max_workers
        self.objective = objective
        self.search_mode = search_mode
        self.random_iter = int(random_iter)
        self.strategy_cls = strategy_cls
        self.strategy_kwargs = strategy_kwargs or {}
        # Timed-entry tuning attributes
        self.entry_min_range = entry_min_range
        self.max_positions_range = max_positions_range
        self.enable_time_filter_values = enable_time_filter_values or []
        self.start_hour_range = start_hour_range
        self.end_hour_range = end_hour_range
        self.trade_direction_values = trade_direction_values or []
        # Callback for saving results
        self.save_callback = save_callback
        # Stop support
        self.stop_event = threading.Event()
        self.signals = WorkerSignals()
        # Track progress
        self._completed = 0
        self._total = 0

    @QtCore.Slot()
    def run(self):
        try:
            import time
            start_time = time.time()
            self.signals.progress.emit(f"🔍 开始参数优化任务 (工作线程数: {self.max_workers})...")
            
            # Log parameter ranges for debugging
            param_info = [
                f"止损: {self.sl_range[0]}-{self.sl_range[1]} step {self.sl_range[2]}",
                f"止盈: {self.tp_range[0]}-{self.tp_range[1]} step {self.tp_range[2]}",
                f"最小波动: {self.mr_range[0]}-{self.mr_range[1]} step {self.mr_range[2]}",
                f"手数: {self.qty_range[0]}-{self.qty_range[1]} step {self.qty_range[2]}",
                f"手数模式: {self.lot_modes}"
            ]
            self.signals.progress.emit("📊 参数范围: " + ", ".join(param_info))
            
            # Add project root to path if not already there
            import sys
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from utils.tools import initialize_mt5, load_data, preprocess_data
            from core.tuner import tune_parameters

            # Initialize MT5 with timeout
            mt5_start = time.time()
            self.signals.progress.emit("🔄 初始化MT5...")
            if not initialize_mt5():
                self.signals.error.emit("❌ MT5 初始化失败")
                return
            self.signals.progress.emit(f"✅ MT5 初始化完成 (耗时: {time.time() - mt5_start:.1f}秒)")

            # Load data with progress updates
            data_start = time.time()
            self.signals.progress.emit(f"📥 加载数据: {self.symbol} (H1:{self.h1_bars}条, M1:{self.m1_bars}条)...")
            df_1h, df_30m, df_15m, df_5m, df_1m = load_data(
                self.symbol, 
                h1_bars=self.h1_bars, 
                m1_bars=self.m1_bars
            )
            
            if any(x is None for x in [df_1h, df_30m, df_15m, df_5m, df_1m]):
                self.signals.error.emit("❌ 数据加载失败")
                return
                
            data_load_time = time.time() - data_start
            self.signals.progress.emit(
                f"✅ 数据加载完成: "
                f"H1={len(df_1h)}, M1={len(df_1m)} "
                f"(耗时: {data_load_time:.1f}秒)"
            )
                
            self.signals.progress.emit(f"✅ 数据加载完成: H1={len(df_1h)}, M1={len(df_1m)}")
            # Preprocess data with timing
            preprocess_start = time.time()
            self.signals.progress.emit("🔄 数据预处理中...")
            try:
                df_1h, df_30m, df_15m, df_5m, df_1m = preprocess_data(df_1h, df_30m, df_15m, df_5m, df_1m)
                if any(x is None for x in [df_1h, df_30m, df_15m, df_5m, df_1m]):
                    self.signals.error.emit("❌ 数据预处理失败")
                    return
                preprocess_time = time.time() - preprocess_start
                self.signals.progress.emit(f"✅ 数据预处理完成 (耗时: {preprocess_time:.1f}秒)")
            except Exception as e:
                self.signals.error.emit(f"❌ 数据预处理异常: {str(e)}")
                import traceback
                self.signals.error.emit(traceback.format_exc())
                return

            # Build value ranges (coarse progress: start/end only)
            def frange(a: float, b: float, s: float):
                """Precise float range using Decimal to avoid步长漂移.
                包含端点，按 step 整数倍生成：a, a+step, ..., <= b。
                """
                try:
                    getcontext().prec = 28
                    da = Decimal(str(a))
                    db = Decimal(str(b))
                    ds = Decimal(str(s)) if s is not None else Decimal('0')
                    if ds <= 0:
                        return [float(da)]
                    vals: List[float] = []
                    n = 0
                    cur = da
                    # 扩一点上界，避免精度问题
                    while cur <= db + Decimal('1e-12'):
                        vals.append(float(cur))
                        n += 1
                        cur = da + ds * n
                    return [round(v, 6) for v in vals]
                except Exception:
                    # 回退到原实现
                    vals = []
                    x = a
                    while x <= b + 1e-9:
                        vals.append(round(x, 6))
                        x += s
                    return vals

            def irange(a: int, b: int, s: int):
                vals = []
                x = a
                while x <= b:
                    vals.append(int(x))
                    x += s
                return vals

            sl_vals = frange(self.sl_range[0], self.sl_range[1], max(0.0001, self.sl_range[2]))
            tp_vals = frange(self.tp_range[0], self.tp_range[1], max(0.0001, self.tp_range[2]))
            mr_vals = frange(self.mr_range[0], self.mr_range[1], max(0.0001, self.mr_range[2]))
            qty_vals = [max(10, q) for q in irange(self.qty_range[0], self.qty_range[1], max(1, self.qty_range[2]))]
            lot_modes = self.lot_modes or ['fixed']
            # Timed-entry grids (optional)
            entry_min_vals = None
            max_pos_vals = None
            if getattr(self, 'entry_min_range', None) is not None:
                a, b, s = self.entry_min_range
                s = max(1, int(s))
                entry_min_vals = [x for x in range(int(a), int(b)+1, s) if 0 <= x <= 59]
            if getattr(self, 'max_positions_range', None) is not None:
                a, b, s = self.max_positions_range
                s = max(1, int(s))
                max_pos_vals = [x for x in range(int(a), int(b)+1, s) if x >= 1]
            # Timed-entry extra grids
            etf_vals = None
            if self.enable_time_filter_values:
                etf_vals = [bool(v) for v in self.enable_time_filter_values]
            sh_vals = None
            if self.start_hour_range is not None:
                a, b, s = self.start_hour_range; s = max(1, int(s))
                sh_vals = [x for x in range(int(a), int(b)+1, s) if 0 <= x <= 23]
            eh_vals = None
            if self.end_hour_range is not None:
                a, b, s = self.end_hour_range; s = max(1, int(s))
                eh_vals = [x for x in range(int(a), int(b)+1, s) if 0 <= x <= 23]
            td_vals = None
            if self.trade_direction_values:
                td_vals = [str(v) for v in self.trade_direction_values]

            # Calculate total combinations for progress tracking
            total_combinations = len(sl_vals) * len(tp_vals) * len(mr_vals) * len(qty_vals) * len(lot_modes)
            if entry_min_vals: total_combinations *= len(entry_min_vals)
            if max_pos_vals: total_combinations *= len(max_pos_vals)
            if etf_vals: total_combinations *= len(etf_vals)
            if sh_vals: total_combinations *= len(sh_vals)
            if eh_vals: total_combinations *= len(eh_vals)
            if td_vals: total_combinations *= len(td_vals)
            
            self.signals.progress.emit(
                f"🚀 开始参数优化 (共 {total_combinations:,} 种组合, 使用 {self.max_workers} 个工作线程)..."
            )
            
            # Progress callback for tune_parameters
            _last_update_time = 0
            _last_progress = 0
            
            def progress_callback(current, total, **kwargs):
                nonlocal _last_update_time, _last_progress
                
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Only update at most once per second or when progress changes significantly
                if (current_time - _last_update_time > 1.0 or 
                    current == total or 
                    current - _last_progress >= 5):
                    
                    pct = (current / total) * 100
                    
                    # Calculate ETA more smoothly
                    if current > 0 and elapsed > 0:
                        speed = current / elapsed  # combinations per second
                        if speed > 0:
                            eta = (total - current) / speed
                        else:
                            eta = 0
                    else:
                        eta = 0
                    
                    # Format time strings with leading zeros
                    def format_time(seconds):
                        mins = int(seconds // 60)
                        secs = int(seconds % 60)
                        return f"{mins:02d}分{secs:02d}秒"
                    
                    # Update progress bar (0-100)
                    self.signals.progress.emit(str(int(pct)))
                    
                    # Log detailed progress
                    progress_msg = (
                        f"⏳ 优化进度: {current}/{total} ({pct:.1f}%) | "
                        f"已用: {format_time(elapsed)} | "
                        f"预计剩余: {format_time(eta)}"
                    )
                    if 'current_params' in kwargs:
                        progress_msg += f"\n  当前参数: {kwargs['current_params']}"
                    
                    self.signals.progress.emit(progress_msg)
                    
                    # Update last update time and progress
                    _last_update_time = current_time
                    _last_progress = current
            
            # Run tuning with progress updates
            tuning_start = time.time()
            results = tune_parameters(
                sl_vals, tp_vals, mr_vals,
                df_1h=df_1h, df_1m=df_1m,
                qty_values=qty_vals,
                lot_mode_values=lot_modes,
                max_workers=self.max_workers,
                stop_event=self.stop_event,
                strategy_cls=self.strategy_cls,
                strategy_kwargs=self.strategy_kwargs,
                entry_minute_values=entry_min_vals,
                max_positions_values=max_pos_vals,
                enable_time_filter_values=etf_vals,
                start_hour_values=sh_vals,
                end_hour_values=eh_vals,
                trade_direction_values=td_vals,
                progress_callback=progress_callback
            )
            
            tuning_time = time.time() - tuning_start
            total_time = time.time() - start_time
            
            # Save results to database in a background worker
            if results:
                self.signals.progress.emit("💾 正在保存优化结果到数据库...")
                save_worker = SaveTuningWorker(
                    results=results,
                    symbol=self.symbol,
                    strategy_cls=self.strategy_cls,
                    objective=self.objective
                )
                save_worker.signals.finished.connect(
                    lambda x: self.signals.progress.emit(
                        f"✅ 参数优化完成！\n"
                        f"• 总耗时: {int(total_time//60)}分{int(total_time%60)}秒\n"
                        f"• 平均每组合: {tuning_time/max(1, len(results)):.2f}秒\n"
                        f"• 测试组合数: {len(results)}\n"
                        f"• 成功保存 {x.get('count', 0)} 条结果到数据库"
                    )
                )
                save_worker.signals.error.connect(
                    lambda e: self.signals.error.emit(f"❌ 保存优化结果到数据库失败: {e}")
                )
                # Use the thread pool to run the save operation
                QtCore.QThreadPool.globalInstance().start(save_worker)
            else:
                self.signals.progress.emit(
                    f"✅ 参数优化完成！\n"
                    f"• 总耗时: {int(total_time//60)}分{int(total_time%60)}秒\n"
                    f"• 平均每组合: {tuning_time/max(1, len(results)):.2f}秒\n"
                    f"• 测试组合数: {len(results)}"
                )
        except ImportError as e:
            error_msg = [
                "❌ 导入模块失败",
                f"错误: {str(e)}",
                "",
                "Python路径:",
                *[f"  - {p}" for p in sys.path],
                "",
                f"当前工作目录: {os.getcwd()}",
                "",
                "请确保所有依赖项已正确安装，并且项目结构完整。"
            ]
            self.signals.error.emit("\n".join(error_msg))
        except Exception as e:
            error_msg = [
                "❌ 调参过程中发生错误",
                f"错误类型: {e.__class__.__name__}",
                f"错误信息: {str(e)}",
                "",
                "错误详情:",
                traceback.format_exc()
            ]
            self.signals.error.emit("\n".join(error_msg))


# Export STRATEGY_REGISTRY here for convenience (remove default Pinbar strategy)
import sys
import os

# Add project root to path if not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from strategies.pinbar_configurable import ConfigurableTimeframePinbarStrategy
    from strategies.timed_entry import TimedEntryStrategy5m
    
    # Create strategy registry with display names
    STRATEGY_REGISTRY = {
        "Pinbar (可配置级别)": ConfigurableTimeframePinbarStrategy,
        "定时做单（5m）": TimedEntryStrategy5m,
    }
    print(f"✅ 已加载策略: {list(STRATEGY_REGISTRY.keys())}")
except Exception as e:
    # If strategies not available at import time, provide empty registry
    print(f"❌ 加载策略时出错: {e}")
    import traceback
    traceback.print_exc()
    STRATEGY_REGISTRY = {}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MT5 Premium Backtesting Suite")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        self.thread_pool = QtCore.QThreadPool.globalInstance()
        
        # Initialize status bar before setting up UI
        self.status = self.statusBar()
        self.status.setStyleSheet("QStatusBar{background:#1e1e1e;color:#fff;border-top:1px solid rgba(255,255,255,0.1);padding:4px;}")
        
        self._last_tuning_results: List[Dict[str, Any]] = []
        self._tuning_running: bool = False
        
        # Setup UI after status bar is initialized
        self._setup_ui()
        self._apply_modern_theme()
        
    def _setup_ui(self):
        """Initialize the main UI components."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Navbar
        navbar = QtWidgets.QFrame()
        navbar.setFixedHeight(110)
        navbar.setStyleSheet(
            "QFrame{background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #121212, stop:1 #1f1f22);"
            "border-bottom:1px solid rgba(255,255,255,0.12);} "
        )
        nlay = QtWidgets.QHBoxLayout(navbar)
        nlay.setContentsMargins(48, 20, 48, 20)
        nlay.setSpacing(24)
        
        title = QtWidgets.QLabel("MT5 Premium Suite")
        title.setStyleSheet("color:#4A9EFF;font-size:28px;font-weight:800;")
        
        subtitle = QtWidgets.QLabel("Professional Trading Analysis & Backtesting Platform")
        subtitle.setStyleSheet("color:#b8b8b8;font-size:13px;")
        
        vtitle = QtWidgets.QVBoxLayout()
        vtitle.addWidget(title)
        vtitle.addWidget(subtitle)
        
        nlay.addWidget(QtWidgets.QLabel("🏛️"))
        nlay.addLayout(vtitle)
        nlay.addStretch()
        
        btn_outputs = ModernButton("📁 输出目录", "ghost")
        btn_outputs.clicked.connect(lambda: self._open_dir("outputs"))
        
        btn_refresh = ModernButton("🔄 刷新", "ghost")
        btn_refresh.clicked.connect(self._refresh_data)
        
        nlay.addWidget(btn_outputs)
        nlay.addWidget(btn_refresh)
        root.addWidget(navbar)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(
            "QTabWidget::pane{border:none;background:#121212;}"
            " QTabBar::tab{background:rgba(45,45,48,0.85);color:#fff;padding:12px 26px;margin-right:4px;"
            "border-top-left-radius:10px;border-top-right-radius:10px;}"
            " QTabBar::tab:selected{background:#4A9EFF;color:#fff;}"
            " QTabBar::tab:hover:!selected{background:rgba(74,158,255,0.28);}"
        )
        
        # Create tabs
        self.run_tab = self._create_run_tab()
        self.tune_tab = self._create_tune_tab()
        self.results_tab = self._create_results_tab()
        self.history_tab = self._create_history_tab()
        
        # Add tabs to the tab widget
        self.tabs.addTab(self.run_tab, "🚀 策略回测")
        self.tabs.addTab(self.tune_tab, "⚙️ 参数优化")
        self.tabs.addTab(self.results_tab, "📊 结果分析")
        self.tabs.addTab(self.history_tab, "🗄 历史回测")
        
        root.addWidget(self.tabs)
        
        # Create menu and connect signals
        self._create_menu()
        
        # Initialize visibility for dynamic controls
        if hasattr(self, 'cmb_strategy'):
            self.cmb_strategy.currentTextChanged.connect(self._update_run_tf_visibility)
            self._update_run_tf_visibility()

    def _update_run_tf_visibility(self):
        """Show timeframe selectors only for configurable timeframe strategy; show timed-entry minute only for TimedEntryStrategy5m."""
        cls = STRATEGY_REGISTRY.get(self.cmb_strategy.currentText())
        is_cfg = (cls is ConfigurableTimeframePinbarStrategy)
        is_timed = (cls is TimedEntryStrategy5m)
        if hasattr(self, 'tf_container') and self.tf_container is not None:
            self.tf_container.setVisible(is_cfg)
        if hasattr(self, 'timed_container') and self.timed_container is not None:
            self.timed_container.setVisible(is_timed)
        if hasattr(self, 'timed_extra_container') and self.timed_extra_container is not None:
            self.timed_extra_container.setVisible(is_timed)
        # Hide min range input for timed-entry strategy
        if hasattr(self, 'w_min_range') and self.w_min_range is not None:
            self.w_min_range.setVisible(not is_timed)

    # -----------------------
    # Utilities / Helpers
    # -----------------------
    def _collect_run_params(self) -> tuple[str, int, int, type, Dict[str, Any]]:
        """Collect parameters from the Run tab widgets.

        Per new UX, symbol and H1/M1 are not shown and use defaults from config.
        """
        symbol = DEFAULT_SYMBOL
        h1 = DEFAULT_H1_BARS
        m1 = DEFAULT_M1_BARS
        strategy_cls = STRATEGY_REGISTRY.get(self.cmb_strategy.currentText(), ConfigurableTimeframePinbarStrategy)
        params = {
            "stop_loss_usd": float(self.dbl_sl.value()),
            "take_profit_usd": float(self.dbl_tp.value()),
            "min_range_usd": float(self.dbl_mr.value()),
            "qty": int(self.spin_qty.value()),
            "lot_mode": self.cmb_lotmode.currentText(),
            "commission_per_10lots_usd": float(self.dbl_commission.value()),
        }
        # If configurable strategy selected, include timeframes
        if strategy_cls is ConfigurableTimeframePinbarStrategy:
            params["signal_tf"] = self.cmb_signal_tf.currentText()
            params["backtest_tf"] = self.cmb_backtest_tf.currentText()
        # If timed-entry strategy selected, include entry_minute
        if strategy_cls is TimedEntryStrategy5m:
            params["entry_minute"] = int(self.spin_entry_minute.value())
            # Remove min_range_usd as it doesn't apply to timed-entry strategy
            params.pop("min_range_usd", None)
            params["enable_time_filter"] = bool(self.chk_time_filter.isChecked())
            params["start_hour"] = int(self.spin_start_hour.value())
            params["end_hour"] = int(self.spin_end_hour.value())
            params["trade_direction"] = self.cmb_trade_dir.currentText()
            params["max_positions"] = int(self.spin_max_positions.value())
        return symbol, h1, m1, strategy_cls, params

    def _set_run_in_progress(self, running: bool) -> None:
        """Toggle UI state for run action."""
        if running:
            self.run_log.append("🚀 正在启动回测引擎...")
            self.btn_run.setEnabled(False)
            self.btn_run.setText("🔄 回测进行中...")
            try: self.btn_stop_run.setEnabled(True)
            except Exception: pass
        else:
            self.btn_run.setEnabled(True)
            self.btn_run.setText("🚀 启动回测")
            try: self.btn_stop_run.setEnabled(False)
            except Exception: pass

    def _append_run_summary(self, stats: Dict[str, Any], strategy: Any) -> None:
        """Append a human-readable summary of a single run to the run log."""
        try:
            lines = [
                "✅ 单次回测完成:",
                f"📊 总交易次数: {stats['total_trades']}",
                f"📈 胜率: {stats['win_rate']:.2%}",
                f"💰 总盈亏: ${stats['total_pnl']:.2f}",
                f"💵 初始资金: ${strategy.initial_cash:.2f}",
                f"💎 最终资金: ${stats['final_cash']:.2f}",
                f"📊 总收益率: {stats['total_return']:.2%}",
            ]
            self.run_log.append("\n".join(lines))
        except Exception:
            # Non-critical
            pass

    def _update_run_kpis(self, stats: Dict[str, Any], trades_df: pd.DataFrame) -> None:
        """Compute and update KPI cards after a run finishes."""
        try:
            total_trades = int(stats.get('total_trades', 0))
            win_rate = float(stats.get('win_rate', 0.0))
            total_pnl = float(stats.get('total_pnl', 0.0))
            total_fees = float(stats.get('total_fees', 0.0))
            losses = 0.0
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty and 'net_pnl' in trades_df.columns:
                losses = abs(sum([float(trades_df.loc[i, 'net_pnl']) for i in range(len(trades_df)) if trades_df.loc[i, 'net_pnl'] < 0]))
            profit_factor = (total_pnl / losses) if losses > 0 else float('inf') if total_pnl > 0 else 0.0
            self.kpi_total_trades.update_value(str(total_trades))
            self.kpi_win_rate.update_value(f"{win_rate:.1%}")
            self.kpi_profit_factor.update_value("∞" if profit_factor == float('inf') else f"{profit_factor:.2f}")
            self.kpi_total_pnl.update_value(f"${total_pnl:.2f}")
            if hasattr(self, 'kpi_total_fees'):
                self.kpi_total_fees.update_value(f"${total_fees:.2f}")
        except Exception:
            # Non-critical
            pass

    def _collect_tune_params(self) -> Dict[str, Any]:
        """Collect tuning parameters from the Tune tab widgets."""
        symbol = DEFAULT_SYMBOL
        h1 = DEFAULT_H1_BARS
        m1 = DEFAULT_M1_BARS
        sl_range = (float(self.sl_min.value()), float(self.sl_max.value()), float(self.sl_step.value()))
        tp_range = (float(self.tp_min.value()), float(self.tp_max.value()), float(self.tp_step.value()))
        mr_range = (float(self.mr_min.value()), float(self.mr_max.value()), float(self.mr_step.value()))
        qty_range = (int(self.qty_min.value()), int(self.qty_max.value()), int(self.qty_step.value()))
        lot_modes: List[str] = []
        if self.chk_fixed.isChecked():
            lot_modes.append("fixed")
        if self.chk_dynamic.isChecked():
            lot_modes.append("dynamic")
        if not lot_modes:
            lot_modes = ["fixed"]
        # Strategy selection and kwargs
        strategy_cls = STRATEGY_REGISTRY.get(self.tune_strategy.currentText(), ConfigurableTimeframePinbarStrategy)
        strategy_kwargs: Dict[str, Any] = {}
        if strategy_cls is ConfigurableTimeframePinbarStrategy:
            strategy_kwargs['signal_tf'] = self.tune_signal_tf.currentText()
            strategy_kwargs['backtest_tf'] = self.tune_backtest_tf.currentText()
        entry_min_range = None
        max_positions_range = None
        # Defaults for timed-entry extra grids
        enable_time_filter_values: List[bool] | None = None
        start_hour_range: tuple[int, int, int] | None = None
        end_hour_range: tuple[int, int, int] | None = None
        trade_direction_values: List[str] | None = None
        if strategy_cls is TimedEntryStrategy5m:
            entry_min_range = (int(self.em_min.value()), int(self.em_max.value()), int(self.em_step.value()))
            max_positions_range = (int(self.mp_min.value()), int(self.mp_max.value()), int(self.mp_step.value()))
            # enable_time_filter values
            etf_vals = []
            if self.chk_etf_true.isChecked(): etf_vals.append(True)
            if self.chk_etf_false.isChecked(): etf_vals.append(False)
            enable_time_filter_values = etf_vals or [True]
            # start/end hour ranges
            start_hour_range = (int(self.sh_min.value()), int(self.sh_max.value()), int(self.sh_step.value()))
            end_hour_range = (int(self.eh_min.value()), int(self.eh_max.value()), int(self.eh_step.value()))
            # trade directions set
            td_vals = []
            if self.chk_dir_long.isChecked(): td_vals.append("Long Only")
            if self.chk_dir_short.isChecked(): td_vals.append("Short Only")
            if self.chk_dir_both.isChecked(): td_vals.append("Both")
            if self.chk_dir_alt.isChecked(): td_vals.append("Alternating")
            trade_direction_values = td_vals or ["Both"]
        # Map UI search mode to TuneWorker expected values
        search_mode_map = {
            "Grid 全面搜索": "Grid",
            "Random 随机搜索": "Random",
            "Genetic 遗传优化 (beta)": "Genetic"
        }
        search_mode = search_mode_map.get(self.cmb_search_mode.currentText(), "Grid")
        
        return {
            "symbol": symbol,
            "h1": h1,
            "m1": m1,
            "sl_range": sl_range,
            "tp_range": tp_range,
            "mr_range": mr_range,
            "qty_range": qty_range,
            "lot_modes": lot_modes,
            "max_workers": int(self.spin_workers.value()),
            "objective": self.cmb_objective.currentText(),
            "search_mode": search_mode,
            "random_iter": self.spin_random_iter.value(),
            "strategy_cls": strategy_cls,
            "strategy_kwargs": strategy_kwargs,
            "entry_min_range": entry_min_range,
            "max_positions_range": max_positions_range,
            "enable_time_filter_values": enable_time_filter_values,
            "start_hour_range": start_hour_range,
            "end_hour_range": end_hour_range,
            "trade_direction_values": trade_direction_values,
        }

    def _set_tune_in_progress(self, running: bool) -> None:
        """Toggle UI state for tuning action."""
        if running:
            self.progress.setValue(0)
            self.btn_tune.setEnabled(False)
            self.btn_tune.setText("⚙️ 优化进行中...")
            self.tune_log.append("🔧 开始参数优化...")
            self._tuning_running = True
        else:
            self.btn_tune.setEnabled(True)
            self.btn_tune.setText("⚙️ 启动参数优化")
            self._tuning_running = False

    def _save_tuning_results(self, results: List[Dict[str, Any]]) -> None:
        """Persist tuning results to both database and CSV file."""
        import os
        from datetime import datetime
        
        # Save to CSV file
        os.makedirs('outputs', exist_ok=True)
        try:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join('outputs', 'tuning_results.csv'), index=False)
        except Exception as e:
            self.tune_log.append(f"⚠️ 保存调参结果到CSV文件失败: {e}")
            
        # Save to database
        if not results:
            return
            
        try:
            # Get strategy name from the first result
            strategy_name = self.tune_strategy.currentText()
            
            # Save results to database
            dbutil.save_tuning_results(
                results=results,
                symbol=results[0].get('symbol', 'XAUUSD'),
                strategy_name=strategy_name,
                objective=self.cmb_objective.currentText(),
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            self.tune_log.append(f"✅ 成功保存 {len(results)} 条调参结果到数据库")
        except Exception as e:
            import traceback
            self.tune_log.append(f"❌ 保存调参结果到数据库时出错: {str(e)}")
            self.tune_log.append(traceback.format_exc())

    def _best_result(self, results: List[Dict[str, Any]], objective: str) -> Dict[str, Any]:
        """Return the best result by the chosen objective."""
        if not results:
            return {}
        return max(results, key=lambda x: x.get(objective, 0))

    def _populate_table_from_csv(self, path: str) -> None:
        """Populate results table from a CSV file path, if exists."""
        model = QtGui.QStandardItemModel(self)
        self.tbl_results.setModel(model)
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                headers = list(df.columns)
                model.setHorizontalHeaderLabels(headers)
                for r in range(len(df)):
                    items = [QtGui.QStandardItem(str(df.iloc[r, c])) for c in range(len(headers))]
                    model.appendRow(items)
        except Exception as e:
            self.status.showMessage(f"❌ 加载调参结果失败: {e}", 5000)

    def _create_run_tab(self):
        tab = QtWidgets.QWidget(); lay = QtWidgets.QHBoxLayout(tab); lay.setContentsMargins(24,24,24,24); lay.setSpacing(24)
        # Left params
        params = GlassCard("策略参数配置"); params.setFixedWidth(380)
        pv = QtWidgets.QVBoxLayout(); pv.setSpacing(16)
        self.cmb_strategy = QtWidgets.QComboBox(); self.cmb_strategy.addItems(list(STRATEGY_REGISTRY.keys()))
        pv.addWidget(ModernInput("策略类型", self.cmb_strategy))
        # Timeframe selectors shown directly under strategy type (only for configurable strategy)
        self.tf_container = QtWidgets.QWidget()
        tf_box = QtWidgets.QHBoxLayout(self.tf_container); tf_box.setContentsMargins(0,0,0,0)
        self.tf_box = tf_box
        self.cmb_signal_tf = QtWidgets.QComboBox(); self.cmb_signal_tf.addItems(["1h","30m","15m","5m","1m"]); self.cmb_signal_tf.setCurrentText("5m")
        self.cmb_backtest_tf = QtWidgets.QComboBox(); self.cmb_backtest_tf.addItems(["1h","30m","15m","5m","1m"]); self.cmb_backtest_tf.setCurrentText("5m")
        tf_box.addWidget(ModernInput("形态级别", self.cmb_signal_tf))
        tf_box.addWidget(ModernInput("回测级别", self.cmb_backtest_tf))
        pv.addWidget(self.tf_container)
        # Timed-entry minute selector (only for TimedEntryStrategy5m)
        self.timed_container = QtWidgets.QWidget()
        t_box = QtWidgets.QHBoxLayout(self.timed_container); t_box.setContentsMargins(0,0,0,0)
        self.spin_entry_minute = QtWidgets.QSpinBox(); self.spin_entry_minute.setRange(0,59); self.spin_entry_minute.setSingleStep(5); self.spin_entry_minute.setValue(20)
        t_box.addWidget(ModernInput("定时开单分钟 (每小时)", self.spin_entry_minute))
        pv.addWidget(self.timed_container)
        # Timed-entry extra controls (time filter / hours / direction / max positions)
        self.timed_extra_container = QtWidgets.QWidget()
        te = QtWidgets.QGridLayout(self.timed_extra_container); te.setContentsMargins(0,0,0,0)
        self.chk_time_filter = QtWidgets.QCheckBox(); self.chk_time_filter.setChecked(True)
        self.spin_start_hour = QtWidgets.QSpinBox(); self.spin_start_hour.setRange(0,23); self.spin_start_hour.setValue(8)
        self.spin_end_hour = QtWidgets.QSpinBox(); self.spin_end_hour.setRange(0,23); self.spin_end_hour.setValue(20)
        self.cmb_trade_dir = QtWidgets.QComboBox(); self.cmb_trade_dir.addItems(["Long Only","Short Only","Both","Alternating"]); self.cmb_trade_dir.setCurrentText("Both")
        self.spin_max_positions = QtWidgets.QSpinBox(); self.spin_max_positions.setRange(1,50); self.spin_max_positions.setValue(3)
        te.addWidget(ModernInput("启用时间过滤", self.chk_time_filter), 0,0)
        te.addWidget(ModernInput("开始小时", self.spin_start_hour), 0,1)
        te.addWidget(ModernInput("结束小时", self.spin_end_hour), 0,2)
        te.addWidget(ModernInput("交易方向", self.cmb_trade_dir), 1,0)
        te.addWidget(ModernInput("最大持仓数", self.spin_max_positions), 1,1)
        pv.addWidget(self.timed_extra_container)
        self.dbl_sl = QtWidgets.QDoubleSpinBox(); self.dbl_sl.setRange(0.1,1000.0); self.dbl_sl.setValue(DEFAULT_STOP_LOSS_USD)
        self.dbl_tp = QtWidgets.QDoubleSpinBox(); self.dbl_tp.setRange(0.1,1000.0); self.dbl_tp.setValue(DEFAULT_TAKE_PROFIT_USD)
        self.dbl_mr = QtWidgets.QDoubleSpinBox(); self.dbl_mr.setRange(0.1,1000.0); self.dbl_mr.setValue(DEFAULT_MIN_RANGE_USD)
        self.spin_qty = QtWidgets.QSpinBox(); self.spin_qty.setRange(1,100000); self.spin_qty.setValue(DEFAULT_QTY)
        self.dbl_commission = QtWidgets.QDoubleSpinBox(); self.dbl_commission.setRange(0.0,100.0); self.dbl_commission.setDecimals(3); self.dbl_commission.setValue(1.6)
        grid = QtWidgets.QGridLayout();
        grid.addWidget(ModernInput("止损金额 ($)", self.dbl_sl),0,0);
        grid.addWidget(ModernInput("止盈金额 ($)", self.dbl_tp),0,1);
        self.w_min_range = ModernInput("最小波动 ($)", self.dbl_mr); grid.addWidget(self.w_min_range,1,0);
        grid.addWidget(ModernInput("初始手数", self.spin_qty),1,1);
        grid.addWidget(ModernInput("10手手续费 ($)", self.dbl_commission),2,0);
        pv.addLayout(grid)
        self.cmb_lotmode = QtWidgets.QComboBox(); self.cmb_lotmode.addItems(["fixed","dynamic"]); self.cmb_lotmode.setCurrentText(DEFAULT_LOT_MODE)
        pv.addWidget(ModernInput("手数模式", self.cmb_lotmode))
        # Cache reuse toggle
        self.chk_use_cache = QtWidgets.QCheckBox(); self.chk_use_cache.setChecked(True)
        pv.addWidget(ModernInput("优先使用缓存（数据库）", self.chk_use_cache))
        # Run controls - main start button on left
        run_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = ModernButton("🚀 启动回测", "primary")
        self.btn_run.clicked.connect(self._on_run_clicked)
        run_btn_layout.addWidget(self.btn_run)
        run_btn_layout.addStretch()
        pv.addLayout(run_btn_layout)
        
        # Stop and clear buttons on right below
        control_btns_layout = QtWidgets.QHBoxLayout()
        self.btn_stop_run = ModernButton("⛔ 停止回测", "danger")
        self.btn_clear_queue = ModernButton("🧹 清理队列", "warning")
        self.btn_stop_run.clicked.connect(self._on_stop_run_clicked)
        self.btn_clear_queue.clicked.connect(self._on_clear_queue_clicked)
        control_btns_layout.addStretch()
        control_btns_layout.addWidget(self.btn_stop_run)
        control_btns_layout.addWidget(self.btn_clear_queue)
        pv.addLayout(control_btns_layout)
        params.addLayout(pv)
        # Right results
        right = QtWidgets.QVBoxLayout(); right.setSpacing(20)
        kpi = QtWidgets.QGridLayout(); kpi.setSpacing(16)
        self.kpi_total_trades = KPICard("总交易", "0", color="#4A9EFF")
        self.kpi_win_rate = KPICard("胜率", "0%", color="#52C41A")
        self.kpi_profit_factor = KPICard("盈亏比", "0.00", color="#FA8C16")
        self.kpi_total_pnl = KPICard("总盈亏", "$0.00", color="#F5222D")
        self.kpi_total_fees = KPICard("总费用", "$0.00", color="#888888")
        kpi.addWidget(self.kpi_total_trades,0,0); kpi.addWidget(self.kpi_win_rate,0,1); kpi.addWidget(self.kpi_profit_factor,0,2); kpi.addWidget(self.kpi_total_pnl,0,3)
        chart = GlassCard("回测图表"); self.canvas = MplCanvas(width=12, height=7, dpi=100); chart.addWidget(self.canvas)
        logc = GlassCard("执行日志"); self.run_log = QtWidgets.QTextEdit(); self.run_log.setReadOnly(True); self.run_log.setMaximumHeight(160); self.run_log.setStyleSheet("QTextEdit{background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.1);border-radius:6px;color:#fff;font-family:'Consolas','Monaco',monospace;font-size:12px;padding:12px;}"); logc.addWidget(self.run_log)
        right.addLayout(kpi); right.addWidget(chart,1); right.addWidget(logc)
        lay.addWidget(params); lay.addLayout(right,1)
        return tab

    def _create_tune_tab(self):
        tab = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        
        # Main content container
        content = QtWidgets.QWidget()
        main = QtWidgets.QVBoxLayout(content)
        main.setContentsMargins(24, 24, 24, 24)
        main.setSpacing(20)
        
        # Basic settings card
        basic = GlassCard("基础设置")
        bl = QtWidgets.QVBoxLayout()
        bl.setSpacing(16)
        
        # Strategy type selector
        self.tune_strategy = QtWidgets.QComboBox()
        if STRATEGY_REGISTRY:
            self.tune_strategy.addItems(list(STRATEGY_REGISTRY.keys()))
        bl.addWidget(ModernInput("策略类型", self.tune_strategy))
        
        # Signal timeframe selector (识别周期)
        self.tune_signal_tf = QtWidgets.QComboBox()
        self.tune_signal_tf.addItems(["1h", "30m", "15m", "5m", "1m"])
        self.tune_signal_tf.setCurrentText("5m")
        self.w_tune_signal_tf = ModernInput("形态识别周期", self.tune_signal_tf)
        bl.addWidget(self.w_tune_signal_tf)
        
        # Backtest timeframe selector (回测周期)
        self.tune_backtest_tf = QtWidgets.QComboBox()
        self.tune_backtest_tf.addItems(["1h", "30m", "15m", "5m", "1m"])
        self.tune_backtest_tf.setCurrentText("5m")
        self.w_tune_backtest_tf = ModernInput("回测周期", self.tune_backtest_tf)
        bl.addWidget(self.w_tune_backtest_tf)
        
        bl.addStretch()
        basic.addLayout(bl)
        main.addWidget(basic)
        # Ranges
        ranges = GlassCard("参数范围设置"); ranges.setFixedWidth(420)
        # SL
        sl_group = QtWidgets.QGroupBox("止损参数"); sl = QtWidgets.QHBoxLayout(sl_group)
        self.sl_min = QtWidgets.QDoubleSpinBox(); self.sl_min.setRange(0.1,5000.0); self.sl_min.setValue(DEFAULT_STOP_LOSS_USD)
        self.sl_max = QtWidgets.QDoubleSpinBox(); self.sl_max.setRange(0.1,5000.0); self.sl_max.setValue(DEFAULT_STOP_LOSS_USD+20)
        self.sl_step = QtWidgets.QDoubleSpinBox(); self.sl_step.setRange(0.1,5000.0); self.sl_step.setValue(5.0)
        sl.addWidget(ModernInput("最小值", self.sl_min)); sl.addWidget(ModernInput("最大值", self.sl_max)); sl.addWidget(ModernInput("步长", self.sl_step))
        # TP
        tp_group = QtWidgets.QGroupBox("止盈参数"); tp = QtWidgets.QHBoxLayout(tp_group)
        self.tp_min = QtWidgets.QDoubleSpinBox(); self.tp_min.setRange(0.1,5000.0); self.tp_min.setValue(DEFAULT_TAKE_PROFIT_USD)
        self.tp_max = QtWidgets.QDoubleSpinBox(); self.tp_max.setRange(0.1,5000.0); self.tp_max.setValue(DEFAULT_TAKE_PROFIT_USD+20)
        self.tp_step = QtWidgets.QDoubleSpinBox(); self.tp_step.setRange(0.1,5000.0); self.tp_step.setValue(5.0)
        tp.addWidget(ModernInput("最小值", self.tp_min)); tp.addWidget(ModernInput("最大值", self.tp_max)); tp.addWidget(ModernInput("步长", self.tp_step))
        # MR
        mr_group = QtWidgets.QGroupBox("最小波动参数"); mr = QtWidgets.QHBoxLayout(mr_group)
        self.mr_min = QtWidgets.QDoubleSpinBox(); self.mr_min.setRange(0.1,5000.0); self.mr_min.setValue(DEFAULT_MIN_RANGE_USD)
        self.mr_max = QtWidgets.QDoubleSpinBox(); self.mr_max.setRange(0.1,5000.0); self.mr_max.setValue(DEFAULT_MIN_RANGE_USD+10)
        self.mr_step = QtWidgets.QDoubleSpinBox(); self.mr_step.setRange(0.1,5000.0); self.mr_step.setValue(2.0)
        mr.addWidget(ModernInput("最小值", self.mr_min)); mr.addWidget(ModernInput("最大值", self.mr_max)); mr.addWidget(ModernInput("步长", self.mr_step))
        self.mr_group = mr_group
        # QTY
        qty_group = QtWidgets.QGroupBox("初始手数参数"); qg = QtWidgets.QHBoxLayout(qty_group)
        self.qty_min = QtWidgets.QSpinBox(); self.qty_min.setRange(1,100000); self.qty_min.setValue(10)
        self.qty_max = QtWidgets.QSpinBox(); self.qty_max.setRange(1,100000); self.qty_max.setValue(50)
        self.qty_step = QtWidgets.QSpinBox(); self.qty_step.setRange(1,100000); self.qty_step.setValue(10)
        qg.addWidget(ModernInput("最小值", self.qty_min)); qg.addWidget(ModernInput("最大值", self.qty_max)); qg.addWidget(ModernInput("步长", self.qty_step))
        # Timed-entry grids
        timed_group = QtWidgets.QGroupBox("定时做单参数"); tg = QtWidgets.QGridLayout(timed_group)
        self.em_min = QtWidgets.QSpinBox(); self.em_min.setRange(0,59); self.em_min.setValue(20); self.em_min.setSingleStep(5)
        self.em_max = QtWidgets.QSpinBox(); self.em_max.setRange(0,59); self.em_max.setValue(40); self.em_max.setSingleStep(5)
        self.em_step = QtWidgets.QSpinBox(); self.em_step.setRange(1,59); self.em_step.setValue(5)
        self.mp_min = QtWidgets.QSpinBox(); self.mp_min.setRange(1,50); self.mp_min.setValue(1)
        self.mp_max = QtWidgets.QSpinBox(); self.mp_max.setRange(1,50); self.mp_max.setValue(5)
        self.mp_step = QtWidgets.QSpinBox(); self.mp_step.setRange(1,50); self.mp_step.setValue(1)
        tg.addWidget(ModernInput("开单分钟-最小", self.em_min), 0,0)
        tg.addWidget(ModernInput("开单分钟-最大", self.em_max), 0,1)
        tg.addWidget(ModernInput("开单分钟-步长(5)", self.em_step), 0,2)
        tg.addWidget(ModernInput("最大持仓-最小", self.mp_min), 1,0)
        tg.addWidget(ModernInput("最大持仓-最大", self.mp_max), 1,1)
        tg.addWidget(ModernInput("最大持仓-步长", self.mp_step), 1,2)
        # Timed-entry: enable_time_filter values
        self.chk_etf_true = QtWidgets.QCheckBox("启用时间过滤=True"); self.chk_etf_true.setChecked(True)
        self.chk_etf_false = QtWidgets.QCheckBox("启用时间过滤=False")
        tg.addWidget(self.chk_etf_true, 2,0)
        tg.addWidget(self.chk_etf_false, 2,1)
        # Timed-entry: trading session ranges
        self.sh_min = QtWidgets.QSpinBox(); self.sh_min.setRange(0,23); self.sh_min.setValue(8)
        self.sh_max = QtWidgets.QSpinBox(); self.sh_max.setRange(0,23); self.sh_max.setValue(8)
        self.sh_step = QtWidgets.QSpinBox(); self.sh_step.setRange(1,23); self.sh_step.setValue(1)
        self.eh_min = QtWidgets.QSpinBox(); self.eh_min.setRange(0,23); self.eh_min.setValue(20)
        self.eh_max = QtWidgets.QSpinBox(); self.eh_max.setRange(0,23); self.eh_max.setValue(20)
        self.eh_step = QtWidgets.QSpinBox(); self.eh_step.setRange(1,23); self.eh_step.setValue(1)
        tg.addWidget(ModernInput("开始小时-最小", self.sh_min), 3,0)
        tg.addWidget(ModernInput("开始小时-最大", self.sh_max), 3,1)
        tg.addWidget(ModernInput("开始小时-步长", self.sh_step), 3,2)
        tg.addWidget(ModernInput("结束小时-最小", self.eh_min), 4,0)
        tg.addWidget(ModernInput("结束小时-最大", self.eh_max), 4,1)
        tg.addWidget(ModernInput("结束小时-步长", self.eh_step), 4,2)
        # Timed-entry: trade direction set
        self.chk_dir_long = QtWidgets.QCheckBox("Long Only")
        self.chk_dir_short = QtWidgets.QCheckBox("Short Only")
        self.chk_dir_both = QtWidgets.QCheckBox("Both"); self.chk_dir_both.setChecked(True)
        self.chk_dir_alt = QtWidgets.QCheckBox("Alternating")
        dir_box = QtWidgets.QHBoxLayout();
        dir_box.addWidget(self.chk_dir_long); dir_box.addWidget(self.chk_dir_short); dir_box.addWidget(self.chk_dir_both); dir_box.addWidget(self.chk_dir_alt)
        dir_wrap = QtWidgets.QWidget(); dir_wrap.setLayout(dir_box)
        tg.addWidget(ModernInput("交易方向集合", dir_wrap), 5,0,1,3)
        self.timed_group = timed_group
        rl = QtWidgets.QVBoxLayout(); rl.addWidget(sl_group); rl.addWidget(tp_group); rl.addWidget(mr_group); rl.addWidget(qty_group); rl.addWidget(timed_group)
        ranges.addLayout(rl)
        # Advanced
        adv = GlassCard("高级选项"); al = QtWidgets.QVBoxLayout(); al.setSpacing(16)
        self.chk_fixed = QtWidgets.QCheckBox("Fixed 固定手数"); self.chk_fixed.setChecked(True)
        self.chk_dynamic = QtWidgets.QCheckBox("Dynamic 动态手数")
        al.addWidget(self.chk_fixed); al.addWidget(self.chk_dynamic)
        self.cmb_objective = QtWidgets.QComboBox(); self.cmb_objective.addItems(["total_return","win_rate","total_pnl","profit_factor","score"]) 
        al.addWidget(ModernInput("优化目标", self.cmb_objective))
        # Search mode + iterations
        self.cmb_search_mode = QtWidgets.QComboBox(); self.cmb_search_mode.addItems(["Grid 全面搜索","Random 随机搜索","Genetic 遗传优化 (beta)"])
        self.spin_random_iter = QtWidgets.QSpinBox(); self.spin_random_iter.setRange(1, 100000); self.spin_random_iter.setValue(50)
        al.addWidget(ModernInput("搜索模式", self.cmb_search_mode))
        al.addWidget(ModernInput("随机搜索迭代次数", self.spin_random_iter))
        # Scoring weights
        self.w_return = QtWidgets.QDoubleSpinBox(); self.w_return.setRange(0.0, 10.0); self.w_return.setDecimals(2); self.w_return.setSingleStep(0.1); self.w_return.setValue(1.0)
        self.w_win = QtWidgets.QDoubleSpinBox(); self.w_win.setRange(0.0, 10.0); self.w_win.setDecimals(2); self.w_win.setSingleStep(0.1); self.w_win.setValue(0.0)
        self.w_pf = QtWidgets.QDoubleSpinBox(); self.w_pf.setRange(0.0, 10.0); self.w_pf.setDecimals(2); self.w_pf.setSingleStep(0.1); self.w_pf.setValue(0.0)
        al.addWidget(ModernInput("评分权重-总收益 (w_return)", self.w_return))
        al.addWidget(ModernInput("评分权重-胜率 (w_win)", self.w_win))
        al.addWidget(ModernInput("评分权重-盈亏比 (w_pf)", self.w_pf))
        # Combo count + Stop
        self.lbl_combo_count = QtWidgets.QLabel("组合数: 0")
        self.btn_stop_tune = ModernButton("⛔ 停止优化")
        self.btn_stop_tune.clicked.connect(self._on_stop_tune)
        al.addWidget(self.lbl_combo_count)
        al.addWidget(self.btn_stop_tune)
        self.spin_workers = QtWidgets.QSpinBox(); self.spin_workers.setRange(1,64); self.spin_workers.setValue(max(1, (os.cpu_count() or 1))); self.spin_workers.setEnabled(True)
        al.addWidget(ModernInput("并发线程", self.spin_workers)); al.addStretch(); adv.addLayout(al)
        # Control
        ctrl = GlassCard("执行控制"); cl = QtWidgets.QVBoxLayout(); pl = QtWidgets.QVBoxLayout();
        pl.addWidget(QtWidgets.QLabel("优化进度")); self.progress = AnimatedProgressBar(); pl.addWidget(self.progress)
        buttons = QtWidgets.QHBoxLayout(); self.btn_tune = ModernButton("⚙️ 启动参数优化","warning"); self.btn_tune.clicked.connect(self._on_tune_clicked)
        self.btn_apply_best = ModernButton("✅ 应用最优参数", "primary"); self.btn_apply_best.setEnabled(False); self.btn_apply_best.clicked.connect(self._on_apply_best_clicked)
        buttons.addWidget(self.btn_tune,1); buttons.addWidget(self.btn_apply_best,1)
        cl.addLayout(pl); cl.addLayout(buttons); ctrl.addLayout(cl)
        # Log
        log = GlassCard("优化日志"); self.tune_log = QtWidgets.QTextEdit(); self.tune_log.setReadOnly(True); self.tune_log.setMaximumHeight(160)
        log.addWidget(self.tune_log)
        # Assemble
        row = QtWidgets.QHBoxLayout(); row.addWidget(basic); row.addWidget(ranges,1); row.addWidget(adv)
        main.addLayout(row); main.addWidget(ctrl); main.addWidget(log); main.addStretch()
        scroll.setWidget(content)
        tl = QtWidgets.QVBoxLayout(tab); tl.setContentsMargins(0,0,0,0); tl.addWidget(scroll)
        # Visibility toggle for tune tab controls by strategy type
        self.tune_strategy.currentTextChanged.connect(self._update_tune_visibility)
        self._update_tune_visibility()
        # 实时更新组合数
        try:
            self._connect_combo_count_signals()
            self._refresh_combo_count()
        except Exception:
            pass
        return tab

    def _create_results_tab(self):
        tab = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(tab); lay.setContentsMargins(24,24,24,24); lay.setSpacing(16)

        # Results table with dark theme
        table_card = GlassCard("调参结果")
        tvl = QtWidgets.QVBoxLayout()
        self.tbl_results = QtWidgets.QTableView()
        self.tbl_results.setSortingEnabled(True)
        self.tbl_results.setAlternatingRowColors(True)
        self.tbl_results.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tbl_results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_results.setStyleSheet(
            """
            QTableView {
                background-color: rgba(0, 0, 0, 0.25);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 6px;
                gridline-color: rgba(255, 255, 255, 0.1);
                selection-background-color: rgba(74, 158, 255, 0.35);
                selection-color: #ffffff;
                alternate-background-color: rgba(255, 255, 255, 0.04);
                font-size: 13px;
            }
            QHeaderView::section {
                background-color: rgba(74, 158, 255, 0.18);
                color: #ffffff;
                padding: 10px 8px;
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
                font-weight: 600;
                font-size: 13px;
            }
            QTableView::item {
                padding: 6px 8px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.08);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.28);
                min-height: 20px;
                border-radius: 6px;
            }
            """
        )
        tvl.addWidget(self.tbl_results)
        table_card.addLayout(tvl)

        lay.addWidget(table_card, 1)
        self._load_outputs_to_table()
        return tab

    def _create_history_tab(self):
        from utils import db as dbutil
        tab = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(tab); lay.setContentsMargins(24,24,24,24); lay.setSpacing(12)
        # Filters row
        filters = QtWidgets.QHBoxLayout(); filters.setSpacing(12)
        self.hist_symbol = QtWidgets.QLineEdit(); self.hist_symbol.setPlaceholderText("品种(模糊)")
        self.hist_strategy = QtWidgets.QComboBox(); self.hist_strategy.addItem("")
        for name, cls in STRATEGY_REGISTRY.items():
            self.hist_strategy.addItem(cls.__name__)
        self.hist_min_wr = QtWidgets.QDoubleSpinBox(); self.hist_min_wr.setRange(0.0, 1.0); self.hist_min_wr.setSingleStep(0.05); self.hist_min_wr.setValue(0.0)
        btn_refresh = ModernButton("刷新", "ghost"); btn_refresh.clicked.connect(self._refresh_history)
        filters.addWidget(ModernInput("品种包含", self.hist_symbol))
        filters.addWidget(ModernInput("策略", self.hist_strategy))
        filters.addWidget(ModernInput("最小胜率(0-1)", self.hist_min_wr))
        filters.addWidget(btn_refresh); filters.addStretch()
        lay.addLayout(filters)
        # Table
        self.tbl_runs = QtWidgets.QTableView(); self.tbl_runs.setSortingEnabled(True)
        self.tbl_runs.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl_runs.doubleClicked.connect(self._open_run_details)
        lay.addWidget(self.tbl_runs, 1)
        # Initial load
        self._refresh_history()
        return tab

    def _refresh_history(self):
        try:
            # 确保必要的导入
            import sys
            import os
            import sqlite3
            import pandas as pd
            
            # 添加项目根目录到路径
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # 现在使用绝对导入
            from mt5_backtest_multi.utils import db as dbutil
            
            # 获取筛选条件
            symbol_like = self.hist_symbol.text().strip() if hasattr(self, 'hist_symbol') and self.hist_symbol.text().strip() else None
            strat_name = self.hist_strategy.currentText().strip() if hasattr(self, 'hist_strategy') and self.hist_strategy.currentText().strip() else None
            min_wr = float(self.hist_min_wr.value()) if hasattr(self, 'hist_min_wr') and self.hist_min_wr.value() > 0 else None
            
            # 查询数据库
            try:
                df = dbutil.query_runs_flat(symbol_like=symbol_like, strategy_name=strat_name, limit=200)
                if df is None or df.empty:
                    # 尝试直接查询数据库表
                    conn = sqlite3.connect(dbutil.DB_PATH)
                    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
                    if 'runs' in tables['name'].values:
                        pd.read_sql("SELECT * FROM runs LIMIT 5", conn)
                    conn.close()
            except Exception as e:
                pass
                raise
            
            # 创建模型
            model = QtGui.QStandardItemModel(self)
            headers = [
                "ID", "类型", "品种", "策略", "信号周期", "回测周期",
                "止损(USD)", "止盈(USD)", "最小范围", "手数", "手数模式", "手续费",
                "胜率", "总收益", "净盈亏", "手续费总额", "最终资金"
            ]
            model.setHorizontalHeaderLabels(headers)
            
            # 填充数据
            if not df.empty:
                if min_wr is not None:
                    df = df[df['win_rate'] >= float(min_wr)]
                
                for _, row in df.iterrows():
                    row_data = [
                        str(int(row.get('id', 0))),
                        '优化' if row.get('is_tuning') else '回测',
                        str(row.get('symbol', '')),
                        str(row.get('strategy', '')),
                        str(row.get('signal_tf', '')),
                        str(row.get('backtest_tf', '')),
                        f"{float(row.get('stop_loss_usd', 0)):.2f}",
                        f"{float(row.get('take_profit_usd', 0)):.2f}",
                        f"{float(row.get('min_range_usd', 0)):.2f}" if 'min_range_usd' in df.columns else 'N/A',
                        str(int(row.get('qty', 0))),
                        str(row.get('lot_mode', '')),
                        f"{float(row.get('commission_per_10lots_usd', 0)):.2f}",
                        f"{float(row.get('win_rate', 0)) * 100:.1f}%",
                        f"{float(row.get('total_return', 0)) * 100:.2f}%",
                        f"{float(row.get('total_pnl', 0)):.2f}",
                        f"{float(row.get('total_fees', 0)):.2f}",
                        f"{float(row.get('final_cash', 0)):.2f}"
                    ]
                    model.appendRow([QtGui.QStandardItem(v) for v in row_data])
            
            # 设置模型并调整列宽
            self.tbl_runs.setModel(model)
            self.tbl_runs.resizeColumnsToContents()
            
            # 显示状态信息
            self.status.showMessage(f"✅ 已加载 {model.rowCount()} 条记录", 3000)
            
        except Exception as e:
            error_msg = f"❌ 加载历史回测失败: {str(e)}"
            print(error_msg)
            self.status.showMessage(error_msg, 5000)
            # 显示详细的错误信息
            error_dialog = QtWidgets.QErrorMessage(self)
            error_dialog.setWindowTitle("错误")
            error_dialog.showMessage(f"加载历史回测数据时出错:\n{str(e)}\n\n请检查数据库连接和表结构。")
            error_dialog.exec_()

    def _open_run_details(self):
        try:
            # 获取选中的行
            index = self.tbl_runs.currentIndex()
            if not index.isValid():
                return
                
            # 获取运行ID
            model = self.tbl_runs.model()
            rid = int(model.item(index.row(), 0).text())
            
            # 导入必要的模块
            import sqlite3
            import pandas as pd
            from PySide6 import QtWidgets, QtGui, QtCore
            
            # 创建对话框
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle(f"回测详情 - 记录 #{rid}")
            dlg.resize(1200, 800)
            
            # 主布局
            layout = QtWidgets.QVBoxLayout(dlg)
            
            # 标签页
            tab_widget = QtWidgets.QTabWidget()
            
            # 1. 交易记录标签页
            trades_tab = QtWidgets.QWidget()
            trades_layout = QtWidgets.QVBoxLayout(trades_tab)
            
            # 获取交易数据
            try:
                from mt5_backtest_multi.utils import db as dbutil
                print(f"正在获取交易记录，run_id={rid}")
                trades = dbutil.get_trades(rid)
                
                if trades is not None and not trades.empty:
                    print(f"找到 {len(trades)} 条交易记录")
                    # 创建交易表格
                    trades_table = QtWidgets.QTableView()
                    model = QtGui.QStandardItemModel()
                    
                    # 设置表头
                    headers = ['开仓时间', '平仓时间', '方向', '开仓价', '平仓价', '手数', '盈亏', '收益率', '持仓时间']
                    model.setHorizontalHeaderLabels(headers)
                    
                    # 添加数据
                    for _, trade in trades.iterrows():
                        try:
                            # 处理方向 - 支持字符串 'BUY'/'SELL' 或数字 1/-1
                            direction = str(trade.get('direction', '')).upper()
                            if direction in ['1', '1.0', 'BUY', 'LONG', '多']:
                                direction_display = '买入'
                            elif direction in ['-1', '-1.0', 'SELL', 'SHORT', '空']:
                                direction_display = '卖出'
                            else:
                                direction_display = str(direction)
                            
                            # 处理价格和数值
                            open_price = float(trade.get('open_price', 0) or 0)
                            close_price = float(trade.get('close_price', 0) or 0)
                            size = float(trade.get('size', trade.get('qty', 0)) or 0)
                            pnl = float(trade.get('pnl', trade.get('net_pnl', 0)) or 0)
                            return_pct = float(trade.get('return_pct', 0) or 0)
                            
                            # 处理时间
                            open_time = trade.get('open_time', trade.get('entry_time', ''))
                            close_time = trade.get('close_time', trade.get('exit_time', ''))
                            
                            # 计算持仓时间（如果可能）
                            duration = trade.get('duration', '')
                            if not duration and open_time and close_time:
                                try:
                                    from datetime import datetime
                                    fmt = '%Y-%m-%d %H:%M:%S'
                                    t1 = datetime.strptime(str(open_time).split('.')[0], fmt)
                                    t2 = datetime.strptime(str(close_time).split('.')[0], fmt)
                                    duration = str(t2 - t1)
                                except Exception:
                                    duration = ''
                            
                            row = [
                                str(open_time),
                                str(close_time),
                                direction_display,
                                f"{open_price:.5f}",
                                f"{close_price:.5f}",
                                str(int(size)),
                                f"{pnl:.2f}",
                                f"{return_pct * 100:.2f}%",
                                str(duration)
                            ]
                            model.appendRow([QtGui.QStandardItem(str(item)) for item in row])
                        except Exception as e:
                            print(f"处理交易记录时出错: {str(e)}", trade)
                            continue
                    
                    if model.rowCount() > 0:
                        trades_table.setModel(model)
                        trades_table.resizeColumnsToContents()
                        trades_layout.addWidget(trades_table)
                    else:
                        trades_layout.addWidget(QtWidgets.QLabel("没有有效的交易记录"))
                else:
                    print("没有找到交易记录或记录为空")
                    trades_layout.addWidget(QtWidgets.QLabel("没有找到交易记录"))
            except Exception as e:
                error_msg = f"加载交易记录时出错: {str(e)}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                trades_layout.addWidget(QtWidgets.QLabel(error_msg))
            
            # 2. 统计信息标签页
            stats_tab = QtWidgets.QWidget()
            stats_layout = QtWidgets.QVBoxLayout(stats_tab)
            
            try:
                # 获取统计信息
                conn = sqlite3.connect(dbutil.DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT stats_json FROM runs WHERE id = ?", (rid,))
                row = cursor.fetchone()
                
                if row and row[0]:
                    stats = json.loads(row[0])
                    
                    # 创建统计信息表格
                    stats_table = QtWidgets.QTableWidget()
                    stats_table.setColumnCount(2)
                    stats_table.setHorizontalHeaderLabels(['指标', '值'])
                    stats_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
                    
                    # 添加统计信息
                    stats_table.setRowCount(len(stats))
                    for i, (key, value) in enumerate(stats.items()):
                        stats_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(key)))
                        stats_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(round(value, 4) if isinstance(value, (int, float)) else value)))
                    
                    stats_table.resizeColumnsToContents()
                    stats_layout.addWidget(stats_table)
                else:
                    stats_layout.addWidget(QtWidgets.QLabel("没有找到统计信息"))
                
                conn.close()
            except Exception as e:
                stats_layout.addWidget(QtWidgets.QLabel(f"加载统计信息时出错: {str(e)}"))
            
            # 添加标签页
            tab_widget.addTab(trades_tab, "交易记录")
            tab_widget.addTab(stats_tab, "统计信息")
            
            # 添加标签页到主布局
            layout.addWidget(tab_widget)
            
            # 添加关闭按钮
            btn_close = QtWidgets.QPushButton("关闭")
            btn_close.clicked.connect(dlg.accept)
            layout.addWidget(btn_close, alignment=QtCore.Qt.AlignRight)
            
            # 显示对话框
            dlg.exec_()
            
        except Exception as e:
            error_msg = f"打开回测详情时出错: {str(e)}"
            print(error_msg)
            self.status.showMessage(error_msg, 5000)

    def _create_menu(self):
        menubar = self.menuBar(); file_menu = menubar.addMenu("📁 文件")
        act_open = QtGui.QAction("📊 打开交易记录 CSV", self); act_open.triggered.connect(self._open_csv); file_menu.addAction(act_open)
        act_dir = QtGui.QAction("📁 打开输出目录", self); act_dir.triggered.connect(lambda: self._open_dir("outputs")); file_menu.addAction(act_dir)
        file_menu.addSeparator(); act_exit = QtGui.QAction("❌ 退出", self); act_exit.triggered.connect(self.close); file_menu.addAction(act_exit)
        help_menu = menubar.addMenu("❓ 帮助"); act_about = QtGui.QAction("ℹ️ 关于", self); act_about.triggered.connect(self._show_about_dialog); help_menu.addAction(act_about)

    def _apply_modern_theme(self):
        pal = self.palette(); pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#121212")); pal.setColor(QtGui.QPalette.Base, QtGui.QColor("#1e1e1e")); pal.setColor(QtGui.QPalette.Text, QtGui.QColor("#ffffff")); pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#4A9EFF")); pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff")); self.setPalette(pal)
        self.setStyleSheet("QMainWindow{background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0f0f0f, stop:0.5 #121212, stop:1 #1a1a1a);color:#fff;} QWidget{color:#fff;}")

    # Handlers
    def _on_run_clicked(self):
        symbol, h1, m1, strategy_cls, params = self._collect_run_params()
        # If tuning is running, block starting a new run to avoid thread pool saturation
        if getattr(self, '_tuning_running', False):
            QtWidgets.QMessageBox.information(self, "提示", "参数优化正在运行，请先停止或等待完成后再回测。")
            return
        # Log selected strategy and commission for verification
        try:
            self.run_log.append(f"▶️ 策略: {strategy_cls.__name__} | 10手手续费: {params.get('commission_per_10lots_usd', '默认')}")
        except Exception:
            pass
        self._set_run_in_progress(True)
        self._run_stop_requested = False
        self._run_progress_seen = False
        # Clear queued tasks to avoid long wait if prior tasks are stuck in queue
        try:
            self.thread_pool.clear()
        except Exception:
            pass
        reuse_db = True
        try:
            reuse_db = bool(self.chk_use_cache.isChecked())
        except Exception:
            pass
        w = RunBacktestWorker(symbol, h1, m1, strategy_cls, params, reuse_db=reuse_db)
        self._active_run_worker = w
        w.signals.finished.connect(self._on_run_finished)
        w.signals.error.connect(self._on_error)
        w.signals.progress.connect(self._on_run_progress)
        self.thread_pool.start(w)
        try:
            self.run_log.append(f"[RUN] 已提交后台任务 | 活动线程: {self.thread_pool.activeThreadCount()} / {self.thread_pool.maxThreadCount()}")
        except Exception:
            pass
        # If 2s 内无任何进度，提示可能仍在排队
        try:
            QtCore.QTimer.singleShot(2000, lambda: self._watchdog_run_queued())
        except Exception:
            pass
        # Watchdog: if 5分钟还未完成，提示可能卡住
        try:
            QtCore.QTimer.singleShot(5 * 60 * 1000, lambda: self._watchdog_run_in_progress())
        except Exception:
            pass

    def _watchdog_run_in_progress(self):
        try:
            # 若按钮仍是“回测进行中…”，说明还没完成
            if hasattr(self, 'btn_run') and self.btn_run.isEnabled() is False:
                self.run_log.append("⚠️ 回测超过5分钟未完成，可能正在等待数据或被服务器限制。可稍候继续，或重启再试。")
                # 不强制重置，避免打断真正的长周期回测
        except Exception:
            pass

    def _watchdog_run_queued(self):
        try:
            # 若已开始有进度或已完成，则不提示
            if getattr(self, '_run_progress_seen', False):
                return
            if hasattr(self, 'btn_run') and self.btn_run.isEnabled():
                return
            # 这里简单输出一次提示
            self.run_log.append("[RUN] 任务已提交，若长时间无进度，请检查线程池是否繁忙或上一次任务是否未结束")
        except Exception:
            pass

    def _on_tune_progress(self, message):
        """Handle progress updates from tuning process."""
        if not hasattr(self, 'tune_log'):
            return
            
        # Update log
        self.tune_log.append(message)
        
        # Auto-scroll to bottom
        self.tune_log.verticalScrollBar().setValue(
            self.tune_log.verticalScrollBar().maximum()
        )
        
        # Parse progress percentage from message if available
        if '优化进度:' in message and '%' in message:
            try:
                # Extract percentage from message
                percent_str = message.split('(')[1].split('%')[0].strip()
                percent = int(float(percent_str))
                
                # Update progress bar if it exists
                if hasattr(self, 'tune_progress'):
                    self.tune_progress.setValue(percent)
                    self.tune_progress.setFormat(f"{percent}%")
                    
                # Force UI update
                QtWidgets.QApplication.processEvents()
            except (IndexError, ValueError) as e:
                # If parsing fails, just log the message without updating progress bar
                pass
                
    def _on_run_progress(self, s: str):
        try:
            self._run_progress_seen = True
        except Exception:
            pass
        try:
            self.run_log.append(str(s))
        except Exception:
            pass

    def _update_tune_visibility(self):
        cls = STRATEGY_REGISTRY.get(self.tune_strategy.currentText())
        is_cfg = (cls is ConfigurableTimeframePinbarStrategy)
        is_timed = (cls is TimedEntryStrategy5m)
        # Timeframe selectors visible only for configurable pinbar
        if hasattr(self, 'w_tune_signal_tf'): self.w_tune_signal_tf.setVisible(is_cfg)
        if hasattr(self, 'w_tune_backtest_tf'): self.w_tune_backtest_tf.setVisible(is_cfg)
        # Min range group hidden for timed strategy
        if hasattr(self, 'mr_group'): self.mr_group.setVisible(not is_timed)
        # Timed grids visible only for timed strategy
        if hasattr(self, 'timed_group'): self.timed_group.setVisible(is_timed)

    def _on_run_finished(self, payload: Dict[str, Any]):
        try:
            # Ignore if user requested stop
            if getattr(self, '_run_stop_requested', False):
                self.run_log.append("⏹ 已停止：忽略此次回测结果。")
                return
            data = payload or {}
            trades_df: pd.DataFrame = data.get("trades_df", pd.DataFrame())
            stats: Dict[str, Any] = data.get("stats", {})
            strategy = data.get("strategy")
            self.run_log.append("✅ 回测完成")
            # Concise DB source/row info
            try:
                src = str(stats.get('source', 'unknown'))
                rid = stats.get('db_run_id')
                pkey = stats.get('param_key')
                key_short = None
                if isinstance(pkey, str) and len(pkey) > 60:
                    key_short = pkey[:57] + '...'
                elif pkey:
                    key_short = pkey
                line = f"🗄 {src.upper()} | run_id={rid}"
                if key_short:
                    line += f" | key={key_short}"
                self.run_log.append(line)
            except Exception:
                pass
            self._append_run_summary(stats, strategy)
            self._update_run_kpis(stats, trades_df)
            # Generate and save PnL 3-panel chart to outputs/ (only for backtest flow)
            try:
                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty and strategy is not None:
                    os.makedirs('outputs', exist_ok=True)
                    title = f"PnL - {self.cmb_strategy.currentText()}"
                    fig = plot_pnl_chart(trades_df, getattr(strategy, 'initial_cash', 0.0), title, show=False)
                    from datetime import datetime as _dt
                    out_png = os.path.join('outputs', f"pnl_chart_{_dt.now().strftime('%Y%m%d_%H%M%S')}.png")
                    fig.savefig(out_png, dpi=150, bbox_inches='tight')
                    self.run_log.append(f"📈 图表已保存: {out_png}")
            except Exception:
                pass
            # Save last trades snapshot
            try:
                if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                    out_csv = os.path.join('outputs', 'last_backtest_trades.csv')
                    trades_df.to_csv(out_csv, index=False)
                    self.run_log.append(f"💾 交易记录已保存: {out_csv}")
            except Exception:
                pass
        finally:
            self._set_run_in_progress(False)

    def _on_stop_run_clicked(self):
        """Soft-cancel current run: mark stop flag, reset UI, and ignore future results."""
        try:
            self._run_stop_requested = True
            self.run_log.append("⛔ 已请求停止当前回测（后续结果将被忽略）。")
            self._set_run_in_progress(False)
        except Exception:
            pass

    def _on_clear_queue_clicked(self):
        try:
            self.thread_pool.clear()
            self.run_log.append("🧹 已清理线程队列。")
        except Exception as e:
            self.run_log.append(f"🧹 清理队列失败: {e}")

    def _on_tune_clicked(self):
        p = self._collect_tune_params()
        # 归一化评分权重，使其和为1
        try:
            w_ret = float(self.w_return.value()); w_win = float(self.w_win.value()); w_pf = float(self.w_pf.value())
            total = w_ret + w_win + w_pf
            if total <= 0:
                w_ret, w_win, w_pf = 1.0, 0.0, 0.0
                total = 1.0
            w_ret_n = w_ret/total; w_win_n = w_win/total; w_pf_n = w_pf/total
            # 回写UI，确保用户看到权重归一化
            self.w_return.setValue(w_ret_n)
            self.w_win.setValue(w_win_n)
            self.w_pf.setValue(w_pf_n)
        except Exception:
            w_ret_n, w_win_n, w_pf_n = 1.0, 0.0, 0.0

        # Get the strategy class from the registry
        strategy_name = self.tune_strategy.currentText()
        strategy_cls = STRATEGY_REGISTRY.get(strategy_name, ConfigurableTimeframePinbarStrategy)
        
        # Get timeframes
        h1_bars = DEFAULT_H1_BARS
        m1_bars = DEFAULT_M1_BARS
        
    def _on_tune_clicked(self):
        # 收集参数
        params = self._collect_tune_params()
        if not params:
            return
            
        # 创建worker
        w = TuneWorker(
            symbol=params['symbol'],
            h1_bars=params['h1'],
            m1_bars=params['m1'],
            sl_range=params['sl_range'],
            tp_range=params['tp_range'],
            mr_range=params['mr_range'],
            qty_range=params['qty_range'],
            lot_modes=params['lot_modes'],
            max_workers=params['max_workers'],
            objective=params['objective'],
            search_mode=params['search_mode'],
            random_iter=params['random_iter'],
            strategy_cls=params['strategy_cls'],
            strategy_kwargs=params.get('strategy_kwargs', {}),
            entry_min_range=params.get('entry_min_range'),
            max_positions_range=params.get('max_positions_range'),
            enable_time_filter_values=params.get('enable_time_filter_values'),
            start_hour_range=params.get('start_hour_range'),
            end_hour_range=params.get('end_hour_range'),
            trade_direction_values=params.get('trade_direction_values'),
            save_callback=self._save_tuning_results_incremental  # 添加增量保存回调
        )
        
        # 连接信号
        w.signals.finished.connect(self._on_tune_finished)
        w.signals.progress.connect(self._on_tune_progress)
        w.signals.error.connect(self._on_error)
        
        # 初始化状态
        self._tuning_results = []
        self._tuning_save_worker = None
        self._tune_worker = w
        self._tuning_running = True
        self._set_tune_in_progress(True)
        
        # 启动worker
        self.thread_pool.start(w)
        
    def _save_tuning_results_incremental(self, result):
        """Incrementally save tuning results as they come in."""
        if not hasattr(self, '_tuning_results'):
            self._tuning_results = []
            
        if result is not None:
            self._tuning_results.append(result)
            
        # If we have enough results or this is a flush (result is None), save them
        if (result is None or len(self._tuning_results) >= 5) and self._tuning_results:
            # Create a copy of the results to save
            results_to_save = self._tuning_results.copy()
            self._tuning_results = []  # Clear the list
            
            # Save to database in a non-blocking way
            try:
                if results_to_save and len(results_to_save) > 0:
                    # Save to database directly in a separate thread
                    from utils import dbutil
                    import threading
                    from datetime import datetime
                    
                    def save_thread():
                        try:
                            strategy_name = self.tune_strategy.currentText()
                            symbol = results_to_save[0].get('symbol', 'XAUUSD')
                            objective = self.cmb_objective.currentText()
                            
                            dbutil.save_tuning_results(
                                results=results_to_save,
                                symbol=symbol,
                                strategy_name=strategy_name,
                                objective=objective,
                                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            )
                            
                            # Update UI on the main thread
                            QtCore.QMetaObject.invokeMethod(
                                self, 
                                '_on_save_progress', 
                                QtCore.Qt.QueuedConnection,
                                QtCore.Q_ARG(str, f"✅ 成功保存 {len(results_to_save)} 条结果到数据库")
                            )
                        except Exception as e:
                            import traceback
                            error_msg = f"❌ 保存结果到数据库时出错: {str(e)}\n{traceback.format_exc()}"
                            QtCore.QMetaObject.invokeMethod(
                                self, 
                                '_on_save_progress', 
                                QtCore.Qt.QueuedConnection,
                                QtCore.Q_ARG(str, error_msg)
                            )
                    
                    # Start the save thread
                    thread = threading.Thread(target=save_thread, daemon=True)
                    thread.start()
                    
            except Exception as e:
                self.tune_log.append(f"❌ 保存结果时出错: {str(e)}")
                import traceback
                self.tune_log.append(traceback.format_exc())
    
    @QtCore.Slot(str)
    def _on_save_progress(self, message):
        """Handle save progress updates from background threads."""
        if not hasattr(self, 'tune_log'):
            return
            
        # Update log
        self.tune_log.append(message)
        
        # Auto-scroll to bottom
        self.tune_log.verticalScrollBar().setValue(
            self.tune_log.verticalScrollBar().maximum()
        )
        
        # Parse progress percentage from message if available
        # Format example: "⏳ 优化进度: 10/60 (16.7%) | 已用: 0分0秒 | 预计剩余: 0分0秒"
        if '优化进度:' in message and '%' in message:
            try:
                # Extract percentage from message
                percent_str = message.split('(')[1].split('%')[0].strip()
                percent = int(float(percent_str))
                
                # Update progress bar
                self.tune_progress.setValue(percent)
                self.tune_progress.setFormat(f"{percent}%")
                
                # Force UI update
                QtWidgets.QApplication.processEvents()
            except (IndexError, ValueError) as e:
                # If parsing fails, just log the message without updating progress bar
                pass

    def _on_tune_finished(self, payload: Dict[str, Any]):
        results: List[Dict[str, Any]] = getattr(self, '_tuning_results', [])
        if not results:
            self.tune_log.append("❌ 无结果返回")
            self._set_tune_in_progress(False)
            return
            
        try:
            # 确保所有结果都已保存
            if hasattr(self, '_tuning_save_worker') and self._tuning_save_worker is not None:
                # 等待最后一个保存任务完成
                self.tune_log.append("等待最后一批结果保存完成...")
                return
                
            # 更新UI和结果表格
            self._update_results_table(results)
            self._populate_results_table_from_list(results)
            
            # 显示最佳结果
            if results:
                best = max(results, key=lambda x: float(x.get('score', 0) or 0))
                self.tune_log.append("\n✨ 调参完成，最优结果:")
                self.tune_log.append(f"得分: {best.get('score', 0):.2f}")
                self.tune_log.append(f"总收益: {best.get('total_return', 0):.2f}%")
                self.tune_log.append(f"胜率: {best.get('win_rate', 0):.1f}%")
                self.tune_log.append(f"盈亏比: {best.get('profit_factor', 0):.2f}")
                
                # 保存最优结果供后续使用
                self._last_tune_best = best
                self._last_tune_strategy_cls = self.strategy_combo.currentData()
                self.btn_apply_best.setEnabled(True)
            
            self.tune_log.append(f"✅ 调参完成，共 {len(results)} 组参数已评估")
            
        except Exception as e:
            self.tune_log.append(f"❌ 处理调参结果时出错: {e}")
            import traceback
            self.tune_log.append(traceback.format_exc())
            
        finally:
            self._set_tune_in_progress(False)
            
            # 清理
            if hasattr(self, '_tuning_results'):
                del self._tuning_results
            try:
                strategy_name = self.tune_strategy.currentText()
                strategy_cls = STRATEGY_REGISTRY.get(strategy_name, ConfigurableTimeframePinbarStrategy)
                deduped = self._dedup_results(results, strategy_cls)
                
                if len(deduped) != len(results):
                    self.tune_log.append(f"🧹 去重：原有 {len(results)} 条，保留 {len(deduped)} 条（按最优优先保留）")
                results = deduped
            except Exception as e:
                self.tune_log.append(f"⚠️ 去重时出错: {str(e)}")

            # 获取并显示最优结果
            objective = self.cmb_objective.currentText()
            best = self._best_result(results, objective)
            
            if best:
                self.tune_log.append("\n✨ 调参完成，最优参数组合:")
                param_keys = ["stop_loss_usd", "take_profit_usd", "min_range_usd", "qty", "lot_mode", 
                            "signal_tf", "backtest_tf", "entry_minute", "max_positions", 
                            "enable_time_filter", "start_hour", "end_hour", "trade_direction"]
                
                for k in param_keys:
                    if k in best and best[k] is not None:
                        self.tune_log.append(f" - {k}: {best[k]}")
                
                perf_metrics = ["total_return", "win_rate", "total_pnl", "profit_factor", "max_drawdown", "sharpe_ratio"]
                self.tune_log.append("\n📊 表现指标:")
                for m in perf_metrics:
                    if m in best and best[m] is not None:
                        self.tune_log.append(f" - {m}: {best[m]:.2f}")
                
                # 保存最优结果供“应用最优参数”使用
                self._last_tune_best = best
                self._last_tuning_results = results
                self._last_tune_strategy_cls = STRATEGY_REGISTRY.get(
                    self.tune_strategy.currentText(), 
                    ConfigurableTimeframePinbarStrategy
                )
                
                # 自动滚动到底部
                self.tune_log.verticalScrollBar().setValue(
                    self.tune_log.verticalScrollBar().maximum()
                )
                
                try:
                    # 保存结果到文件
                    try:
                        self._save_tuning_results(results)
                    except Exception as e:
                        self.tune_log.append(f"⚠️ 保存结果到文件时出错: {str(e)}")
                    
                    # 更新UI状态
                    self.btn_apply_best.setEnabled(True)
                    self.btn_stop_tune.setEnabled(False)
                    
                except Exception as e:
                    self.tune_log.append(f"❌ 处理优化结果时发生错误: {str(e)}")
                    import traceback
                    self.tune_log.append(traceback.format_exc())
                finally:
                    # 确保UI状态正确重置
                    self._set_tune_in_progress(False)
                    # 确保进度条重置
                    try:
                        self.tune_progress.setValue(0)
                        self.tune_progress.setFormat("准备就绪")
                    except Exception:
                        pass

    def _dedup_results(self, results: List[Dict[str, Any]], strategy_cls) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        is_timed = (strategy_cls is TimedEntryStrategy5m)
        for r in results:
            base_key = (
                round(float(r.get('stop_loss_usd', 0.0)), 6),
                round(float(r.get('take_profit_usd', 0.0)), 6),
                int(r.get('qty', 0)),
                str(r.get('lot_mode', '')),
                str(r.get('signal_tf', '')),
                str(r.get('backtest_tf', '')),
            )
            if is_timed:
                k = base_key + (
                    int(r.get('entry_minute', -1)),
                    int(r.get('max_positions', -1)),
                    bool(r.get('enable_time_filter', True)),
                    int(r.get('start_hour', -1)),
                    int(r.get('end_hour', -1)),
                    str(r.get('trade_direction', '')),
                )
            else:
                k = base_key + (
                    None if r.get('min_range_usd') is None else round(float(r.get('min_range_usd')), 6),
                )
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    def _on_apply_best_clicked(self):
        """应用最优参数到回测页面"""
        try:
            # 获取最优参数和策略类
            best = getattr(self, "_last_tune_best", None)
            strategy_cls = getattr(self, "_last_tune_strategy_cls", None)
            
            # 验证是否有可用的调参结果
            if not best or strategy_cls is None:
                QtWidgets.QMessageBox.information(self, "提示", "请先完成一次参数优化，以获取最优参数。")
                return
                
            # 记录应用了哪些参数
            applied_params = []
            
            # 1. 切换到对应的策略
            target_name = None
            for name, cls in STRATEGY_REGISTRY.items():
                if cls is strategy_cls:
                    target_name = name
                    break
                    
            if target_name:
                self.cmb_strategy.setCurrentText(target_name)
                self._update_run_tf_visibility()
                applied_params.append(f"策略: {target_name}")
            
            # 2. 设置通用参数
            param_mappings = [
                ("stop_loss_usd", self.dbl_sl, float, "止损"),
                ("take_profit_usd", self.dbl_tp, float, "止盈"),
                ("qty", self.spin_qty, int, "数量"),
                ("min_range_usd", self.dbl_mr, float, "最小波动")
            ]
            
            for param, widget, conv_type, name in param_mappings:
                if param in best and best[param] is not None:
                    try:
                        value = conv_type(best[param])
                        widget.setValue(value)
                        applied_params.append(f"{name}: {value}")
                    except (ValueError, TypeError) as e:
                        print(f"设置参数 {param} 时出错: {e}")
            
            # 3. 设置手数模式
            if "lot_mode" in best and best["lot_mode"] in ["fixed", "dynamic"]:
                try:
                    self.cmb_lotmode.setCurrentText(best["lot_mode"])
                    applied_params.append(f"手数模式: {best['lot_mode']}")
                except Exception as e:
                    print(f"设置手数模式时出错: {e}")
            
            # 4. 定时入场策略特有参数
            if strategy_cls is TimedEntryStrategy5m:
                timed_params = [
                    ("entry_minute", self.spin_entry_minute, int, "入场分钟"),
                    ("max_positions", self.spin_max_positions, int, "最大持仓")
                ]
                
                for param, widget, conv_type, name in timed_params:
                    if param in best and best[param] is not None:
                        try:
                            value = conv_type(best[param])
                            widget.setValue(value)
                            applied_params.append(f"{name}: {value}")
                        except (ValueError, TypeError) as e:
                            print(f"设置定时入场参数 {param} 时出错: {e}")
                
                # 处理时间过滤相关参数
                if "enable_time_filter" in best:
                    try:
                        enable = bool(best["enable_time_filter"])
                        self.chk_enable_time_filter.setChecked(enable)
                        applied_params.append(f"启用时间过滤: {'是' if enable else '否'}")
                    except Exception as e:
                        print(f"设置时间过滤时出错: {e}")
                
                time_params = [
                    ("start_hour", self.spin_start_hour, int, "开始时间"),
                    ("end_hour", self.spin_end_hour, int, "结束时间")
                ]
                
                for param, widget, conv_type, name in time_params:
                    if param in best and best[param] is not None:
                        try:
                            value = conv_type(best[param])
                            widget.setValue(value)
                            applied_params.append(f"{name}: {value}:00")
                        except (ValueError, TypeError) as e:
                            print(f"设置时间参数 {param} 时出错: {e}")
                
                # 交易方向
                if "trade_direction" in best and best["trade_direction"] in ["long", "short", "both"]:
                    try:
                        self.cmb_trade_direction.setCurrentText(best["trade_direction"])
                        applied_params.append(f"交易方向: {best['trade_direction']}")
                    except Exception as e:
                        print(f"设置交易方向时出错: {e}")
            
            # 5. 切换到回测页面
            try:
                if hasattr(self, 'tabs'):
                    self.tabs.setCurrentIndex(0)  # 切换到回测标签页
                
                # 显示应用成功的消息
                msg = "✅ 已应用最优参数到回测页面\n\n"
                msg += "\n".join(f"• {p}" for p in applied_params)
                
                # 在日志中显示详细参数
                self.run_log.append("\n✨ 已应用最优参数:")
                for p in applied_params:
                    self.run_log.append(f"   - {p}")
                
                # 显示状态消息
                self.status.showMessage("✨ 最优参数已成功应用", 5000)
                
                # 弹出提示框显示应用的参数
                QtWidgets.QMessageBox.information(
                    self, 
                    "参数应用成功",
                    msg,
                    QtWidgets.QMessageBox.Ok
                )
                
            except Exception as e:
                print(f"切换页面时出错: {e}")
                self.status.showMessage("参数应用完成，但发生了一些小问题", 3000)
                
        except Exception as e:
            error_msg = f"应用最优参数时发生错误: {str(e)}"
            print(error_msg)
            QtWidgets.QMessageBox.critical(
                self,
                "错误",
                error_msg,
                QtWidgets.QMessageBox.Ok
            )

    def _load_outputs_to_table(self):
        import os
        path = os.path.join('outputs','tuning_results.csv')
        self._populate_table_from_csv(path)

    def _populate_results_table_from_list(self, results: List[Dict[str, Any]]):
        """Populate results table directly from a list of result dicts with unified columns."""
        model = QtGui.QStandardItemModel(self)
        self.tbl_results.setModel(model)
        # Define strategy-aware column orders
        pinbar_cols = [
            'strategy','signal_tf','backtest_tf',
            'stop_loss_usd','take_profit_usd','min_range_usd','qty','lot_mode',
            'total_trades','winning_trades','losing_trades','win_rate','profit_factor','total_pnl','final_cash','total_return','total_fees','initial_cash','score',
            'trade_record_file'
        ]
        timed_cols = [
            'strategy','signal_tf','backtest_tf',
            'stop_loss_usd','take_profit_usd','qty','lot_mode',
            'entry_minute','max_positions','enable_time_filter','start_hour','end_hour','trade_direction',
            'total_trades','winning_trades','losing_trades','win_rate','profit_factor','total_pnl','final_cash','total_return','total_fees','initial_cash','score',
            'trade_record_file'
        ]
        try:
            import pandas as _pd
            df = _pd.DataFrame(results)
            # Decide columns strictly by currently selected strategy in Tune tab
            selected_cls = STRATEGY_REGISTRY.get(self.tune_strategy.currentText())
            selected_cls_name = selected_cls.__name__ if selected_cls else ''
            if selected_cls_name == 'TimedEntryStrategy5m':
                cols = timed_cols
                # Filter rows to only timed strategy
                if 'strategy' in df.columns:
                    df = df[df['strategy'] == 'TimedEntryStrategy5m']
            else:
                cols = pinbar_cols
                # Filter rows to only pinbar strategies
                if 'strategy' in df.columns:
                    df = df[df['strategy'].isin(['ConfigurableTimeframePinbarStrategy'])]
            # Ensure all expected columns exist
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df = df[cols]
            model.setHorizontalHeaderLabels(cols)
            # Chunked append to avoid blocking the UI; yield to event loop periodically
            CHUNK = 200
            total_rows = len(df)
            for r in range(total_rows):
                vals = ["" if _pd.isna(df.iloc[r, c]) else str(df.iloc[r, c]) for c in range(len(cols))]
                items = [QtGui.QStandardItem(v) for v in vals]
                model.appendRow(items)
                if (r + 1) % CHUNK == 0:
                    try:
                        QtWidgets.QApplication.processEvents()
                    except Exception:
                        pass
        except Exception as e:
            self.tune_log.append(f"⚠️ 结果表渲染失败: {e}")

    def _refresh_combo_count(self):
        try:
            p = self._collect_tune_params()
            mode = self.cmb_search_mode.currentText()
            if str(mode).lower().startswith("random"):
                self.lbl_combo_count.setText(f"组合数: 随机 {int(self.spin_random_iter.value())} 次")
            else:
                cnt = self._estimate_combo_count(p)
                self.lbl_combo_count.setText(f"组合数: {cnt}")
        except Exception:
            pass

    def _connect_combo_count_signals(self):
        # 基础范围
        for w in [self.sl_min, self.sl_max, self.sl_step, self.tp_min, self.tp_max, self.tp_step,
                  self.mr_min, self.mr_max, self.mr_step, self.qty_min, self.qty_max, self.qty_step,
                  self.em_min, self.em_max, self.em_step, self.mp_min, self.mp_max, self.mp_step,
                  self.sh_min, self.sh_max, self.sh_step, self.eh_min, self.eh_max, self.eh_step,
                  self.spin_random_iter, self.spin_workers]:
            try:
                w.valueChanged.connect(self._refresh_combo_count)
            except Exception:
                pass
        # 复选项
        for c in [self.chk_fixed, self.chk_dynamic, self.chk_etf_true, self.chk_etf_false,
                  self.chk_dir_long, self.chk_dir_short, self.chk_dir_both, self.chk_dir_alt]:
            try:
                c.toggled.connect(self._refresh_combo_count)
            except Exception:
                pass
        # 策略选择 & 搜索模式
        try:
            self.tune_strategy.currentTextChanged.connect(self._refresh_combo_count)
            self.cmb_search_mode.currentTextChanged.connect(self._refresh_combo_count)
        except Exception:
            pass

    def _on_stop_tune(self):
        try:
            if hasattr(self, '_active_tune_worker') and self._active_tune_worker is not None:
                self._active_tune_worker.stop_event.set()
                self.tune_log.append("⛔ 已请求停止优化，正在等待当前任务结束...")
                try:
                    self.btn_stop_tune.setEnabled(False)
                except Exception:
                    pass
        except Exception:
            pass

    def _estimate_combo_count(self, p: Dict[str, Any]) -> int:
        """Estimate total combinations for current ranges and mode (Grid only)."""
        # Build counts for each grid
        def rng_count(triple: tuple | None, clamp=None, step_is_int=True):
            if not triple:
                return 1
            a, b, s = triple
            if step_is_int:
                a = int(a); b = int(b); s = max(1, int(s))
                return max(0, ((b - a) // s) + 1)
            else:
                a = float(a); b = float(b); s = float(s)
                if s <= 0: return 0
                n = int((b - a) / s) + 1
                return max(0, n)
        slc = rng_count(p.get('sl_range'), step_is_int=False)
        tpc = rng_count(p.get('tp_range'), step_is_int=False)
        mrc = rng_count(p.get('mr_range'), step_is_int=False)
        qtyc = rng_count(p.get('qty_range'), step_is_int=True)
        lmc = max(1, len(p.get('lot_modes') or []))
        emc = rng_count(p.get('entry_min_range'), step_is_int=True)
        mpc = rng_count(p.get('max_positions_range'), step_is_int=True)
        etfc = len(p.get('enable_time_filter_values') or [None])
        shc = rng_count(p.get('start_hour_range'), step_is_int=True)
        ehc = rng_count(p.get('end_hour_range'), step_is_int=True)
        tdc = len(p.get('trade_direction_values') or [None])
        total = slc * tpc * mrc * qtyc * lmc * emc * mpc * etfc * shc * ehc * tdc
        return int(total)

    def _open_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择CSV文件", "outputs", "CSV Files (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path)
            dlg = QtWidgets.QDialog(self); dlg.setWindowTitle(path); dlg.resize(1000,700)
            v = QtWidgets.QVBoxLayout(dlg); view = QtWidgets.QTableView(); model = QtGui.QStandardItemModel()
            model.setHorizontalHeaderLabels(list(df.columns))
            for r in range(len(df)):
                model.appendRow([QtGui.QStandardItem(str(df.iloc[r,c])) for c in range(len(df.columns))])
            view.setModel(model); view.setSortingEnabled(True); view.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
            v.addWidget(view); dlg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"❌ 无法打开CSV: {e}")

    def _open_dir(self, directory: str):
        import os
        os.makedirs(directory, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.abspath(directory)))

    def _refresh_data(self):
        self.status.showMessage("🔄 数据已刷新", 2000); self._load_outputs_to_table()

    def _on_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "错误", msg); self.status.showMessage(msg, 5000)
        if hasattr(self, 'btn_run'): self.btn_run.setEnabled(True); self.btn_run.setText("🚀 启动回测")
        if hasattr(self, 'btn_tune'): self.btn_tune.setEnabled(True); self.btn_tune.setText("⚙️ 启动参数优化")

    def _show_about_dialog(self):
        QtWidgets.QMessageBox.information(self, "关于", "MT5 Premium Suite - Professional Backtesting Platform")
