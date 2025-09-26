import itertools
import concurrent.futures
from typing import Iterable, Dict, Any, List
from threading import Event

import pandas as pd

from .strategies.pinbar_configurable import ConfigurableTimeframePinbarStrategy
import os
import random


def _run_param_combination(args) -> Dict[str, Any]:
    """Worker to execute a single parameter combination.

    args: tuple(
        stop_loss, take_profit, min_range, qty, lot_mode,
        entry_minute, max_positions,
        enable_time_filter, start_hour, end_hour, trade_direction,
        dfs: Dict[str, Any],
        strategy_cls,
        strategy_kwargs: Dict[str, Any] | None,
    )
    """
    stop_loss, take_profit, min_range, qty, lot_mode, entry_minute, max_positions, enable_time_filter, start_hour, end_hour, trade_direction, dfs, strategy_cls, strategy_kwargs = args
    strategy_kwargs = dict(strategy_kwargs or {})
    # Inject per-combination TimedEntry grids if present
    if entry_minute is not None:
        strategy_kwargs['entry_minute'] = int(entry_minute)
    if max_positions is not None:
        strategy_kwargs['max_positions'] = int(max_positions)
    if enable_time_filter is not None:
        strategy_kwargs['enable_time_filter'] = bool(enable_time_filter)
    if start_hour is not None:
        strategy_kwargs['start_hour'] = int(start_hour)
    if end_hour is not None:
        strategy_kwargs['end_hour'] = int(end_hour)
    if trade_direction is not None:
        strategy_kwargs['trade_direction'] = str(trade_direction)
    strategy = strategy_cls(
        stop_loss_usd=stop_loss,
        take_profit_usd=take_profit,
        # 对于无此参数的策略（如 TimedEntry），该参数将被忽略
        min_range_usd=min_range,
        qty=qty,
        lot_mode=lot_mode,
        **strategy_kwargs,
    )
    # Select dataframes based on strategy kwargs if present
    signal_tf = getattr(strategy, 'signal_tf', '1h')
    backtest_tf = getattr(strategy, 'backtest_tf', '1m')
    df_signal = dfs.get(signal_tf)
    if df_signal is None:
        df_signal = dfs.get('1h')
    df_bt = dfs.get(backtest_tf)
    if df_bt is None:
        df_bt = dfs.get('1m')
    strategy.preprocess_hourly_signals(df_signal.copy())
    # Avoid duplicate CSV: let tuner handle saving, suppress autosave in strategy
    try:
        setattr(strategy, 'suppress_autosave', True)
    except Exception:
        pass
    strategy.run_backtest(df_bt.copy())
    stats = strategy.get_stats()
    # Normalize final_cash to avoid NaN: fallback to initial_cash + total_pnl
    try:
        fc = stats.get('final_cash') if isinstance(stats, dict) else None
        tp = float(stats.get('total_pnl', 0.0)) if isinstance(stats, dict) else 0.0
        if fc is None or (isinstance(fc, float) and (fc != fc)):
            try:
                fc = float(getattr(strategy, 'initial_cash', 0.0)) + float(tp)
            except Exception:
                fc = float(tp)
        stats['final_cash'] = float(fc)
    except Exception:
        try:
            stats['final_cash'] = float(getattr(strategy, 'initial_cash', 0.0))
        except Exception:
            pass
    # Derive profit_factor from trade records if possible
    profit_factor = None
    try:
        losses = 0.0
        gains = 0.0
        for tr in getattr(strategy, 'trade_records', []):
            pnl = float(tr.get('net_pnl', 0.0))
            if pnl >= 0:
                gains += pnl
            else:
                losses += -pnl
        if losses > 0:
            profit_factor = gains / losses
        elif gains > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0.0
    except Exception:
        profit_factor = None
    os.makedirs('outputs', exist_ok=True)
    # Strategy-aware trade record filename (by class name to avoid import issues)
    cls_name = type(strategy).__name__
    if cls_name == 'TimedEntryStrategy5m':
        em = int(getattr(strategy, 'entry_minute', entry_minute if entry_minute is not None else 20))
        mp = int(getattr(strategy, 'max_positions', max_positions if max_positions is not None else 3))
        trade_record_file = os.path.join(
            'outputs',
            f"trade_records_timed5m_em{em}_mp{mp}_sl{stop_loss}_tp{take_profit}_q{qty}_{lot_mode}.csv"
        )
    elif cls_name == 'ConfigurableTimeframePinbarStrategy':
        trade_record_file = os.path.join(
            'outputs',
            f"trade_records_cfgpinbar_{signal_tf}_{backtest_tf}_sl{stop_loss}_tp{take_profit}_mr{min_range}_q{qty}_{lot_mode}.csv"
        )
    else:
        # Default: Other strategies (including legacy pinbar)
        trade_record_file = os.path.join(
            'outputs',
            f"trade_records_pinbar_sl{stop_loss}_tp{take_profit}_mr{min_range}_q{qty}_{lot_mode}.csv"
        )
    # Save trade records robustly (Windows may lock files if opened externally)
    try:
        strategy.save_trade_records(trade_record_file)
    except PermissionError:
        # Retry with a unique suffix
        import time
        uniq = f"{int(time.time()*1000)}_{os.getpid()}"
        base, ext = os.path.splitext(trade_record_file)
        fallback = f"{base}_{uniq}{ext or '.csv'}"
        try:
            strategy.save_trade_records(fallback)
            trade_record_file = fallback
        except Exception:
            # As a last resort, skip saving but continue returning stats
            pass
    except Exception:
        # Non-fatal: skip saving but continue
        pass
    # 统一结果字段，便于表格展示
    result: Dict[str, Any] = {
        'strategy': strategy_cls.__name__,
        'signal_tf': signal_tf,
        'backtest_tf': backtest_tf,
        'stop_loss_usd': float(stop_loss),
        'take_profit_usd': float(take_profit),
        'min_range_usd': float(min_range) if hasattr(strategy, 'min_range_usd') else None,
        'qty': int(qty),
        'lot_mode': lot_mode,
        'total_trades': int(stats['total_trades']),
        'winning_trades': int(stats['winning_trades']),
        'losing_trades': int(stats['losing_trades']),
        'win_rate': float(stats['win_rate']),
        'total_pnl': float(stats['total_pnl']),
        'final_cash': float(stats.get('final_cash', getattr(strategy, 'initial_cash', 0.0))),
        'total_return': float(stats['total_return']),
        'profit_factor': profit_factor if profit_factor is not None else None,
        'total_fees': float(stats.get('total_fees', 0.0)) if isinstance(stats, dict) else None,
        'trade_record_file': trade_record_file,
        'initial_cash': float(strategy.initial_cash),
    }
    # Include commission so it is persisted in params_json
    try:
        commission = None
        if isinstance(strategy_kwargs, dict) and 'commission_per_10lots_usd' in strategy_kwargs:
            commission = float(strategy_kwargs.get('commission_per_10lots_usd'))
        else:
            commission = float(getattr(strategy, 'commission_per_10lots_usd', None))
        if commission is not None:
            result['commission_per_10lots_usd'] = commission
    except Exception:
        pass
    # TimedEntryStrategy 专属字段
    if cls_name == 'TimedEntryStrategy5m':
        result['entry_minute'] = int(getattr(strategy, 'entry_minute', 20))
        result['max_positions'] = int(getattr(strategy, 'max_positions', 3))
        # 对于无 min_range 的策略，明确置 None，避免误解
        result['min_range_usd'] = None
        # 追加定时策略的其余可调参数，便于报告
        try:
            result['enable_time_filter'] = bool(getattr(strategy, 'enable_time_filter', True))
        except Exception:
            result['enable_time_filter'] = None
        try:
            result['start_hour'] = int(getattr(strategy, 'start_hour', 8))
            result['end_hour'] = int(getattr(strategy, 'end_hour', 20))
        except Exception:
            result['start_hour'] = None; result['end_hour'] = None
        try:
            result['trade_direction'] = str(getattr(strategy, 'trade_direction', 'Both'))
        except Exception:
            result['trade_direction'] = None
    return result


def tune_parameters(
    stop_loss_values: Iterable[float],
    take_profit_values: Iterable[float],
    min_range_values: Iterable[float],
    dfs: Dict[str, pd.DataFrame] | None = None,
    df_1h: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    qty_values: Iterable[int] | None = None,
    lot_mode_values: Iterable[str] | None = None,
    max_workers: int | None = None,
    use_process: bool = False,  # Default to False as threads are generally better for I/O bound tasks
    strategy_cls=ConfigurableTimeframePinbarStrategy,
    strategy_kwargs: Dict[str, Any] | None = None,
    # TimedEntry-specific optional grids
    entry_minute_values: Iterable[int] | None = None,
    max_positions_values: Iterable[int] | None = None,
    enable_time_filter_values: Iterable[bool] | None = None,
    start_hour_values: Iterable[int] | None = None,
    end_hour_values: Iterable[int] | None = None,
    trade_direction_values: Iterable[str] | None = None,
    stop_event: Event | None = None,
    progress_callback=None,  # Add progress callback parameter
):
    """并行执行参数调优，返回结果列表。"""
    # Backward compatibility for callers passing df_1h/df_1m only
    if dfs is None:
        dfs = {}
        if df_1h is not None:
            dfs['1h'] = df_1h
        if df_1m is not None:
            dfs['1m'] = df_1m
    if qty_values is None:
        qty_values = [10]
    if lot_mode_values is None:
        lot_mode_values = ['fixed']
    # Default singletons for optional grids
    if entry_minute_values is None:
        entry_minute_values = [None]
    if max_positions_values is None:
        max_positions_values = [None]
    if enable_time_filter_values is None:
        enable_time_filter_values = [None]
    if start_hour_values is None:
        start_hour_values = [None]
    if end_hour_values is None:
        end_hour_values = [None]
    if trade_direction_values is None:
        trade_direction_values = [None]
    param_combinations = list(itertools.product(
        stop_loss_values, take_profit_values, min_range_values,
        qty_values, lot_mode_values,
        entry_minute_values, max_positions_values,
        enable_time_filter_values, start_hour_values, end_hour_values, trade_direction_values
    ))
    param_args = [
        (sl, tp, mr, q, lm, em, mp, etf, sh, eh, td, dfs, strategy_cls, strategy_kwargs)
        for (sl, tp, mr, q, lm, em, mp, etf, sh, eh, td) in param_combinations
    ]
    # 计算总组合数用于进度跟踪
    total_combinations = len(param_combinations)
    if progress_callback:
        progress_callback(0, total_combinations, 
                         current_params="正在准备参数组合...")
    
    # 准备共享数据
    dfs = dfs or {}
    if df_1h is not None:
        dfs['1h'] = df_1h
    if df_1m is not None:
        dfs['1m'] = df_1m
        
    # 复制策略参数以避免共享状态问题
    strategy_kwargs = dict(strategy_kwargs or {})
    param_combinations = list(itertools.product(
        stop_loss_values, take_profit_values, min_range_values,
        qty_values, lot_mode_values,
        entry_minute_values, max_positions_values,
        enable_time_filter_values, start_hour_values, end_hour_values, trade_direction_values
    ))
    
    # 准备参数元组列表，并添加进度跟踪
    param_args = []
    for i, params in enumerate(param_combinations):
        param_args.append((*params, dfs, strategy_cls, strategy_kwargs))
        
        # 每10个组合或最后一个组合时更新进度
        if progress_callback and (i % 10 == 0 or i == len(param_combinations) - 1):
            current_params = {
                'sl': params[0],
                'tp': params[1],
                'mr': params[2],
                'qty': params[3],
                'lot_mode': params[4],
            }
            progress_callback(i, total_combinations, current_params=current_params)
            
            # 检查是否请求停止
            if stop_event and stop_event.is_set():
                print("检测到停止信号，正在终止调优...")
                return []
    
    # Fallbacks for GUI/thread context on Windows
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if max_workers <= 1 or (stop_event is not None and not use_process):
        results: List[Dict[str, Any]] = []
        for args in param_args:
            if stop_event is not None and stop_event.is_set():
                break
            results.append(_run_param_combination(args))
        return results
    
    results = []
    try:
        chunk_size = max(1, len(param_args) // (max_workers or 4))  # 每个工作项处理一个参数块
        
        # 分块处理参数，减少任务调度开销
        for i in range(0, len(param_args), chunk_size):
            chunk = param_args[i:i + chunk_size]
            
            # 检查停止信号
            if stop_event and stop_event.is_set():
                print("检测到停止信号，正在终止调优...")
                break
                
            # 处理当前块
            chunk_results = run_param_args(
                chunk,
                max_workers=max_workers,
                use_process=use_process,
                stop_event=stop_event,
            )
            
            # 收集结果
            results.extend(chunk_results)
            
            # 更新进度
            if progress_callback:
                current = min(i + len(chunk), total_combinations)
                progress_callback(current, total_combinations, 
                               current_params=f"已处理 {current}/{total_combinations} 个组合")
        
        # 最终进度更新
        if progress_callback:
            progress_callback(total_combinations, total_combinations, 
                           current_params="优化完成")
    except Exception as e:
        import traceback
        print(f"优化过程中发生错误: {e}\n{traceback.format_exc()}")
        if progress_callback:
            progress_callback(len(results), total_combinations, 
                           error=f"优化过程中发生错误: {str(e)}")
    return results


def run_param_args(
    param_args: list[tuple],
    *,
    max_workers: int | None = None,
    use_process: bool = False,
    stop_event: Event | None = None,
) -> List[Dict[str, Any]]:
    """Execute a preselected list of parameter combination argument tuples.

    Each element in param_args must match the signature consumed by _run_param_combination.
    Mirrors the execution behavior of tune_parameters with thread/process support and stop_event for threads.
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if max_workers <= 1 or (stop_event is not None and not use_process):
        results: List[Dict[str, Any]] = []
        for args in param_args:
            if stop_event is not None and stop_event.is_set():
                break
            results.append(_run_param_combination(args))
        return results
    Executor = concurrent.futures.ProcessPoolExecutor if use_process else concurrent.futures.ThreadPoolExecutor
    if not use_process:
        results: List[Dict[str, Any]] = []
        with Executor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_param_combination, args) for args in param_args]
            try:
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception:
                        pass
                    if stop_event is not None and stop_event.is_set():
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break
            finally:
                return results
    with Executor(max_workers=max_workers) as executor:
        return list(executor.map(_run_param_combination, param_args))


def random_search_parameters(
    df_1h: pd.DataFrame,
    df_1m: pd.DataFrame,
    sl_range: tuple[float, float],
    tp_range: tuple[float, float],
    mr_range: tuple[float, float],
    qty_range: tuple[int, int] | None = None,
    lot_mode_values: List[str] | None = None,
    n_iter: int = 30,
    seed: int | None = None,
    max_workers: int | None = None,
    use_process: bool = True,
) -> List[Dict[str, Any]]:
    """随机搜索超参数。

    - sl_range, tp_range, mr_range: (min, max)
    - qty_range: (min, max), 将按 10 的步长取整到 >=10 的值
    - lot_mode_values: e.g., ['fixed','dynamic']
    - n_iter: 迭代次数
    - seed: 随机种子
    """
    if seed is not None:
        random.seed(seed)
    if qty_range is None:
        qty_range = (10, 10)
    if lot_mode_values is None or not lot_mode_values:
        lot_mode_values = ['fixed']

    tasks = []
    for _ in range(n_iter):
        sl = random.uniform(sl_range[0], sl_range[1])
        tp = random.uniform(tp_range[0], tp_range[1])
        mr = random.uniform(mr_range[0], mr_range[1])
        q = max(10, int(round(random.uniform(qty_range[0], qty_range[1]) / 10.0) * 10))
        lm = random.choice(lot_mode_values)
        tasks.append((sl, tp, mr, q, lm, df_1h, df_1m))

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if max_workers <= 1:
        return [
            _run_param_combination(args)
            for args in tasks
        ]
    Executor = concurrent.futures.ProcessPoolExecutor if use_process else concurrent.futures.ThreadPoolExecutor
    with Executor(max_workers=max_workers) as executor:
        return list(executor.map(_run_param_combination, tasks))


def random_search_parameters_full(
    dfs: Dict[str, pd.DataFrame],
    sl_range: tuple[float, float],
    tp_range: tuple[float, float],
    mr_range: tuple[float, float] | None,
    qty_range: tuple[int, int] | None,
    lot_mode_values: List[str] | None,
    n_iter: int,
    strategy_cls=ConfigurableTimeframePinbarStrategy,
    strategy_kwargs: Dict[str, Any] | None = None,
    # Timed-entry
    entry_min_range: tuple[int, int, int] | None = None,
    max_positions_range: tuple[int, int, int] | None = None,
    enable_time_filter_values: Iterable[bool] | None = None,
    start_hour_range: tuple[int, int, int] | None = None,
    end_hour_range: tuple[int, int, int] | None = None,
    trade_direction_values: Iterable[str] | None = None,
    stop_event: Event | None = None,
) -> List[Dict[str, Any]]:
    """Random search that supports timed-entry grids and multi-tf dfs."""
    if qty_range is None:
        qty_range = (10, 10)
    if lot_mode_values is None or not lot_mode_values:
        lot_mode_values = ['fixed']

    def _sample_int_range(rng: tuple[int, int, int] | None, clamp: tuple[int, int] | None = None) -> int | None:
        if rng is None:
            return None
        a, b, s = int(rng[0]), int(rng[1]), max(1, int(rng[2]))
        import random as _rand
        val = _rand.randrange(a, b + 1, s)
        if clamp:
            val = max(clamp[0], min(clamp[1], val))
        return val

    import random as _rand
    results: List[Dict[str, Any]] = []
    for _ in range(int(n_iter)):
        if stop_event is not None and stop_event.is_set():
            break
        sl = _rand.uniform(sl_range[0], sl_range[1])
        tp = _rand.uniform(tp_range[0], tp_range[1])
        mr = None
        if mr_range is not None:
            mr = _rand.uniform(mr_range[0], mr_range[1])
        q = max(10, int(round(_rand.uniform(qty_range[0], qty_range[1]) / 10.0) * 10))
        lm = _rand.choice(lot_mode_values)
        em = _sample_int_range(entry_min_range, (0, 59))
        mp = _sample_int_range(max_positions_range, (1, 9999))
        etf = None if enable_time_filter_values is None else _rand.choice(list(enable_time_filter_values))
        sh = _sample_int_range(start_hour_range, (0, 23))
        eh = _sample_int_range(end_hour_range, (0, 23))
        td = None if trade_direction_values is None else _rand.choice(list(trade_direction_values))
        args = (sl, tp, mr, q, lm, em, mp, etf, sh, eh, td, dfs, strategy_cls, strategy_kwargs)
        results.append(_run_param_combination(args))
    return results
