from typing import Dict, Any, Tuple
import json

import pandas as pd

# Use absolute imports to avoid relative import issues
from mt5_backtest_multi.utils.tools import initialize_mt5, load_data, preprocess_data
from mt5_backtest_multi.utils import db as dbutil
from mt5_backtest_multi.strategies.pinbar_configurable import ConfigurableTimeframePinbarStrategy
from mt5_backtest_multi.strategies.base import BaseStrategy
from mt5_backtest_multi.config import (
    DEFAULT_SYMBOL,
    DEFAULT_H1_BARS,
    DEFAULT_M1_BARS,
    DEFAULT_STOP_LOSS_USD,
    DEFAULT_TAKE_PROFIT_USD,
    DEFAULT_MIN_RANGE_USD,
    DEFAULT_QTY,
    DEFAULT_LOT_MODE,
)


def run_backtest_pipeline(
    symbol: str = DEFAULT_SYMBOL,
    h1_bars: int = DEFAULT_H1_BARS,
    m1_bars: int = DEFAULT_M1_BARS,
    strategy_params: Dict[str, Any] | None = None,
    strategy_cls=ConfigurableTimeframePinbarStrategy,
    reuse_db: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], BaseStrategy | None]:
    """Run a single backtest with provided parameters.

    Returns (trade_records_df, stats, strategy)
    """
    if strategy_params is None:
        strategy_params = {
            "stop_loss_usd": DEFAULT_STOP_LOSS_USD,
            "take_profit_usd": DEFAULT_TAKE_PROFIT_USD,
            "min_range_usd": DEFAULT_MIN_RANGE_USD,
            "qty": DEFAULT_QTY,
            "lot_mode": DEFAULT_LOT_MODE,
        }

    # 1) Try DB cache first (default enabled)
    if reuse_db:
        try:
            strategy_name = getattr(strategy_cls, '__name__', str(strategy_cls))
            # Prefer param_key-based lookup to ensure exact combo match
            try:
                sig_tf = strategy_params.get('signal_tf') if strategy_params else None
                bt_tf = strategy_params.get('backtest_tf') if strategy_params else None
                tfl = [x for x in [sig_tf, bt_tf] if x]
                pk = dbutil._build_param_key(symbol, strategy_name, strategy_params or {}, tfl)
                row = dbutil.find_run_by_param_key(param_key=pk, is_tuning=0)
            except Exception:
                row = None
            if row is None:
                # Fallback to params_json exact match (legacy)
                row = dbutil.find_latest_run(symbol, strategy_name, strategy_params)
            if row is not None:
                # If timeframes are part of params, they are already matched by params_json.
                # For extra safety, when explicit tfs exist on the row, ensure they match requested ones.
                try:
                    exp_sig = strategy_params.get('signal_tf') if strategy_params else None
                    exp_bt = strategy_params.get('backtest_tf') if strategy_params else None
                    row_sig = row.get('signal_tf')
                    row_bt = row.get('backtest_tf')
                    if (exp_sig is not None or row_sig is not None) and str(exp_sig) != str(row_sig):
                        raise ValueError('tf_mismatch')
                    if (exp_bt is not None or row_bt is not None) and str(exp_bt) != str(row_bt):
                        raise ValueError('tf_mismatch')
                except ValueError:
                    # mismatch -> treat as cache miss
                    raise
                # Load trades and stats from DB
                trades_df = dbutil.get_trades(int(row['id']))
                stats = json.loads(row.get('stats_json') or '{}')
                try:
                    stats['source'] = 'cache'
                    stats['db_saved'] = True
                    stats['db_run_id'] = int(row.get('id'))
                    if 'param_key' in row:
                        stats['param_key'] = row.get('param_key')
                    stats['db_params_json'] = row.get('params_json')
                except Exception:
                    pass
                # Return a lightweight strategy instance for UI fields when possible
                try:
                    strategy = strategy_cls(**strategy_params)
                    setattr(strategy, 'suppress_autosave', True)
                except Exception:
                    strategy = None  # Non-critical
                return trades_df, stats, strategy
        except Exception:
            # Cache miss or error -> continue to run normally
            pass

    if not initialize_mt5():
        raise RuntimeError("MT5 初始化失败")

    # Load multi-timeframe data (H1, M30, M15, M5, M1)
    df_1h, df_30m, df_15m, df_5m, df_1m = load_data(symbol, h1_bars=h1_bars, m1_bars=m1_bars)
    if any(x is None for x in [df_1h, df_30m, df_15m, df_5m, df_1m]):
        raise RuntimeError("数据加载失败")

    # Preprocess all timeframes
    df_1h, df_30m, df_15m, df_5m, df_1m = preprocess_data(df_1h, df_30m, df_15m, df_5m, df_1m)
    if any(x is None for x in [df_1h, df_30m, df_15m, df_5m, df_1m]):
        raise RuntimeError("数据预处理失败")

    strategy = strategy_cls(**strategy_params)
    # Avoid CSV autosave at strategy layer; we'll persist centrally to DB
    try:
        setattr(strategy, 'suppress_autosave', True)
    except Exception:
        pass
    # Map dataframes for flexible timeframe selection
    dfs = {
        '1h': df_1h,
        '30m': df_30m,
        '15m': df_15m,
        '5m': df_5m,
        '1m': df_1m,
    }
    # Choose signal/backtest frames if the strategy supports configurable timeframes
    if hasattr(strategy, 'signal_tf') and hasattr(strategy, 'backtest_tf'):
        df_signal = dfs.get(getattr(strategy, 'signal_tf', '1h'), df_1h)
        df_bt = dfs.get(getattr(strategy, 'backtest_tf', '1m'), df_1m)
    else:
        df_signal = df_1h
        df_bt = df_1m
    strategy.preprocess_hourly_signals(df_signal)
    strategy.run_backtest(df_bt)
    stats = strategy.get_stats()
    try:
        stats['source'] = 'fresh'
    except Exception:
        pass

    # Ensure DataFrame output
    trades_df = pd.DataFrame(strategy.trade_records)
    # Persist this run into SQLite for later querying (only for fresh runs)
    try:
        run_id = dbutil.save_backtest_run(
            symbol=symbol,
            h1_bars=h1_bars,
            m1_bars=m1_bars,
            strategy_name=getattr(strategy_cls, '__name__', str(strategy_cls)),
            params=strategy_params,
            stats=stats,
            trades_df=trades_df,
        )
        try:
            stats['db_saved'] = True
            # Build and attach param_key for transparency
            try:
                sig_tf = strategy_params.get('signal_tf') if strategy_params else None
                bt_tf = strategy_params.get('backtest_tf') if strategy_params else None
                tfl = [x for x in [sig_tf, bt_tf] if x]
                pkey = dbutil._build_param_key(symbol, getattr(strategy_cls, '__name__', str(strategy_cls)), strategy_params or {}, tfl)
                stats['param_key'] = pkey
                # Ensure we report the earliest run_id bound to this key
                row_pk = dbutil.find_run_by_param_key(param_key=pkey, is_tuning=0)
                if row_pk is not None:
                    stats['db_run_id'] = int(row_pk.get('id'))
                    stats['db_params_json'] = row_pk.get('params_json')
                else:
                    stats['db_run_id'] = int(run_id)
                    stats['db_params_json'] = json.dumps(strategy_params or {}, ensure_ascii=False, sort_keys=True)
            except Exception:
                stats['db_run_id'] = int(run_id)
                try:
                    stats['db_params_json'] = json.dumps(strategy_params or {}, ensure_ascii=False, sort_keys=True)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception as e:
        # Persistence failure should not break the flow; annotate for UI
        try:
            stats['db_saved'] = False
            stats['db_error'] = str(e)
        except Exception:
            pass
    return trades_df, stats, strategy
