import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta


def initialize_mt5(timeout_sec: int = 10):
    """Initialize MT5 with a timeout to avoid indefinite hang.

    Returns True if initialized, False otherwise.
    """
    import threading
    ok = {'v': False}
    def _do_init():
        try:
            ok['v'] = bool(mt5.initialize())
        except Exception:
            ok['v'] = False
    t = threading.Thread(target=_do_init, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        print(f"[MT5] 初始化超时（>{timeout_sec}s）")
        try:
            # Best effort shutdown in case of partial init
            mt5.shutdown()
        except Exception:
            pass
        return False
    if not ok['v']:
        print("初始化 MT5 失败")
        return False
    print("[MT5] 初始化成功")
    return True


def load_data(symbol, h1_bars=2000, m30_bars=4000, m15_bars=8000, m5_bars=24000, m1_bars=80000):
    """Load rates up to server's today 00:00, with bars windows per timeframe:
    1h=h1_bars, 30m=m30_bars, 15m=m15_bars, 5m=m5_bars, 1m=m1_bars.
    """
    # Determine cutoff at MT5 server day 00:00 in UTC (MT5 times are UTC-based)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or getattr(tick, 'time', None) is None:
        return None, None, None, None, None
    now_server_utc = datetime.utcfromtimestamp(int(tick.time))
    cutoff = datetime(now_server_utc.year, now_server_utc.month, now_server_utc.day)  # UTC midnight
    def seconds_per_bar(tf):
        return {
            mt5.TIMEFRAME_H1: 60*60,
            mt5.TIMEFRAME_M30: 30*60,
            mt5.TIMEFRAME_M15: 15*60,
            mt5.TIMEFRAME_M5: 5*60,
            mt5.TIMEFRAME_M1: 60,
        }.get(tf, 60)

    def fetch_last_n_before_cutoff(tf, bars: int):
        # Expand window until we get at least 'bars' before cutoff, then trim exactly 'bars'
        spb = seconds_per_bar(tf)
        span_mult = 3  # start with 3x window
        for _ in range(4):  # up to 4 expansions
            try:
                span = max(1, int(bars) * span_mult) * spb + 5*spb
                start = cutoff - timedelta(seconds=span)
                arr = mt5.copy_rates_range(symbol, tf, start, cutoff)
                if arr is None or len(arr) == 0:
                    span_mult *= 2
                    continue
                # Filter strictly before cutoff (UTC seconds) using numpy mask to keep structured dtypes
                import numpy as _np
                cutoff_ts = int(cutoff.replace(tzinfo=None).timestamp())
                try:
                    mask = arr['time'] < cutoff_ts
                    arr = arr[mask]
                except Exception:
                    # Fallback: if not structured, try list filtering
                    arr = [r for r in arr if int(r['time']) < cutoff_ts]
                if len(arr) >= int(bars):
                    arr = arr[-int(bars):]
                    return arr
                # not enough -> expand
                span_mult *= 2
            except Exception:
                span_mult *= 2
                continue
        # Return whatever we could get (may be less than bars)
        try:
            span = max(1, int(bars)) * spb + 5*spb
            start = cutoff - timedelta(seconds=span)
            return mt5.copy_rates_range(symbol, tf, start, cutoff)
        except Exception:
            return None

    rates_1h = fetch_last_n_before_cutoff(mt5.TIMEFRAME_H1, h1_bars)
    rates_30m = fetch_last_n_before_cutoff(mt5.TIMEFRAME_M30, m30_bars) if hasattr(mt5, 'TIMEFRAME_M30') else fetch_last_n_before_cutoff(mt5.TIMEFRAME_M15, m15_bars)
    rates_15m = fetch_last_n_before_cutoff(mt5.TIMEFRAME_M15, m15_bars)
    rates_5m = fetch_last_n_before_cutoff(mt5.TIMEFRAME_M5, m5_bars)
    rates_1m = fetch_last_n_before_cutoff(mt5.TIMEFRAME_M1, m1_bars)
    mt5.shutdown()
    if any(rates is None or len(rates) == 0 for rates in [rates_1h, rates_30m, rates_15m, rates_5m, rates_1m]):
        print("数据获取失败，退出程序")
        return None, None, None, None, None
    print(f"数据截取截止（服务器 00:00, UTC）: {cutoff}")
    # Build DataFrames from records to ensure columns like 'time' exist
    def to_df(arr):
        try:
            return pd.DataFrame.from_records(arr)
        except Exception:
            return pd.DataFrame(arr)
    return (
        to_df(rates_1h),
        to_df(rates_30m),
        to_df(rates_15m),
        to_df(rates_5m),
        to_df(rates_1m),
    )


def preprocess_data(df_1h, df_30m, df_15m, df_5m, df_1m):
    dfs = {'1h': df_1h, '30m': df_30m, '15m': df_15m, '5m': df_5m, '1m': df_1m}
    for timeframe, df in dfs.items():
        if df is None or 'time' not in df.columns:
            print(f"{timeframe} 数据缺少 'time' 列，列名: {df.columns if df is not None else 'None'}")
            return None, None, None, None, None
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df = df.dropna(subset=['timestamp'])
        print(f"\n{timeframe} 数据: {len(df)} 条")
        if not df.empty:
            print(f"{timeframe} 时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        dfs[timeframe] = df
    return dfs['1h'], dfs['30m'], dfs['15m'], dfs['5m'], dfs['1m']


def check_pinbar_hourly(row, min_range_usd):
    try:
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
    except (ValueError, TypeError):
        return None
    price_range = h - l
    if price_range < min_range_usd:
        return None
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body = abs(c - o)
    upper_pinbar = (price_range >= min_range_usd and
                    upper_wick > 0.5 * price_range and
                    upper_wick > 2 * body and
                    body < 0.5 * price_range)
    lower_pinbar = (price_range >= min_range_usd and
                    lower_wick > 0.5 * price_range and
                    lower_wick > 2 * body and
                    body < 0.5 * price_range)
    if upper_pinbar:
        return "BUY"
    elif lower_pinbar:
        return "SELL"
    return None


def check_pinbar(row, min_range_usd):
    """Timeframe-agnostic Pinbar check using OHLC columns in a row.

    Returns "BUY" for upper-pinbar (long), "SELL" for lower-pinbar (short), or None.
    """
    try:
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
    except (ValueError, TypeError, KeyError):
        return None
    price_range = h - l
    if price_range < min_range_usd:
        return None
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body = abs(c - o)
    upper_pinbar = (price_range >= min_range_usd and
                    upper_wick > 0.5 * price_range and
                    upper_wick > 2 * body and
                    body < 0.5 * price_range)
    lower_pinbar = (price_range >= min_range_usd and
                    lower_wick > 0.5 * price_range and
                    lower_wick > 2 * body and
                    body < 0.5 * price_range)
    if upper_pinbar:
        return "BUY"
    elif lower_pinbar:
        return "SELL"
    return None
