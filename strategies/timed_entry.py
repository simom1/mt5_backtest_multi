import pandas as pd
from datetime import datetime
from typing import Dict, Any
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now use absolute imports
from strategies.base import BaseStrategy
from config import INITIAL_CASH


class TimedEntryStrategy5m(BaseStrategy):
    """
    策略2：定时做单策略（5分钟级别，固定手数 + 固定美元止盈止损）。

    参数（均有默认值，可在后续 GUI 中扩展成可配置）:
      - entry_minute: 每小时的第几分钟触发开单（默认 20，步长 5）
      - tp_usd: 止盈美元（默认 6.0）
      - sl_usd: 止损美元（默认 5.0）
      - fixed_qty: 固定手数（默认 10）
      - enable_time_filter: 是否启用交易时间过滤（默认 True）
      - start_hour, end_hour: 交易时段（默认 8-20）
      - trade_direction: 交易方向（"Long Only" | "Short Only" | "Both" | "Alternating"），默认 "Both"
      - max_positions: 最大同时持仓数量（默认 3）

    约束：本策略固定在 5m 上进行识别与回测。
    """

    def __init__(self,
                 entry_minute: int = 20,
                 tp_usd: float = 6.0,
                 sl_usd: float = 5.0,
                 fixed_qty: float = 10,
                 enable_time_filter: bool = True,
                 start_hour: int = 8,
                 end_hour: int = 20,
                 trade_direction: str = "Both",
                 max_positions: int = 3,
                 # Aliases for integration with existing GUI/backtester
                 stop_loss_usd: float | None = None,
                 take_profit_usd: float | None = None,
                 qty: float | None = None,
                 # Compatibility no-op params
                 min_range_usd: float | None = None,
                 lot_mode: str | None = None,
                 # Commission settings (single-side): 10手 = 1.6 美元
                 commission_per_10lots_usd: float = 1.6,
                 **kwargs: dict):
        super().__init__()
        # 固定时间级别（引擎会用到）
        self.signal_tf = '5m'
        self.backtest_tf = '5m'
        # 参数
        # 归一化到 5 分钟网格，范围 [0,59]
        try:
            em = int(entry_minute)
            if em < 0: em = 0
            if em > 59: em = 59
            em = em - (em % 5)
            self.entry_minute = em
        except Exception:
            self.entry_minute = 20
        # Aliases take precedence if provided
        self.tp_usd = take_profit_usd if take_profit_usd is not None else tp_usd
        self.sl_usd = stop_loss_usd if stop_loss_usd is not None else sl_usd
        self.fixed_qty = qty if qty is not None else fixed_qty
        self.enable_time_filter = enable_time_filter
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.trade_direction = trade_direction
        self.max_positions = max_positions
        # lot mode
        self.lot_mode = lot_mode or 'fixed'
        # 运行时状态
        self.positions = []
        self.initial_cash = INITIAL_CASH
        self.current_cash = INITIAL_CASH
        self.trade_counter = 0  # 用于 Alternating
        self.lot_size_history = []
        self.commission_per_10lots_usd = float(commission_per_10lots_usd)
        self.total_fees_usd = 0.0

    # 兼容接口：此策略不需要预处理信号
    def preprocess_hourly_signals(self, df_signal: pd.DataFrame) -> None:
        return

    def _within_time_filter(self, ts: pd.Timestamp) -> bool:
        if not self.enable_time_filter:
            return True
        hr = int(ts.hour)
        return self.start_hour <= hr <= self.end_hour

    def _entry_allowed_now(self, ts: pd.Timestamp, is_new_bar: bool) -> bool:
        # 在回测中逐条遍历，每条都是新 bar；此处保留 is_new_bar 以便未来扩展
        return (ts.minute == self.entry_minute) and is_new_bar and self._within_time_filter(ts)

    def _maybe_open_trades(self, ts: pd.Timestamp, price: float):
        if len(self.positions) >= self.max_positions:
            return
        should_long = False
        should_short = False
        if self.trade_direction == "Long Only":
            should_long = True
        elif self.trade_direction == "Short Only":
            should_short = True
        elif self.trade_direction == "Both":
            should_long = True
            should_short = True
        elif self.trade_direction == "Alternating":
            if self.trade_counter % 2 == 0:
                should_long = True
            else:
                should_short = True
            self.trade_counter += 1
        if should_long:
            self._open_position(ts, price, direction='BUY', qty=self.fixed_qty)
        if should_short:
            self._open_position(ts, price, direction='SELL', qty=self.fixed_qty)

    def _open_position(self, ts: pd.Timestamp, entry_price: float, direction: str, qty: float):
        if direction == 'BUY':
            stop_price = entry_price - self.sl_usd
            take_profit = entry_price + self.tp_usd
        else:
            stop_price = entry_price + self.sl_usd
            take_profit = entry_price - self.tp_usd
        self.positions.append({
            'entry_time': ts,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_price,
            'take_profit': take_profit,
            'size': qty,
        })

    def _check_positions(self, ts: pd.Timestamp, price: float):
        for pos in self.positions[:]:
            direction = pos['direction']
            qty = pos['size']
            sl = pos['stop_loss']
            tp = pos['take_profit']
            exit_reason = None
            if direction == 'BUY':
                if price <= sl:
                    exit_reason = '止损'
                elif price >= tp:
                    exit_reason = '止盈'
                net_pnl = (price - pos['entry_price']) * qty
            else:
                if price >= sl:
                    exit_reason = '止损'
                elif price <= tp:
                    exit_reason = '止盈'
                net_pnl = (pos['entry_price'] - price) * qty
            if exit_reason is not None:
                # 手续费：按 10 手 = 固定 16 美元等比例计算
                fee = (self.commission_per_10lots_usd * (qty / 10.0)) if self.commission_per_10lots_usd > 0 else 0.0
                net_pnl_after_fee = net_pnl - fee
                self.current_cash += net_pnl_after_fee
                self.total_fees_usd += fee
                self.trade_records.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': ts,
                    'entry_price': pos['entry_price'],
                    'exit_price': price,
                    'direction': direction,
                    'net_pnl': net_pnl_after_fee,
                    'fee': fee,
                    'exit_reason': exit_reason,
                    'qty': qty,
                })
                # 动态手数调整
                self._adjust_lot_size(net_pnl)
                self.positions.remove(pos)

    def _adjust_lot_size(self, net_pnl: float) -> None:
        if self.lot_mode == 'fixed':
            # 对齐其他策略，固定模式下手数回到至少 10
            self.fixed_qty = 10
        else:
            prev_qty = self.fixed_qty
            if net_pnl > 0:
                # 盈利减仓 10 手，最小 10 手
                self.fixed_qty = max(10, self.fixed_qty - 10)
            elif net_pnl < 0:
                # 亏损加仓 10 手，最小 10 手
                self.fixed_qty = max(10, self.fixed_qty + 10)
        self.lot_size_history.append(self.fixed_qty)

    def run_backtest(self, df_bt: pd.DataFrame) -> None:
        # 期望 df_bt 为 5m 数据，且包含 timestamp/open/high/low/close
        prev_index = None
        for idx, row in df_bt.iterrows():
            ts = row['timestamp']
            price = float(row['close'])
            is_new_bar = (prev_index is None) or (idx != prev_index)
            prev_index = idx
            # 先检查既有持仓
            self._check_positions(ts, price)
            # 判断是否触发定时开单
            if self._entry_allowed_now(ts, is_new_bar):
                self._maybe_open_trades(ts, price)
        # 强制平仓（以最后一根收盘价）
        if self.positions:
            final_price = float(df_bt.iloc[-1]['close'])
            final_time = df_bt.iloc[-1]['timestamp']
            for pos in self.positions:
                direction = pos['direction']
                qty = pos['size']
                if direction == 'BUY':
                    net_pnl = (final_price - pos['entry_price']) * qty
                else:
                    net_pnl = (pos['entry_price'] - final_price) * qty
                fee = (self.commission_per_10lots_usd * (qty / 10.0)) if self.commission_per_10lots_usd > 0 else 0.0
                net_pnl_after_fee = net_pnl - fee
                self.current_cash += net_pnl_after_fee
                self.total_fees_usd += fee
                self.trade_records.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': final_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'direction': direction,
                    'net_pnl': net_pnl_after_fee,
                    'fee': fee,
                    'exit_reason': '强制平仓',
                    'qty': qty,
                })
            self.positions = []
        # 仅当未被外部（如调参器）抑制时才自动保存，避免重复CSV
        if not getattr(self, 'suppress_autosave', False):
            self.save_trade_records()

    def get_stats(self) -> Dict[str, Any]:
        if not self.trade_records:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'final_cash': self.current_cash,
                'total_return': 0.0,
            }
        total_pnl = sum(trade['net_pnl'] for trade in self.trade_records)
        winning_trades = len([t for t in self.trade_records if t['net_pnl'] > 0])
        losing_trades = len([t for t in self.trade_records if t['net_pnl'] < 0])
        win_rate = winning_trades / len(self.trade_records)
        total_return = (self.current_cash - self.initial_cash) / self.initial_cash
        return {
            'total_trades': len(self.trade_records),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_cash': self.current_cash,
            'total_return': total_return,
            'total_fees': self.total_fees_usd,
        }

    def save_trade_records(self, filename: str = 'trade_records.csv') -> None:
        if not self.trade_records:
            return
        os.makedirs('outputs', exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not filename or filename == 'trade_records.csv' or not os.path.dirname(filename):
            filename = os.path.join('outputs', f'trade_records_timed5m_{ts}.csv')
        pd.DataFrame(self.trade_records).to_csv(
            filename,
            index=False,
            columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'direction', 'net_pnl', 'fee', 'qty', 'exit_reason']
        )


__all__ = ["TimedEntryStrategy5m"]
