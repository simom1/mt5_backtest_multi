import pandas as pd
from datetime import datetime
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now use absolute imports
from strategies.base import BaseStrategy
from utils.tools import check_pinbar
from config import INITIAL_CASH


class ConfigurableTimeframePinbarStrategy(BaseStrategy):
    """
    可配置形态识别级别与回测级别的 Pinbar 策略。

    参数:
        stop_loss_usd, take_profit_usd, min_range_usd, qty, lot_mode 同原策略
        signal_tf: 形态识别时间级别，取值 {'1h','30m','15m','5m','1m'}
        backtest_tf: 回测执行时间级别，取值同上
    """

    def __init__(self,
                 stop_loss_usd=15,
                 take_profit_usd=15,
                 min_range_usd=10,
                 qty=10,
                 lot_mode='fixed',
                 signal_tf: str = '5m',
                 backtest_tf: str = '5m',
                 commission_per_10lots_usd: float = 1.6):
        super().__init__()
        self.positions = []
        self.stop_loss_usd = stop_loss_usd
        self.take_profit_usd = take_profit_usd
        self.min_range_usd = min_range_usd
        self.qty = qty
        self.lot_mode = lot_mode
        self.signal_tf = signal_tf
        self.backtest_tf = backtest_tf

        self.hourly_signals = {}
        self.current_hourly_signal = None
        self.signal_start_time = None
        self.initial_cash = INITIAL_CASH
        self.current_cash = INITIAL_CASH
        self.lot_size_history = []
        self.lot_adjustment_log = []
        # Commission settings (default: 10手 = 1.6 美元)
        self.commission_per_10lots_usd = float(commission_per_10lots_usd)
        self.total_fees_usd = 0.0

    # 名称保持兼容，但可接受任意时间级别的 df
    def preprocess_hourly_signals(self, df_signal: pd.DataFrame) -> None:
        print(f"\n开始预处理{self.signal_tf} Pinbar信号...")
        self.hourly_signals.clear()
        self.signal_start_time = None
        for idx, row in df_signal.iterrows():
            signal_type = check_pinbar(row, self.min_range_usd)
            if signal_type and idx + 1 < len(df_signal):
                # 使用该级别的 next bar open 作为入场
                signal_time = pd.to_datetime(row['timestamp']).floor(self._freq_alias(self.signal_tf))
                next_open = float(df_signal.iloc[idx + 1]['open'])
                self.hourly_signals[signal_time] = {
                    'type': signal_type,
                    'next_open_price': next_open
                }
        print(f"总共检测到 {len(self.hourly_signals)} 个 {self.signal_tf} 级 Pinbar 信号")

    def _freq_alias(self, tf: str) -> str:
        return {
            '1h': 'h',
            '30m': '30min',
            '15m': '15min',
            '5m': '5min',
            '1m': 'min',
        }.get(tf, 'min')

    def get_current_hour_signal(self, current_time: pd.Timestamp) -> bool:
        # 取上一个信号周期的时间窗口
        step = {
            '1h': pd.Timedelta(hours=1),
            '30m': pd.Timedelta(minutes=30),
            '15m': pd.Timedelta(minutes=15),
            '5m': pd.Timedelta(minutes=5),
            '1m': pd.Timedelta(minutes=1),
        }.get(self.signal_tf, pd.Timedelta(minutes=1))
        prev_period = (current_time - step).floor(self._freq_alias(self.signal_tf))
        if prev_period in self.hourly_signals:
            if self.signal_start_time != prev_period:
                self.current_hourly_signal = self.hourly_signals[prev_period]
                self.signal_start_time = prev_period
                return True
        # 信号有效期：两个周期
        if self.signal_start_time and (current_time - self.signal_start_time) >= (2 * step):
            self.current_hourly_signal = None
            self.signal_start_time = None
        return False

    def calculate_levels(self, entry_price: float, direction: str):
        if direction == "BUY":
            stop_price = entry_price - self.stop_loss_usd
            profit_price = entry_price + self.take_profit_usd
        else:
            stop_price = entry_price + self.stop_loss_usd
            profit_price = entry_price - self.take_profit_usd
        return stop_price, profit_price

    def adjust_lot_size(self, net_pnl: float) -> None:
        if self.lot_mode == 'fixed':
            self.qty = 10
        else:
            prev_qty = self.qty
            if net_pnl > 0:
                self.qty = max(10, self.qty - 10)
            elif net_pnl < 0:
                self.qty = max(10, self.qty + 10)
            if prev_qty != self.qty:
                self.lot_adjustment_log.append({
                    'time': datetime.now(),
                    'prev_qty': prev_qty,
                    'new_qty': self.qty,
                    'reason': '盈利' if net_pnl > 0 else '亏损'
                })
        self.lot_size_history.append(self.qty)

    def check_positions(self, current_time: pd.Timestamp, current_price: float) -> None:
        for pos in self.positions[:]:
            direction = pos['direction']
            stop_price = pos['stop_loss']
            take_profit = pos['take_profit']
            qty = pos['size']
            should_close = False
            exit_reason = ""
            if direction == 'BUY':
                if current_price <= stop_price:
                    should_close = True
                    exit_reason = "止损"
                elif current_price >= take_profit:
                    should_close = True
                    exit_reason = "止盈"
            else:
                if current_price >= stop_price:
                    should_close = True
                    exit_reason = "止损"
                elif current_price <= take_profit:
                    should_close = True
                    exit_reason = "止盈"
            if should_close:
                if direction == 'BUY':
                    net_pnl = (current_price - pos['entry_price']) * qty
                else:
                    net_pnl = (pos['entry_price'] - current_price) * qty
                fee = (self.commission_per_10lots_usd * (qty / 10.0)) if self.commission_per_10lots_usd > 0 else 0.0
                net_after_fee = net_pnl - fee
                self.current_cash += net_after_fee
                self.total_fees_usd += fee
                self.trade_records.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': current_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'direction': direction,
                    'net_pnl': net_after_fee,
                    'fee': fee,
                    'exit_reason': exit_reason,
                    'qty': qty
                })
                self.adjust_lot_size(net_pnl)
                self.positions.remove(pos)

    def close_position_on_new_signal(self, current_time: pd.Timestamp, current_price: float) -> None:
        for pos in self.positions[:]:
            direction = pos['direction']
            qty = pos['size']
            if direction == 'BUY':
                net_pnl = (current_price - pos['entry_price']) * qty
            else:
                net_pnl = (pos['entry_price'] - current_price) * qty
            fee = (self.commission_per_10lots_usd * (qty / 10.0)) if self.commission_per_10lots_usd > 0 else 0.0
            net_after_fee = net_pnl - fee
            self.current_cash += net_after_fee
            self.total_fees_usd += fee
            self.trade_records.append({
                'entry_time': pos['entry_time'],
                'exit_time': current_time,
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'direction': direction,
                'net_pnl': net_after_fee,
                'fee': fee,
                'exit_reason': '新信号平仓',
                'qty': qty
            })
            self.adjust_lot_size(net_pnl)
            self.positions.remove(pos)

    def process_bar(self, row: pd.Series) -> None:
        current_time = row['timestamp']
        current_price = float(row['close'])
        self.check_positions(current_time, current_price)
        is_new_signal = self.get_current_hour_signal(current_time)
        if is_new_signal and self.current_hourly_signal:
            current_period = current_time.floor(self._freq_alias(self.backtest_tf))
            signal_period = self.signal_start_time
            step = {
                '1h': pd.Timedelta(hours=1),
                '30m': pd.Timedelta(minutes=30),
                '15m': pd.Timedelta(minutes=15),
                '5m': pd.Timedelta(minutes=5),
                '1m': pd.Timedelta(minutes=1),
            }.get(self.signal_tf, pd.Timedelta(minutes=1))
            if current_period >= signal_period + step and current_period < signal_period + 2 * step:
                if self.positions:
                    self.close_position_on_new_signal(current_time, current_price)
                entry_price = self.current_hourly_signal['next_open_price']
                stop_price, tp_price = self.calculate_levels(entry_price, self.current_hourly_signal['type'])
                self.positions.append({
                    'entry_time': current_time,
                    'direction': self.current_hourly_signal['type'],
                    'entry_price': entry_price,
                    'stop_loss': stop_price,
                    'take_profit': tp_price,
                    'size': self.qty,
                })

    def run_backtest(self, df_bt: pd.DataFrame) -> None:
        for _, row in df_bt.iterrows():
            self.process_bar(row)
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
                net_after_fee = net_pnl - fee
                self.current_cash += net_after_fee
                self.total_fees_usd += fee
                self.trade_records.append({
                    'entry_time': pos['entry_time'],
                    'exit_time': final_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'direction': direction,
                    'net_pnl': net_after_fee,
                    'fee': fee,
                    'exit_reason': '强制平仓',
                    'qty': qty
                })
        self.positions = []
        # 仅当未被外部（如回测引擎/调参器）抑制时才自动保存，避免重复CSV
        if not getattr(self, 'suppress_autosave', False):
            self.save_trade_records()

    def get_stats(self):
        if not self.trade_records:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'final_cash': self.current_cash,
                'total_return': 0.0,
                'max_lot_size': self.qty,
                'min_lot_size': self.qty,
                'final_lot_size': self.qty
            }
        total_pnl = sum(trade['net_pnl'] for trade in self.trade_records)
        winning_trades = len([t for t in self.trade_records if t['net_pnl'] > 0])
        losing_trades = len([t for t in self.trade_records if t['net_pnl'] < 0])
        win_rate = winning_trades / len(self.trade_records) if self.trade_records else 0
        total_return = (self.current_cash - self.initial_cash) / self.initial_cash
        max_lot_size = max(self.lot_size_history) if self.lot_size_history else self.qty
        min_lot_size = min(self.lot_size_history) if self.lot_size_history else self.qty
        final_lot_size = self.qty
        return {
            'total_trades': len(self.trade_records),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'final_cash': self.current_cash,
            'total_return': total_return,
            'average_trade': total_pnl / len(self.trade_records) if self.trade_records else 0,
            'max_lot_size': max_lot_size,
            'min_lot_size': min_lot_size,
            'final_lot_size': final_lot_size,
            'total_fees': self.total_fees_usd,
        }

    def save_trade_records(self, filename: str = 'trade_records.csv') -> None:
        if not self.trade_records:
            print("无交易记录可保存")
            return
        outputs_dir = 'outputs'
        os.makedirs(outputs_dir, exist_ok=True)
        # If filename is default or has no parent directory, create a unique name to avoid collisions/locks
        if not filename or filename == 'trade_records.csv' or not os.path.dirname(filename):
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(
                outputs_dir,
                f"trade_records_cfgpinbar_{self.signal_tf}_{self.backtest_tf}_{ts}.csv"
            )
        import pandas as pd
        pd.DataFrame(self.trade_records).to_csv(
            filename,
            index=False,
            columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'direction', 'net_pnl', 'fee', 'qty', 'exit_reason']
        )
        print(f"交易记录已保存至 {filename}")


__all__ = ["ConfigurableTimeframePinbarStrategy"]
