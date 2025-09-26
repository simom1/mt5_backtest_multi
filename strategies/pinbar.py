import pandas as pd
from datetime import datetime

from .base import BaseStrategy
from ..utils.tools import check_pinbar_hourly
from ..config import INITIAL_CASH
import os


class MultiTimeframePinbarStrategy(BaseStrategy):
    """基于多时间周期的 Pinbar 形态交易策略。"""

    def __init__(self, stop_loss_usd=15, take_profit_usd=15, min_range_usd=10, qty=10, lot_mode='fixed',
                 signal_level: int = 3, backtest_level: int = 3,
                 commission_per_10lots_usd: float = 1.6):
        super().__init__()
        self.positions = []
        self.stop_loss_usd = stop_loss_usd
        self.take_profit_usd = take_profit_usd
        self.min_range_usd = min_range_usd
        self.qty = qty
        self.lot_mode = lot_mode
        # 可选的可视化/强度级别（目前未直接参与计算，可用于未来扩展）
        self.signal_level = signal_level
        self.backtest_level = backtest_level
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

    def preprocess_hourly_signals(self, df_1h: pd.DataFrame) -> None:
        print("\n开始预处理1小时Pinbar信号...")
        signal_count = 0
        for idx, row in df_1h.iterrows():
            signal_type = check_pinbar_hourly(row, self.min_range_usd)
            if signal_type and idx + 1 < len(df_1h):
                signal_time = row['timestamp'].floor('h')
                next_open = float(df_1h.iloc[idx + 1]['open'])
                self.hourly_signals[signal_time] = {
                    'type': signal_type,
                    'next_open_price': next_open
                }
                signal_count += 1
        print(f"总共检测到 {len(self.hourly_signals)} 个小时级Pinbar信号")

    def get_current_hour_signal(self, current_time: pd.Timestamp) -> bool:
        prev_hour = (current_time - pd.Timedelta(hours=1)).floor('h')
        if prev_hour in self.hourly_signals:
            if self.signal_start_time != prev_hour:
                self.current_hourly_signal = self.hourly_signals[prev_hour]
                self.signal_start_time = prev_hour
                return True
        if self.signal_start_time and (current_time - self.signal_start_time).total_seconds() >= 7200:
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
                # 盈利减仓 10 手，最小 10 手
                self.qty = max(10, self.qty - 10)
            elif net_pnl < 0:
                # 亏损加仓 10 手，最小 10 手
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
                trade_record = {
                    'entry_time': pos['entry_time'],
                    'exit_time': current_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'direction': direction,
                    'net_pnl': net_after_fee,
                    'fee': fee,
                    'exit_reason': exit_reason,
                    'qty': qty
                }
                self.trade_records.append(trade_record)
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
            trade_record = {
                'entry_time': pos['entry_time'],
                'exit_time': current_time,
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'direction': direction,
                'net_pnl': net_after_fee,
                'fee': fee,
                'exit_reason': '新信号平仓',
                'qty': qty
            }
            self.trade_records.append(trade_record)
            self.adjust_lot_size(net_pnl)
            self.positions.remove(pos)
            print(
                f"平仓 - {direction}: 进场价 {pos['entry_price']:.5f}, 出场价 {current_price:.5f}, 手数 {qty}, 盈亏 {net_pnl:.2f}")

    def save_trade_records(self, filename: str = 'trade_records.csv') -> None:
        if not self.trade_records:
            print("无交易记录可保存")
            return
        # Ensure outputs directory
        outputs_dir = 'outputs'
        os.makedirs(outputs_dir, exist_ok=True)
        # If filename has no directory, place it under outputs/
        if not os.path.dirname(filename):
            filename = os.path.join(outputs_dir, filename)
        df_trades = pd.DataFrame(self.trade_records)
        df_trades.to_csv(filename, index=False,
                         columns=['entry_time', 'exit_time', 'entry_price', 'exit_price', 'direction', 'net_pnl', 'fee', 'qty',
                                  'exit_reason'])
        print(f"交易记录已保存至 {filename}")

    def process_bar(self, row: pd.Series) -> None:
        current_time = row['timestamp']
        current_price = float(row['close'])
        self.check_positions(current_time, current_price)
        is_new_signal = self.get_current_hour_signal(current_time)
        if is_new_signal and self.current_hourly_signal:
            current_hour = current_time.floor('h')
            signal_hour = self.signal_start_time
            if current_hour >= signal_hour + pd.Timedelta(hours=1) and current_hour < signal_hour + pd.Timedelta(hours=2):
                if self.positions:
                    self.close_position_on_new_signal(current_time, current_price)
                entry_price = self.current_hourly_signal['next_open_price']
                stop_price, tp_price = self.calculate_levels(entry_price, self.current_hourly_signal['type'])
                new_position = {
                    "entry_time": current_time,
                    "direction": self.current_hourly_signal['type'],
                    "entry_price": entry_price,
                    "stop_loss": stop_price,
                    "take_profit": tp_price,
                    "size": self.qty,
                }
                self.positions.append(new_position)
                print(f"\n开仓: {self.current_hourly_signal['type']} at {entry_price:.5f}, 手数: {self.qty}")
                print(f"止损: {stop_price:.5f}, 止盈: {tp_price:.5f}")
                print(f"时间: {current_time}")

    def run_backtest(self, df_1m: pd.DataFrame) -> None:
        for idx, row in df_1m.iterrows():
            self.process_bar(row)
        if self.positions:
            final_price = float(df_1m.iloc[-1]['close'])
            final_time = df_1m.iloc[-1]['timestamp']
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
                trade_record = {
                    'entry_time': pos['entry_time'],
                    'exit_time': final_time,
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'direction': direction,
                    'net_pnl': net_after_fee,
                    'fee': fee,
                    'exit_reason': '强制平仓',
                    'qty': qty
                }
                self.trade_records.append(trade_record)
                self.adjust_lot_size(net_pnl)
            self.positions = []
        # 仅当未被外部（如调参器）抑制时才自动保存，避免重复CSV
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


__all__ = ["MultiTimeframePinbarStrategy"]
