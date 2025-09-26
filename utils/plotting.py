import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_pnl_chart(trade_records, initial_cash=50000, title="PnL Chart", fig=None, axes=None, show=True):
    """绘制回测结果图表，包括账户余额、单笔交易盈亏和手数变化。

    参数:
        trade_records (list or pd.DataFrame): 交易记录列表，每条记录包含 entry_time, exit_time, net_pnl, qty 等。
        initial_cash (float): 初始账户资金，默认为 50000。
        title (str): 图表标题，默认为 "PnL Chart"。
    """
    # 接受 list[dict] 或 pd.DataFrame
    if isinstance(trade_records, pd.DataFrame):
        df_trades = trade_records.copy()
    else:
        df_trades = pd.DataFrame(trade_records)

    if df_trades.empty:
        print("无交易记录，无法绘制图表")
        return

    # 计算账户余额曲线
    balance = initial_cash
    balance_history = [initial_cash]
    for _, trade in df_trades.iterrows():
        balance += trade['net_pnl']
        balance_history.append(balance)

    # 创建三合一图表（支持外部传入 Figure / Axes 以便嵌入 GUI）
    created_fig = False
    if fig is None or axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        created_fig = True
    ax1, ax2, ax3 = axes

    # 账户余额曲线
    ax1.plot(df_trades['exit_time'], balance_history[1:], label='账户余额', color='blue')
    ax1.set_title(title)
    ax1.set_ylabel('账户余额 ($)')
    ax1.legend()
    ax1.grid(True)

    # 单笔交易盈亏
    ax2.bar(df_trades['exit_time'], df_trades['net_pnl'],
            color=['green' if x > 0 else 'red' for x in df_trades['net_pnl']],
            label='单笔盈亏')
    ax2.set_ylabel('盈亏 ($)')
    ax2.legend()
    ax2.grid(True)

    # 手数变化
    ax3.plot(df_trades['exit_time'], df_trades['qty'], label='交易手数', color='purple')
    ax3.set_xlabel('时间')
    ax3.set_ylabel('手数')
    ax3.legend()
    ax3.grid(True)

    # 计算并显示统计指标
    total_pnl = df_trades['net_pnl'].sum()
    winning_trades = len(df_trades[df_trades['net_pnl'] > 0])
    total_trades = len(df_trades)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    max_drawdown = min(0, (min(balance_history) - initial_cash) / initial_cash) if balance_history else 0

    stats_text = (f'总盈亏: ${total_pnl:.2f}\n'
                  f'胜率: {win_rate:.2%}\n'
                  f'最大回撤: {max_drawdown:.2%}')
    fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    if show and created_fig:
        plt.show()
    return fig
