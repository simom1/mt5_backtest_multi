import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

from ..engines.backtester import run_backtest_pipeline
from ..utils.plotting import plot_pnl_chart
from ..utils.tools import initialize_mt5, load_data, preprocess_data
from .tuner import tune_parameters
from ..config import (
    DEFAULT_SYMBOL, DEFAULT_H1_BARS, DEFAULT_M1_BARS,
    DEFAULT_STOP_LOSS_USD, DEFAULT_TAKE_PROFIT_USD, DEFAULT_MIN_RANGE_USD,
    DEFAULT_QTY, DEFAULT_LOT_MODE
)

# Matplotlib 全局配置在 utils.plotting 中统一设置


def parse_list(s: str):
    """解析逗号分隔的数字列表字符串。"""
    return [float(x) for x in s.split(',') if x.strip()]


def cli():
    parser = argparse.ArgumentParser(description="MT5 多周期回测 - 简洁入口")
    parser.add_argument('--mode', choices=['run', 'tune'], default='run', help='运行模式: run 单次回测, tune 参数调优')
    parser.add_argument('--symbol', default=DEFAULT_SYMBOL, help='交易品种')
    parser.add_argument('--h1_bars', type=int, default=DEFAULT_H1_BARS, help='H1 数据条数')
    parser.add_argument('--m1_bars', type=int, default=DEFAULT_M1_BARS, help='M1 数据条数')

    # 策略参数（run）
    parser.add_argument('--sl', type=float, default=DEFAULT_STOP_LOSS_USD, help='止损（美元）')
    parser.add_argument('--tp', type=float, default=DEFAULT_TAKE_PROFIT_USD, help='止盈（美元）')
    parser.add_argument('--mr', type=float, default=DEFAULT_MIN_RANGE_USD, help='Pinbar 最小波动（美元）')
    parser.add_argument('--qty', type=int, default=DEFAULT_QTY, help='初始手数')
    parser.add_argument('--lot_mode', choices=['fixed', 'dynamic'], default=DEFAULT_LOT_MODE, help='手数模式')

    # 调参网格（tune）
    parser.add_argument('--grid_sl', type=str, default=str(DEFAULT_STOP_LOSS_USD), help='调参-止损列表，逗号分隔')
    parser.add_argument('--grid_tp', type=str, default=str(DEFAULT_TAKE_PROFIT_USD), help='调参-止盈列表，逗号分隔')
    parser.add_argument('--grid_mr', type=str, default=str(DEFAULT_MIN_RANGE_USD), help='调参-最小波动列表，逗号分隔')

    args = parser.parse_args()

    if args.mode == 'run':
        trades_df, stats, strategy = run_backtest_pipeline(
            symbol=args.symbol,
            h1_bars=args.h1_bars,
            m1_bars=args.m1_bars,
            strategy_params={
                'stop_loss_usd': args.sl,
                'take_profit_usd': args.tp,
                'min_range_usd': args.mr,
                'qty': args.qty,
                'lot_mode': args.lot_mode,
            }
        )

        # 打印结果
        print("\n单次回测结果")
        print(f"总交易次数: {stats['total_trades']}")
        print(f"盈利交易: {stats['winning_trades']}")
        print(f"亏损交易: {stats['losing_trades']}")
        print(f"胜率: {stats['win_rate']:.2%}")
        print(f"总盈亏: ${stats['total_pnl']:.2f}")
        print(f"初始资金: ${strategy.initial_cash:.2f}")
        print(f"最终资金: ${stats['final_cash']:.2f}")
        print(f"总收益率: {stats['total_return']:.2%}")

        # 保存交易记录
        os.makedirs('outputs', exist_ok=True)
        out_csv = os.path.join(
            'outputs',
            f"trade_records_{args.symbol}_{args.lot_mode}_sl{int(args.sl)}_tp{int(args.tp)}_mr{int(args.mr)}.csv"
        )
        trades_df.to_csv(out_csv, index=False)
        print(f"交易记录已保存至 {out_csv}")

        # 绘图
        plot_pnl_chart(trades_df, strategy.initial_cash,
                       f"{args.symbol} {args.lot_mode} (sl={args.sl}, tp={args.tp}, mr={args.mr})")

    else:  # tune
        # 初始化 + 加载 + 预处理一次（使用统一的 utils.tools API，返回 5 个周期）
        if not initialize_mt5():
            print("MT5 初始化失败")
            return
        df_1h, df_30m, df_15m, df_5m, df_1m = load_data(args.symbol, h1_bars=args.h1_bars, m1_bars=args.m1_bars)
        if any(x is None for x in [df_1h, df_30m, df_15m, df_5m, df_1m]):
            print("数据加载失败")
            return
        df_1h, df_30m, df_15m, df_5m, df_1m = preprocess_data(df_1h, df_30m, df_15m, df_5m, df_1m)
        if any(x is None for x in [df_1h, df_30m, df_15m, df_5m, df_1m]):
            print("数据预处理失败")
            return

        stop_loss_values = parse_list(args.grid_sl)
        take_profit_values = parse_list(args.grid_tp)
        min_range_values = parse_list(args.grid_mr)

        print("\n固定手数模式超参数调优")
        tuning_results = tune_parameters(
            df_1h=df_1h, df_1m=df_1m,
            stop_loss_values=stop_loss_values,
            take_profit_values=take_profit_values,
            min_range_values=min_range_values,
        )

        # 输出与保存
        for res in tuning_results:
            print(f"\n固定手数模式 (sl={res['stop_loss_usd']}, tp={res['take_profit_usd']}, mr={res['min_range_usd']}):")
            print(f"总交易次数: {res['total_trades']}")
            print(f"盈利交易: {res['winning_trades']}")
            print(f"亏损交易: {res['losing_trades']}")
            print(f"胜率: {res['win_rate']:.2%}")
            print(f"总盈亏: ${res['total_pnl']:.2f}")
            print(f"初始资金: ${res['initial_cash']:.2f}")
            print(f"最终资金: ${res['final_cash']:.2f}")
            print(f"总收益率: {res['total_return']:.2%}")

        df_tuning_results = pd.DataFrame(tuning_results)
        os.makedirs('outputs', exist_ok=True)
        tuning_csv = os.path.join('outputs', 'tuning_results.csv')
        df_tuning_results.to_csv(tuning_csv, index=False)
        print(f"\n超参数调优结果已保存至 {tuning_csv}")

        # 最佳参数并绘图
        best_result = max(tuning_results, key=lambda x: x['total_return'])
        print("\n最佳参数组合:")
        print(f"stop_loss_usd: {best_result['stop_loss_usd']}")
        print(f"take_profit_usd: {best_result['take_profit_usd']}")
        print(f"min_range_usd: {best_result['min_range_usd']}")
        print(f"总收益率: {best_result['total_return']:.2%}")
        print(f"总盈亏: ${best_result['total_pnl']:.2f}")
        print(f"胜率: {best_result['win_rate']:.2%}")

        best_trade_records = pd.read_csv(best_result['trade_record_file'])
        plot_pnl_chart(best_trade_records, best_result['initial_cash'],
                       f"Pinbar 固定手数 (sl={best_result['stop_loss_usd']}, tp={best_result['take_profit_usd']}, mr={best_result['min_range_usd']})")


if __name__ == "__main__":
    cli()