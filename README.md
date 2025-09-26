# MT5 多周期回测系统

## 项目概述
这是一个基于 Python 和 MetaTrader 5 (MT5) 的多周期回测系统，提供图形化界面和命令行界面，支持多种交易策略的回测和参数优化。

## 功能特点

- 🖥️ 现代化图形用户界面 (GUI)
- 📊 多时间框架分析 (M1, M5, M15, M30, H1)
- ⚡ 支持多种交易策略
- 🔍 参数优化和网格搜索
- 📈 详细的回测结果和可视化
- 💾 交易记录和统计数据的CSV导出

## 项目结构

```
mt5多周期回测/
├── engines/           # 回测引擎
│   └── backtester.py  # 回测核心逻辑
├── strategies/        # 交易策略
│   ├── __init__.py
│   ├── base.py        # 策略基类
│   └── pinbar.py      # Pinbar策略实现
├── utils/             # 工具函数
│   ├── __init__.py
│   ├── db.py          # 数据库操作
│   └── plotting.py    # 图表绘制
├── main.py           # 命令行入口
├── main_window.py    # 主窗口实现
├── gui_app.py        # GUI应用入口
├── config.py         # 配置文件
└── requirements.txt  # 依赖包
```

## 安装指南

1. 确保已安装 Python 3.8+
2. 克隆仓库
3. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 安装 MetaTrader 5 Python 包：
   ```bash
   pip install MetaTrader5
   ```

## 使用方法

### 图形界面模式
```bash
python gui_app.py
```

### 命令行模式
```bash
# 单次回测
python -m mt5多周期回测.main --mode run --symbol XAUUSD --sl 30 --tp 60 --mr 15 --qty 0.1 --lot_mode fixed

# 参数优化
python -m mt5多周期回测.main --mode tune --symbol XAUUSD --grid_sl "20,30,40" --grid_tp "40,60,80" --grid_mr "10,15,20"
```

## 支持的交易品种
- XAUUSD (黄金/美元)
- EURUSD (欧元/美元)
- 其他 MT5 支持的交易品种

## 策略说明

### Pinbar 策略
基于 Pinbar 形态的交易策略，支持以下参数：
- 止损 (Stop Loss)
- 止盈 (Take Profit)
- 最小波动范围 (Min Range)
- 手数模式 (固定/动态)

## 开发指南

### 添加新策略
1. 在 `strategies` 目录下创建新策略文件
2. 继承 `BaseStrategy` 类
3. 实现必要的方法
4. 在 `main_window.py` 中注册新策略

### 运行测试
```bash
pytest tests/
```

## 性能优化

- 使用多进程进行参数优化
- 数据缓存机制
- 增量更新回测结果

## 常见问题

### 数据获取失败
- 确保 MT5 终端已安装并登录
- 检查网络连接
- 验证交易品种名称是否正确

### 回测速度慢
- 减少数据量
- 增加 `--max_workers` 参数
- 使用更高配置的机器

## 贡献指南

欢迎提交 Issue 和 Pull Request。提交代码前请确保通过所有测试。

## 许可证

MIT License

## 作者

[Bill] 

## 致谢

- MetaTrader 5 团队
- Python 社区
- 所有贡献者
