#!/usr/bin/env python3
"""
GUI测试脚本 - 验证MT5多周期回测GUI的基本功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """测试所有必要的导入"""
    print("测试导入...")
    
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        print("[OK] PySide6 导入成功")
    except ImportError as e:
        print(f"[ERROR] PySide6 导入失败: {e}")
        return False
    
    try:
        import matplotlib
        print("[OK] matplotlib 导入成功")
    except ImportError as e:
        print(f"[ERROR] matplotlib 导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        print("[OK] pandas 导入成功")
    except ImportError as e:
        print(f"[ERROR] pandas 导入失败: {e}")
        return False
    
    try:
        import numpy as np
        print("[OK] numpy 导入成功")
    except ImportError as e:
        print(f"[ERROR] numpy 导入失败: {e}")
        return False
    
    return True

def test_ui_components():
    """测试UI组件"""
    print("\n测试UI组件...")
    
    try:
        from ui_components import GlassCard, ModernButton, AnimatedProgressBar, KPICard, ModernInput
        print("[OK] UI组件导入成功")
        
        # 测试组件实例化
        from PySide6 import QtWidgets
        app = QtWidgets.QApplication(sys.argv)
        
        # 测试GlassCard
        card = GlassCard("测试卡片")
        print("[OK] GlassCard 实例化成功")
        
        # 测试ModernButton
        button = ModernButton("测试按钮")
        print("[OK] ModernButton 实例化成功")
        
        # 测试KPICard
        kpi = KPICard("测试KPI", "100")
        print("[OK] KPICard 实例化成功")
        
        app.quit()
        return True
        
    except Exception as e:
        print(f"[ERROR] UI组件测试失败: {e}")
        return False

def test_main_window():
    """测试主窗口"""
    print("\n测试主窗口...")
    
    try:
        # 使用绝对导入避免相对导入问题
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        
        # 测试导入修复后的版本
        import main_window_fixed
        print("[OK] main_window_fixed 模块导入成功")
        
        # 检查MainWindow类是否存在
        if hasattr(main_window_fixed, 'MainWindow'):
            print("[OK] MainWindow 类存在")
            return True
        else:
            print("[ERROR] MainWindow 类不存在")
            return False
        
    except Exception as e:
        print(f"[ERROR] 主窗口测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("MT5多周期回测GUI测试开始...")
    print("=" * 50)
    
    # 测试导入
    if not test_imports():
        print("\n导入测试失败，请检查依赖安装")
        return False
    
    # 测试UI组件
    if not test_ui_components():
        print("\nUI组件测试失败")
        return False
    
    # 测试主窗口
    if not test_main_window():
        print("\n主窗口测试失败")
        return False
    
    print("\n" + "=" * 50)
    print("[OK] 所有测试通过！GUI可以正常运行")
    print("\n启动GUI命令:")
    print("python gui_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
