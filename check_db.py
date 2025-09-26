import os
import sqlite3
from pathlib import Path

def check_database():
    # 获取项目根目录
    project_root = Path(__file__).parent
    db_path = project_root / 'outputs' / 'backtests.db'
    
    print(f"数据库路径: {db_path}")
    print(f"数据库文件存在: {db_path.exists()}")
    
    if not db_path.exists():
        print("错误: 数据库文件不存在，请先运行回测以创建数据库")
        return
    
    try:
        # 连接到数据库
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # 获取所有表名
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("\n数据库中的表:")
        for table in tables:
            table_name = table[0]
            print(f"\n表: {table_name}")
            
            # 获取表结构
            try:
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                print("字段:")
                for col in columns:
                    print(f"  {col[1]} ({col[2]})")
                
                # 获取前5行数据
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                rows = cursor.fetchall()
                
                if rows:
                    print("\n前5行数据:")
                    for row in rows:
                        print(f"  {row}")
                else:
                    print("  (无数据)")
                    
            except sqlite3.Error as e:
                print(f"  读取表 {table_name} 时出错: {e}")
        
    except Exception as e:
        print(f"\n数据库连接错误: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_database()
