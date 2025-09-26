import sqlite3
import pandas as pd

# Database path
db_path = r"C:\Users\Administrator\PycharmProjects\pythonProject\outputs\backtests.db"

# Connect to the database
conn = sqlite3.connect(db_path)

# Get list of all tables
print("Tables in the database:")
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print(tables)

# Get schema for each table
for table in tables['name']:
    print(f"\nSchema for table '{table}':")
    schema = pd.read_sql(f"PRAGMA table_info({table});", conn)
    print(schema)
    
    # Show first few rows if the table is not empty
    count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table};", conn).iloc[0]['count']
    if count > 0:
        print(f"\nFirst 3 rows of '{table}':")
        rows = pd.read_sql(f"SELECT * FROM {table} LIMIT 3;", conn)
        print(rows)
    else:
        print(f"Table '{table}' is empty")

conn.close()
