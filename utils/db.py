import os
import sqlite3
import json
from typing import Dict, Any, Optional

import pandas as pd

# Use absolute path to the database
DB_DIR = r"C:\Users\Administrator\PycharmProjects\pythonProject\outputs"
DB_PATH = os.path.join(DB_DIR, 'backtests.db')

# Ensure the output directory exists
os.makedirs(DB_DIR, exist_ok=True)


def _ensure_db() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Enable WAL for better concurrent read
    try:
        conn.execute('PRAGMA journal_mode=WAL;')
    except Exception:
        pass
    _init_schema(conn)
    return conn


def _deduplicate_runs_by_param_key(conn: sqlite3.Connection, *, is_tuning: int = 0) -> None:
    """Ensure only one row per (is_tuning, param_key). Keep the earliest id."""
    try:
        cur = conn.cursor()
        # Find param_keys with duplicates
        cur.execute(
            """
            SELECT param_key
            FROM runs
            WHERE is_tuning = ? AND param_key IS NOT NULL
            GROUP BY param_key
            HAVING COUNT(1) > 1
            """,
            (int(is_tuning),),
        )
        keys = [row[0] for row in cur.fetchall()]
        for pk in keys:
            # Keep MIN(id), delete others
            cur.execute(
                "SELECT MIN(id) FROM runs WHERE is_tuning = ? AND param_key = ?",
                (int(is_tuning), pk),
            )
            keep_id = cur.fetchone()[0]
            conn.execute(
                "DELETE FROM runs WHERE is_tuning = ? AND param_key = ? AND id <> ?",
                (int(is_tuning), pk, int(keep_id)),
            )
        conn.commit()
    except Exception:
        pass


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cur.fetchall()]
        return column in cols
    except Exception:
        return False


def _init_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    # runs table: one row per backtest execution
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            strategy TEXT,
            params_json TEXT,
            stats_json TEXT,
            timeframes TEXT,
            signal_tf TEXT,
            backtest_tf TEXT,
            is_tuning INTEGER DEFAULT 0,
            param_key TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(is_tuning, param_key) ON CONFLICT IGNORE
        );
        """
    )
    
    # Add created_at and updated_at columns if they don't exist
    if not _has_column(conn, 'runs', 'created_at'):
        # First add the column without default
        cur.execute("ALTER TABLE runs ADD COLUMN created_at TIMESTAMP")
        # Then update existing rows to current timestamp
        cur.execute("UPDATE runs SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
        
    if not _has_column(conn, 'runs', 'updated_at'):
        # First add the column without default
        cur.execute("ALTER TABLE runs ADD COLUMN updated_at TIMESTAMP")
        # Then update existing rows to current timestamp
        cur.execute("UPDATE runs SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL")
    
    # Create trigger to update updated_at on row update if it doesn't exist
    try:
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS update_runs_timestamp
            AFTER UPDATE ON runs
            FOR EACH ROW
            BEGIN
                UPDATE runs SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;
        """)
    except sqlite3.OperationalError as e:
        if "already exists" not in str(e):
            raise
    # If legacy columns exist, migrate table to drop them (h1_bars/m1_bars, created_at)
    try:
        has_h1 = _has_column(conn, "runs", "h1_bars")
        has_m1 = _has_column(conn, "runs", "m1_bars")
        has_created = _has_column(conn, "runs", "created_at")
        if has_h1 or has_m1 or has_created:
            # Build new schema without h1_bars/m1_bars
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    strategy TEXT,
                    params_json TEXT,
                    stats_json TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    timeframes TEXT,
                    config_json TEXT,
                    code_version TEXT,
                    signal_tf TEXT,
                    backtest_tf TEXT,
                    is_tuning INTEGER,
                    param_key TEXT,
                    objective TEXT,
                    score REAL
                );
                """
            )
            # Copy data (map columns; missing new columns default NULL)
            cur.execute(
                """
                INSERT INTO runs_new(id, symbol, strategy, params_json, stats_json,
                                     start_time, end_time, timeframes, config_json, code_version,
                                     signal_tf, backtest_tf, is_tuning, param_key, objective, score)
                SELECT id, symbol, strategy, params_json, stats_json,
                       start_time, end_time, timeframes, config_json, code_version,
                       signal_tf, backtest_tf, is_tuning, param_key, objective, score
                FROM runs
                """
            )
            cur.execute("DROP TABLE runs")
            cur.execute("ALTER TABLE runs_new RENAME TO runs")
    except Exception:
        # Non-fatal: continue with additive migrations
        pass
    # Backward-compatible: add new columns if missing
    for col in [
        ("start_time", "TEXT"),
        ("end_time", "TEXT"),
        ("timeframes", "TEXT"),  # JSON/text like ["H1","M1"]
        ("config_json", "TEXT"),
        ("code_version", "TEXT"),
        ("signal_tf", "TEXT"),
        ("backtest_tf", "TEXT"),
        ("is_tuning", "INTEGER"),
        ("param_key", "TEXT"),
        ("objective", "TEXT"),
        ("score", "REAL"),
    ]:
        if not _has_column(conn, "runs", col[0]):
            try:
                cur.execute(f"ALTER TABLE runs ADD COLUMN {col[0]} {col[1]}")
            except Exception:
                pass

    # trades table: detailed trade records for each run
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            entry_time TEXT,
            exit_time TEXT,
            entry_price REAL,
            exit_price REAL,
            direction TEXT,
            net_pnl REAL,
            fee REAL,
            qty REAL,
            exit_reason TEXT,
            FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
        );
        """
    )
    # Backward-compatible: add symbol column to trades if missing
    if not _has_column(conn, "trades", "symbol"):
        try:
            cur.execute("ALTER TABLE trades ADD COLUMN symbol TEXT")
        except Exception:
            pass

    # orders table: orders and their status
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            symbol TEXT,
            timeframe TEXT,
            side TEXT, -- buy/sell
            entry_time TEXT,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            qty REAL,
            reason TEXT,
            status TEXT, -- filled/canceled/rejected/closed
            FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
        );
        """
    )

    # metrics table: aggregated metrics per run
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            run_id INTEGER PRIMARY KEY,
            total_pnl REAL,
            sharpe REAL,
            sortino REAL,
            max_dd REAL,
            calmar REAL,
            win_rate REAL,
            avg_r REAL,
            profit_factor REAL,
            exposure REAL,
            turnover REAL,
            avg_trade_duration REAL,
            FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
        );
        """
    )
    # Backward-compatible: add commonly used summary fields if missing
    for col, typ in [
        ("total_trades", "INTEGER"),
        ("winning_trades", "INTEGER"),
        ("losing_trades", "INTEGER"),
        ("total_fees", "REAL"),
        ("final_cash", "REAL"),
        ("average_trade", "REAL"),
        ("min_lot_size", "REAL"),
        ("max_lot_size", "REAL"),
        ("final_lot_size", "REAL"),
        ("total_return", "REAL"),
    ]:
        if not _has_column(conn, "metrics", col):
            try:
                cur.execute(f"ALTER TABLE metrics ADD COLUMN {col} {typ}")
            except Exception:
                pass

    # optional candles table for persistence/visualization
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            time TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, timeframe, time)
        );
        """
    )

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_symbol ON runs(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_param_key ON runs(param_key, id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_run_id ON orders(run_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol_time ON orders(symbol, entry_time)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id)")
    # If symbol column exists, add symbol/time index
    if _has_column(conn, "trades", "symbol"):
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, entry_time)")
        except Exception:
            pass
    conn.commit()

    # Backfill missing param_key for legacy rows and add unique index to avoid duplicates
    try:
        _backfill_param_keys(conn)
        # Ensure NULL -> 0 for is_tuning
        cur.execute("UPDATE runs SET is_tuning = 0 WHERE is_tuning IS NULL")
        # Deduplicate non-tuning by param_key, keep earliest id
        _deduplicate_runs_by_param_key(conn, is_tuning=0)
        # Unique index on (is_tuning, param_key) to make UPSERT possible
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_runs_tuning_param ON runs(is_tuning, param_key)")
        conn.commit()
    except Exception:
        pass

def _backfill_param_keys(conn: sqlite3.Connection) -> None:
    """Fill missing param_key values for legacy rows using current key builder.

    Also normalizes is_tuning NULL to 0 for compatibility with unique constraints.
    """
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        # Backfill param_key for rows where it's NULL
        cur.execute("SELECT id, symbol, strategy, params_json, timeframes FROM runs WHERE param_key IS NULL LIMIT 10000")
        rows = cur.fetchall()
        for row in rows:
            try:
                symbol = row["symbol"]
                strategy_name = row["strategy"]
                params = json.loads(row["params_json"] or '{}')
                tfl = []
                try:
                    tfl = json.loads(row["timeframes"] or '[]')
                except Exception:
                    tfl = []
                pk = _build_param_key(symbol, strategy_name, params, tfl)
                conn.execute("UPDATE runs SET param_key = ? WHERE id = ?", (pk, int(row["id"])) )
            except Exception:
                continue
        # Normalize is_tuning
        conn.execute("UPDATE runs SET is_tuning = 0 WHERE is_tuning IS NULL")
        conn.commit()
    except Exception:
        pass


def save_backtest_run(
    *,
    symbol: str,
    h1_bars: int,
    m1_bars: int,
    strategy_name: str,
    params: Dict[str, Any],
    stats: Dict[str, Any],
    trades_df: pd.DataFrame,
    update_existing: bool = True,  # 新增参数，控制是否更新已存在的记录
) -> int:
    """
    Persist a backtest run and its trades. 
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        h1_bars: Number of H1 bars used
        m1_bars: Number of M1 bars used
        strategy_name: Name of the strategy
        params: Strategy parameters
        stats: Backtest statistics
        trades_df: DataFrame containing trade data
        update_existing: If True, will update existing record with same parameters instead of creating new one
        
    Returns:
        int: The run_id of the created or updated run
    """
    conn = _ensure_db()
    try:
        with conn:
            # 提取时间框架参数
            sig_tf = params.get('signal_tf')
            bt_tf = params.get('backtest_tf')
            
            # 构建时间框架列表和JSON
            tfl = [x for x in [sig_tf, bt_tf] if x]
            tf_json = json.dumps(tfl, ensure_ascii=False, sort_keys=True, default=str) if tfl else json.dumps([], ensure_ascii=False)
            
            # 准备参数和统计数据的JSON
            params_json = json.dumps(params or {}, ensure_ascii=False, sort_keys=True, default=str)
            stats_json = json.dumps(stats or {}, ensure_ascii=False, sort_keys=True, default=str)
            
            # 构建唯一标识符
            pk = _build_param_key(symbol, strategy_name, params or {}, tfl)
            
            # 检查是否已存在相同参数的记录
            run_id = None
            if update_existing:
                cur = conn.execute(
                    """
                    SELECT id FROM runs 
                    WHERE (is_tuning IS NULL OR is_tuning = 0) 
                    AND param_key = ? 
                    ORDER BY id DESC LIMIT 1
                    """,
                    (pk,)
                )
                result = cur.fetchone()
                if result:
                    run_id = result[0]
            
            if run_id is not None and update_existing:
                # Update existing record
                cur = conn.execute(
                    """
                    UPDATE runs 
                    SET 
                        stats_json = ?,
                        updated_at = CURRENT_TIMESTAMP,
                        end_time = datetime('now')
                    WHERE id = ?
                    """,
                    (stats_json, run_id)
                )
            else:
                # Insert new record
                cur = conn.execute(
                    """
                    INSERT INTO runs (
                        symbol, strategy, params_json, stats_json,
                        start_time, end_time, timeframes, signal_tf, backtest_tf,
                        is_tuning, param_key
                    ) VALUES (?, ?, ?, ?, datetime('now'), datetime('now'), ?, ?, ?, 0, ?)
                    """,
                    (
                        symbol, strategy_name, params_json, stats_json,
                        tf_json, sig_tf, bt_tf, pk
                    )
                )
                run_id = cur.lastrowid
            
            # Save trades if provided
            if not trades_df.empty:
                insert_trades(run_id, trades_df, symbol)
            
            return run_id
    finally:
        conn.close()


def query_latest_tuning_by_param(
    *, symbol: Optional[str] = None, strategy_name: Optional[str] = None, limit: int = 200
) -> pd.DataFrame:
    """Return the latest tuning run per param_key (data aging aware)."""
    conn = _ensure_db()
    try:
        where = ["is_tuning = 1"]
        params: list[Any] = []
        if symbol:
            where.append("symbol = ?"); params.append(symbol)
        if strategy_name:
            where.append("strategy = ?"); params.append(strategy_name)
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
            WITH latest AS (
                SELECT param_key, MAX(id) AS max_id
                FROM runs
                {where_sql}
                GROUP BY param_key
            )
            SELECT r.*
            FROM runs r
            JOIN latest l ON r.id = l.max_id
            ORDER BY r.id DESC
            LIMIT ?
        """
        params.append(int(limit))
        df = pd.read_sql_query(sql, conn, params=tuple(params))
        return df
    finally:
        conn.close()


def get_runs(limit: int = 50) -> pd.DataFrame:
    """Fetch recent runs for inspection."""
    conn = _ensure_db()
    try:
        df = pd.read_sql_query(
            "SELECT id, symbol, strategy, signal_tf, backtest_tf FROM runs ORDER BY id DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )
        return df
    finally:
        conn.close()


def get_trades(run_id):
    """Return trades for a run as a DataFrame.
    
    Args:
        run_id: The ID of the run (can be int or str, will be converted to int)
    """
    try:
        # Ensure run_id is an integer
        run_id = int(run_id)
        conn = _ensure_db()
        
        # First check if the run exists
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,))
        if not cur.fetchone():
            print(f"Warning: No run found with ID {run_id}")
            return pd.DataFrame()
            
        # Get trades if they exist
        # First check if trades table exists and get its columns
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades';")
        if not cur.fetchone():
            print("Trades table does not exist")
            return pd.DataFrame()
            
        # Get column names from the trades table
        cur.execute("PRAGMA table_info(trades);")
        columns = [col[1] for col in cur.fetchall()]
        print(f"Available columns in trades table: {columns}")
        
        # Build the query with existing columns
        select_columns = "*"
        order_by = ""
        
        # If we have entry_time, use it for ordering, otherwise use id
        if 'entry_time' in columns:
            order_by = "ORDER BY entry_time"
        elif 'id' in columns:
            order_by = "ORDER BY id"
            
        query = f"SELECT {select_columns} FROM trades WHERE run_id = ? {order_by}"
        print(f"Executing query: {query} with run_id={run_id}")
        
        df = pd.read_sql(query, conn, params=(run_id,))
        
        if not df.empty:
            print(f"Found {len(df)} trades for run_id {run_id}")
            # Map column names if needed
            column_mapping = {
                'entry_time': 'open_time',
                'exit_time': 'close_time',
                'direction': 'direction',
                'entry_price': 'open_price',
                'exit_price': 'close_price',
                'size': 'size',
                'pnl': 'pnl',
                'return_pct': 'return_pct',
                'duration': 'duration'
            }
            
            # Rename columns to match expected format
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and new_col not in df.columns:
                    df[new_col] = df[old_col]
            
            # Ensure we have required columns
            required_columns = ['open_time', 'close_time', 'direction', 'open_price', 
                              'close_price', 'size', 'pnl', 'return_pct', 'duration']
            
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
                    
        return df
    except Exception as e:
        print(f"Error getting trades for run {run_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    finally:
        if 'conn' in locals():
            conn.close()


def query_runs(
    *,
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    symbol_like: Optional[str] = None,
    strategy_name: Optional[str] = None,
    min_win_rate: Optional[float] = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Query runs with optional filters. Time strings should be in 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD'.

    Note: min_win_rate is applied client-side by parsing stats_json.
    """
    conn = _ensure_db()
    try:
        clauses = []
        params: list[Any] = []
        if start_datetime:
            clauses.append("created_at >= ?")
            params.append(start_datetime)
        if end_datetime:
            clauses.append("created_at <= ?")
            params.append(end_datetime)
        if symbol_like:
            clauses.append("symbol LIKE ?")
            params.append(f"%{symbol_like}%")
        if strategy_name:
            clauses.append("strategy = ?")
            params.append(strategy_name)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT id, is_tuning, symbol, strategy, signal_tf, backtest_tf, params_json, stats_json "
            f"FROM runs {where} ORDER BY id DESC LIMIT ?"
        )
        params.append(int(limit))
        df = pd.read_sql_query(sql, conn, params=tuple(params))
        # Derive win_rate for optional filtering and display
        def _get_wr(x: str) -> float:
            try:
                d = json.loads(x or '{}')
                return float(d.get('win_rate', 0.0))
            except Exception:
                return 0.0
        def _get_pnl(x: str) -> float:
            try:
                d = json.loads(x or '{}')
                return float(d.get('total_pnl', 0.0))
            except Exception:
                return 0.0
        if not df.empty:
            df['win_rate'] = df['stats_json'].map(_get_wr)
            df['total_pnl'] = df['stats_json'].map(_get_pnl)
            if min_win_rate is not None:
                df = df[df['win_rate'] >= float(min_win_rate)]
        return df.reset_index(drop=True)
    finally:
        conn.close()


def query_runs_flat(
    *,
    is_tuning: Optional[int] = None,
    symbol_like: Optional[str] = None,
    strategy_name: Optional[str] = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Return runs with parameters expanded into columns for convenient sorting/filtering.

    Columns include:
    id, symbol, strategy, signal_tf, backtest_tf,
    stop_loss_usd, take_profit_usd, min_range_usd, qty, lot_mode, commission_per_10lots_usd,
    total_trades, winning_trades, losing_trades, win_rate, total_pnl, total_return, total_fees, final_cash
    """
    conn = _ensure_db()
    try:
        clauses = []
        params: list[Any] = []
        if is_tuning is not None:
            clauses.append("is_tuning = ?"); params.append(int(is_tuning))
        if symbol_like:
            clauses.append("symbol LIKE ?"); params.append(f"%{symbol_like}%")
        if strategy_name:
            clauses.append("strategy = ?"); params.append(strategy_name)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT id, symbol, strategy, signal_tf, backtest_tf, params_json, stats_json "
            f"FROM runs {where} ORDER BY id DESC LIMIT ?"
        )
        params.append(int(limit))
        df = pd.read_sql_query(sql, conn, params=tuple(params))
        if df.empty:
            return df
        # Parse params and stats into columns
        import json as _json
        def parse_json(s):
            try:
                return _json.loads(s or '{}')
            except Exception:
                return {}
        p = df['params_json'].map(parse_json)
        s = df['stats_json'].map(parse_json)
        # Parameter fields
        def getp(key, default=None):
            return p.map(lambda d: d.get(key, default))
        df['stop_loss_usd'] = getp('stop_loss_usd')
        df['take_profit_usd'] = getp('take_profit_usd')
        df['min_range_usd'] = getp('min_range_usd')
        df['qty'] = getp('qty')
        df['lot_mode'] = getp('lot_mode')
        df['commission_per_10lots_usd'] = getp('commission_per_10lots_usd')
        # Stats fields
        def gets(key, default=None):
            return s.map(lambda d: d.get(key, default))
        df['total_trades'] = gets('total_trades', 0)
        df['winning_trades'] = gets('winning_trades', 0)
        df['losing_trades'] = gets('losing_trades', 0)
        df['win_rate'] = gets('win_rate', 0.0)
        df['total_pnl'] = gets('total_pnl', 0.0)
        df['total_fees'] = gets('total_fees', 0.0)
        df['final_cash'] = gets('final_cash', None)
        df['total_return'] = gets('total_return', 0.0)
        # Reorder columns
        cols = [
            'id','is_tuning','symbol','strategy','signal_tf','backtest_tf',
            'stop_loss_usd','take_profit_usd','min_range_usd','qty','lot_mode','commission_per_10lots_usd',
            'total_trades','winning_trades','losing_trades','win_rate','total_pnl','total_fees','final_cash','total_return',
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols].reset_index(drop=True)
    finally:
        conn.close()


def get_db_status() -> Dict[str, Any]:
    """Return DB status without creating a new file.

    Fields: exists, path, runs_count, trades_count.
    If DB file doesn't exist, counts are 0 and no file is created.
    """
    status: Dict[str, Any] = {
        'path': os.path.abspath(DB_PATH),
        'exists': os.path.exists(DB_PATH),
        'runs_count': 0,
        'trades_count': 0,
    }
    if not status['exists']:
        return status
    # If exists, open and count
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM runs"); status['runs_count'] = int(cur.fetchone()[0])
        except Exception:
            status['runs_count'] = 0
        try:
            cur.execute("SELECT COUNT(*) FROM trades"); status['trades_count'] = int(cur.fetchone()[0])
        except Exception:
            status['trades_count'] = 0
        return status
    finally:
        conn.close()


# -------- High-level helper APIs for the new schema --------
def _build_param_key(symbol: str, strategy_name: str, params: Dict[str, Any], timeframes: list[str] | None = None) -> str:
    key_obj = {
        'symbol': symbol,
        'strategy': strategy_name,
        'timeframes': list(timeframes or []),
        'params': params or {},
    }
    return json.dumps(key_obj, ensure_ascii=False, sort_keys=True, default=str)

def begin_run(
    *,
    symbol: str,
    timeframes: list[str] | None = None,
    config: Dict[str, Any] | None = None,
    code_version: Optional[str] = None,
    start_time: Optional[str] = None,
    h1_bars: Optional[int] = None,
    m1_bars: Optional[int] = None,
    strategy_name: Optional[str] = None,
) -> int:
    """Create a runs row early and return run_id."""
    conn = _ensure_db()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO runs(symbol, h1_bars, m1_bars, strategy, params_json, stats_json, start_time, timeframes, config_json, code_version)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    symbol,
                    int(h1_bars) if h1_bars is not None else None,
                    int(m1_bars) if m1_bars is not None else None,
                    strategy_name,
                    json.dumps({}, ensure_ascii=False, sort_keys=True, default=str),
                    json.dumps({}, ensure_ascii=False, sort_keys=True, default=str),
                    start_time,
                    json.dumps(timeframes or [], ensure_ascii=False, sort_keys=True, default=str),
                    json.dumps(config or {}, ensure_ascii=False, sort_keys=True, default=str),
                    code_version,
                ),
            )
            return int(cur.lastrowid)
    finally:
        conn.close()


def finalize_run(
    *,
    run_id: int,
    end_time: Optional[str] = None,
    stats: Dict[str, Any] | None = None,
    metrics: Dict[str, Any] | None = None,
) -> None:
    """Finalize a run with end_time and optional stats/metrics."""
    conn = _ensure_db()
    try:
        with conn:
            if stats is not None or end_time is not None:
                conn.execute(
                    "UPDATE runs SET stats_json = COALESCE(?, stats_json), end_time = COALESCE(?, end_time) WHERE id = ?",
                    (
                        json.dumps(stats or {}, ensure_ascii=False, sort_keys=True, default=str) if stats is not None else None,
                        end_time,
                        int(run_id),
                    ),
                )
            if metrics is not None:
                # upsert metrics (SQLite: replace into on PK)
                fields = [
                    'total_pnl','sharpe','sortino','max_dd','calmar','win_rate','avg_r','profit_factor','exposure','turnover','avg_trade_duration'
                ]
                vals = [metrics.get(k) for k in fields]
                conn.execute(
                    """
                    INSERT INTO metrics(run_id, total_pnl, sharpe, sortino, max_dd, calmar, win_rate, avg_r, profit_factor, exposure, turnover, avg_trade_duration)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        total_pnl=excluded.total_pnl,
                        sharpe=excluded.sharpe,
                        sortino=excluded.sortino,
                        max_dd=excluded.max_dd,
                        calmar=excluded.calmar,
                        win_rate=excluded.win_rate,
                        avg_r=excluded.avg_r,
                        profit_factor=excluded.profit_factor,
                        exposure=excluded.exposure,
                        turnover=excluded.turnover,
                        avg_trade_duration=excluded.avg_trade_duration
                    """,
                    [int(run_id), *vals],
                )
    finally:
        conn.close()


def insert_orders(run_id: int, orders: list[Dict[str, Any]]) -> None:
    """Batch insert orders records."""
    if not orders:
        return
    conn = _ensure_db()
    try:
        with conn:
            rows = []
            for od in orders:
                rows.append(
                    (
                        int(run_id),
                        od.get('symbol'),
                        od.get('timeframe'),
                        od.get('side'),
                        od.get('entry_time'),
                        od.get('entry_price'),
                        od.get('stop_loss'),
                        od.get('take_profit'),
                        od.get('qty'),
                        od.get('reason'),
                        od.get('status'),
                    )
                )
            conn.executemany(
                """
                INSERT INTO orders(run_id, symbol, timeframe, side, entry_time, entry_price, stop_loss, take_profit, qty, reason, status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
    finally:
        conn.close()


def insert_trades(run_id: int, trades_df: pd.DataFrame, symbol: Optional[str] = None) -> None:
    """Batch insert trades; convenience if not using save_backtest_run."""
    if not isinstance(trades_df, pd.DataFrame) or trades_df.empty:
        return
    conn = _ensure_db()
    try:
        with conn:
            cols = ['entry_time','exit_time','entry_price','exit_price','direction','net_pnl','fee','qty','exit_reason']
            for c in cols:
                if c not in trades_df.columns:
                    trades_df[c] = None
            def _to_str(x):
                try:
                    return None if x is None else str(x)
                except Exception:
                    return None
            def _to_float(x):
                try:
                    if x is None or (isinstance(x, float) and (x != x)):
                        return None
                    return float(x)
                except Exception:
                    try:
                        return float(str(x))
                    except Exception:
                        return None
            cleaned_rows = []
            for _, rec in trades_df[cols].iterrows():
                cleaned_rows.append(
                    (
                        int(run_id),
                        _to_str(rec.get('entry_time')),
                        _to_str(rec.get('exit_time')),
                        _to_float(rec.get('entry_price')),
                        _to_float(rec.get('exit_price')),
                        _to_str(rec.get('direction')),
                        _to_float(rec.get('net_pnl')),
                        _to_float(rec.get('fee')),
                        _to_float(rec.get('qty')),
                        _to_str(rec.get('exit_reason')),
                    )
                )
            if _has_column(conn, 'trades', 'symbol') and symbol is not None:
                conn.executemany(
                    """
                    INSERT INTO trades(
                        run_id, entry_time, exit_time, entry_price, exit_price, direction, net_pnl, fee, qty, exit_reason, symbol
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    [row + (symbol,) for row in cleaned_rows],
                )
            else:
                conn.executemany(
                    """
                    INSERT INTO trades(
                        run_id, entry_time, exit_time, entry_price, exit_price, direction, net_pnl, fee, qty, exit_reason
                    ) VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    cleaned_rows,
                )
    finally:
        conn.close()


def update_or_insert_orders(symbol: str, strategy_name: str, new_orders: list[Dict[str, Any]], 
                         params: Dict[str, Any], timeframes: list[str] | None = None) -> int:
    """
    Update or insert order data, replacing only the most recent week's data.
    
    Args:
        symbol: Trading symbol (e.g., 'XAUUSD')
        strategy_name: Name of the strategy
        new_orders: List of order dictionaries to insert
        params: Strategy parameters
        timeframes: List of timeframes used (optional)
        
    Returns:
        int: The run_id of the updated or newly created run
    """
    conn = _ensure_db()
    cursor = conn.cursor()
    
    try:
        # Get current timestamp
        from datetime import datetime, timedelta
        now = datetime.utcnow()
        one_week_ago = now - timedelta(days=7)
        
        # Find existing run for these parameters within the last week
        cursor.execute("""
            SELECT id, created_at 
            FROM runs 
            WHERE symbol = ? 
              AND strategy_name = ? 
              AND param_key = ?
              AND created_at >= ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (symbol, strategy_name, _build_param_key(symbol, strategy_name, params, timeframes), 
             one_week_ago.strftime('%Y-%m-%d %H:%M:%S')))
        
        result = cursor.fetchone()
        
        if result:
            # Update existing run
            run_id = result[0]
            
            # Delete existing orders for this run
            cursor.execute("DELETE FROM orders WHERE run_id = ?", (run_id,))
            
            # Insert new orders
            if new_orders:
                insert_orders(run_id, new_orders)
                
            # Update the run's updated_at timestamp
            cursor.execute(
                "UPDATE runs SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (run_id,)
            )
            
            conn.commit()
            return run_id
            
        else:
            # Create a new run if no recent one exists
            run_id = begin_run(
                symbol=symbol,
                timeframes=timeframes,
                config=params,
                strategy_name=strategy_name
            )
            
            if new_orders:
                insert_orders(run_id, new_orders)
                
            return run_id
            
    except Exception as e:
        conn.rollback()
        raise Exception(f"Error updating/inserting orders: {str(e)}")
    finally:
        conn.close()


def save_candles(symbol: str, timeframe: str, df: pd.DataFrame) -> None:
    """Optional: persist candles for later visualization. Expects columns: time, open, high, low, close, volume."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    required = ['time','open','high','low','close','volume']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"candles df missing column: {c}")
    conn = _ensure_db()
    try:
        with conn:
            rows = []
            for _, r in df[required].iterrows():
                rows.append((symbol, timeframe, str(r['time']), float(r['open']), float(r['high']), float(r['low']), float(r['close']), float(r['volume'])))
            conn.executemany(
                """
                INSERT OR REPLACE INTO candles(symbol, timeframe, time, open, high, low, close, volume)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                rows,
            )
    finally:
        conn.close()
