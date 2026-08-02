"""SQLite 持久层

单文件 SQLite, 无需额外服务。所有 DDL 集中在 SCHEMA, 启动时自动建表/迁移。
连接走线程局部 (每个线程一个连接), 因为 worker 在线程里跑。

表:
- users      用户 (含 role: admin|user)
- sessions   登录会话 (token -> user_id, 过期时间)
- tasks      评测任务 (核心表; model/judge config 存 json 但不含 api_key)
- task_logs  任务日志 (开发模式调试追踪)

约定:
- 时间一律存 ISO 字符串 (与现有 started_at/finished_at 一致)
- json 字段用 TEXT 存 json.dumps(..., ensure_ascii=False) 的结果
- api_key 绝不进任何表
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

# 表结构 (一条 SQL 建一张表; IF NOT EXISTS 保证可重复执行)
# 注: SQLite 的 TEXT 存 ISO 时间; role 取值 'admin'/'user'; status 取值见 taskman
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'user',   -- admin | user
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token      TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_user  ON sessions(user_id);

CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name          TEXT NOT NULL,
    mode          TEXT NOT NULL DEFAULT 'real',      -- real | dry_run | quick
    model_config  TEXT NOT NULL,                     -- json, 不含 api_key
    judge_config  TEXT,                              -- json, 不含 api_key (可空)
    benchmarks    TEXT NOT NULL,                     -- json: ["mmlu","gsm8k",...]
    run_params    TEXT NOT NULL,                     -- json: {limit,concurrency,streaming,...}
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending|running|done|failed|cancelled
    summary       TEXT,                              -- json: [{benchmark,stage,score,num_samples}]
    error         TEXT,
    report_path   TEXT,                              -- 相对 results 目录的报告名 (无扩展名 stem)
    num_samples   INTEGER NOT NULL DEFAULT 0,
    progress      TEXT,                              -- json: {bench: {done,total,pct,status}}
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    deleted_at    TEXT                            -- 软删除时间戳; NULL=正常, 非空=在回收站
);
CREATE INDEX IF NOT EXISTS idx_tasks_user   ON tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
-- idx_tasks_deleted 由 _migrate 创建 (旧库无 deleted_at 列时建索引会报错)

CREATE TABLE IF NOT EXISTS task_logs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    ts      TEXT NOT NULL,
    level   TEXT NOT NULL DEFAULT 'info',  -- info|warn|error|debug
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_task ON task_logs(task_id, id);
"""

_DB_PATH: Optional[str] = None
_local = threading.local()  # 线程局部连接


def init_db(db_path: str) -> None:
    """初始化数据库 (建表/迁移)。进程级设置一次 db_path。

    若 db_path 变更 (如测试切库), 清掉所有线程局部缓存连接, 令后续重连。
    """
    global _DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    prev = _DB_PATH
    _DB_PATH = db_path
    if prev and os.path.abspath(prev) != os.path.abspath(db_path):
        # 切库: 关闭并清空所有已缓存的线程局部连接
        _reset_local_conns()
    # 用独立连接建表, 不污染线程局部缓存 (建完即关)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """增量迁移: 给旧库补新加的列 (CREATE TABLE IF NOT EXISTS 不会改已有表结构)。"""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    if "deleted_at" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN deleted_at TEXT")
    if "progress" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN progress TEXT")
    # 索引对新旧库都建 (IF NOT EXISTS 幂等)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_deleted ON tasks(deleted_at)")


def _reset_local_conns() -> None:
    """关闭并清空当前线程的缓存连接 (切库时用)。"""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        _local.conn = None


def db_path() -> Optional[str]:
    return _DB_PATH


def _connect() -> sqlite3.Connection:
    """当前线程的连接 (启用外键 + Row 工厂)。"""
    if _DB_PATH is None:
        raise RuntimeError("数据库未初始化, 请先调用 init_db()")
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        _local.conn = conn
    return conn


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="seconds")


# ----------------------------- 用户 -----------------------------
def create_user(username: str, password_hash: str, role: str = "user") -> int:
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, role, is_active, created_at) VALUES (?,?,?,?,?)",
        (username, password_hash, role, 1, _now_iso()),
    )
    conn.commit()
    return cur.lastrowid


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    row = _connect().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_name(username: str) -> Optional[Dict[str, Any]]:
    row = _connect().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def list_users() -> List[Dict[str, Any]]:
    rows = _connect().execute("SELECT * FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def update_user_role(user_id: int, role: str) -> bool:
    conn = _connect()
    cur = conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
    conn.commit()
    return cur.rowcount > 0


def set_user_active(user_id: int, is_active: bool) -> bool:
    conn = _connect()
    cur = conn.execute("UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, user_id))
    conn.commit()
    return cur.rowcount > 0


def set_user_password(user_id: int, password_hash: str) -> bool:
    """更新用户密码哈希 (用户自己改 / 管理员重置都用这个)。"""
    conn = _connect()
    cur = conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
    conn.commit()
    return cur.rowcount > 0


def count_users() -> int:
    row = _connect().execute("SELECT COUNT(*) AS n FROM users").fetchone()
    return row["n"] if row else 0


# ----------------------------- 会话 -----------------------------
def create_session(user_id: int, token: str, expires_at: str) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO sessions (user_id, token, created_at, expires_at) VALUES (?,?,?,?)",
        (user_id, token, _now_iso(), expires_at),
    )
    conn.commit()


def get_session(token: str) -> Optional[Dict[str, Any]]:
    row = _connect().execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    conn = _connect()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()


def delete_expired_sessions() -> int:
    """清理过期会话, 返回删除条数。"""
    conn = _connect()
    cur = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (_now_iso(),))
    conn.commit()
    return cur.rowcount


# ----------------------------- 任务 -----------------------------
def create_task(
    user_id: int,
    name: str,
    mode: str,
    model_config: Dict[str, Any],
    judge_config: Optional[Dict[str, Any]],
    benchmarks: List[str],
    run_params: Dict[str, Any],
) -> int:
    """新建任务 (status=pending)。model_config/judge_config 里不得含 api_key。"""
    conn = _connect()
    cur = conn.execute(
        """INSERT INTO tasks
           (user_id, name, mode, model_config, judge_config, benchmarks, run_params,
            status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            user_id, name, mode,
            json.dumps(model_config, ensure_ascii=False),
            json.dumps(judge_config, ensure_ascii=False) if judge_config else None,
            json.dumps(benchmarks, ensure_ascii=False),
            json.dumps(run_params, ensure_ascii=False),
            "pending", _now_iso(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_task(task_id: int, include_deleted: bool = False) -> Optional[Dict[str, Any]]:
    sql = "SELECT * FROM tasks WHERE id = ?"
    if not include_deleted:
        sql += " AND deleted_at IS NULL"
    row = _connect().execute(sql, (task_id,)).fetchone()
    return _decode_task(dict(row)) if row else None


def list_tasks(
    user_id: Optional[int] = None,
    status: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    include_deleted: bool = False,
) -> List[Dict[str, Any]]:
    """列任务。

    - user_id=None 表示全部 (admin 用)
    - status: 按状态筛选
    - query: 关键词, 模糊搜 name 和 benchmarks (json 文本)
    - limit=0 表示全部 (不加 LIMIT 子句); 否则 LIMIT ? OFFSET ?
    - include_deleted=False (默认) 排除回收站任务
    """
    sql = "SELECT * FROM tasks"
    clauses, params = [], []
    if not include_deleted:
        clauses.append("deleted_at IS NULL")
    if user_id is not None:
        clauses.append("user_id = ?"); params.append(user_id)
    if status:
        clauses.append("status = ?"); params.append(status)
    if query:
        clauses.append("(name LIKE ? OR benchmarks LIKE ?)")
        kw = f"%{query}%"
        params.extend([kw, kw])
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC"
    if limit and limit > 0:
        sql += " LIMIT ? OFFSET ?"
        params.append(limit)
        params.append(offset)
    rows = _connect().execute(sql, params).fetchall()
    return [_decode_task(dict(r)) for r in rows]


def count_tasks(
    user_id: Optional[int] = None,
    status: Optional[str] = None,
    query: Optional[str] = None,
    include_deleted: bool = False,
) -> int:
    """统计任务数 (与 list_tasks 同过滤条件, 用于分页总数)。"""
    sql = "SELECT COUNT(*) AS n FROM tasks"
    clauses, params = [], []
    if not include_deleted:
        clauses.append("deleted_at IS NULL")
    if user_id is not None:
        clauses.append("user_id = ?"); params.append(user_id)
    if status:
        clauses.append("status = ?"); params.append(status)
    if query:
        clauses.append("(name LIKE ? OR benchmarks LIKE ?)")
        kw = f"%{query}%"
        params.extend([kw, kw])
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    row = _connect().execute(sql, params).fetchone()
    return row["n"] if row else 0


def update_task_status(
    task_id: int,
    status: str,
    *,
    error: Optional[str] = None,
    summary: Optional[List[Dict[str, Any]]] = None,
    report_path: Optional[str] = None,
    num_samples: Optional[int] = None,
    started: bool = False,
    finished: bool = False,
) -> None:
    """更新任务状态与相关字段。只更新非 None 的字段。"""
    sets, params = ["status = ?"], [status]
    if error is not None:
        sets.append("error = ?"); params.append(error)
    if summary is not None:
        sets.append("summary = ?"); params.append(json.dumps(summary, ensure_ascii=False))
    if report_path is not None:
        sets.append("report_path = ?"); params.append(report_path)
    if num_samples is not None:
        sets.append("num_samples = ?"); params.append(num_samples)
    if started:
        sets.append("started_at = ?"); params.append(_now_iso())
    if finished:
        sets.append("finished_at = ?"); params.append(_now_iso())
    params.append(task_id)
    conn = _connect()
    conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", params)
    conn.commit()


def update_task_progress(task_id: int, progress: Dict[str, Any]) -> None:
    """更新任务进度 (各评测集的 done/total/pct/status)。存 JSON, 供详情页展示。"""
    conn = _connect()
    conn.execute("UPDATE tasks SET progress = ? WHERE id = ?",
                 (json.dumps(progress, ensure_ascii=False), task_id))
    conn.commit()


def delete_task(task_id: int) -> bool:
    """软删除: 标记 deleted_at, 进回收站。30 天后由 purge_expired 彻底清除。"""
    conn = _connect()
    cur = conn.execute(
        "UPDATE tasks SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL",
        (_now_iso(), task_id),
    )
    conn.commit()
    return cur.rowcount > 0


def restore_task(task_id: int) -> bool:
    """从回收站恢复任务 (清 deleted_at)。"""
    conn = _connect()
    cur = conn.execute(
        "UPDATE tasks SET deleted_at = NULL WHERE id = ? AND deleted_at IS NOT NULL",
        (task_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def hard_delete_task(task_id: int) -> Optional[Dict[str, Any]]:
    """彻底删除任务 (从回收站永久清除)。返回被删任务 (含 report_path, 供清理文件)。"""
    conn = _connect()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if not row:
        return None
    t = _decode_task(dict(row))
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    return t


def list_trashed_tasks(limit: int = 500) -> List[Dict[str, Any]]:
    """列回收站任务 (deleted_at 非空), 按删除时间倒序。仅管理岗用。"""
    rows = _connect().execute(
        "SELECT * FROM tasks WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_decode_task(dict(r)) for r in rows]


def purge_expired(days: int = 30) -> List[Dict[str, Any]]:
    """彻底清除回收站里超过 days 天的任务。返回被删任务列表 (供清理报告文件)。"""
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM tasks WHERE deleted_at IS NOT NULL AND deleted_at < ?",
        (cutoff,),
    ).fetchall()
    expired = [_decode_task(dict(r)) for r in rows]
    if expired:
        ids = [str(t["id"]) for t in expired]
        conn.execute(f"DELETE FROM tasks WHERE id IN ({','.join('?' * len(ids))})", ids)
        conn.commit()
    return expired


def empty_trash() -> List[Dict[str, Any]]:
    """清空回收站: 彻底删除全部已软删除任务。返回被删任务列表 (含 report_path, 供清理报告文件)。"""
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM tasks WHERE deleted_at IS NOT NULL ORDER BY deleted_at DESC"
    ).fetchall()
    deleted = [_decode_task(dict(r)) for r in rows]
    if deleted:
        ids = [str(t["id"]) for t in deleted]
        conn.execute(f"DELETE FROM tasks WHERE id IN ({','.join('?' * len(ids))})", ids)
        conn.commit()
    return deleted


def count_tasks_by_status() -> Dict[str, int]:
    """按状态聚合任务数 (admin 监控用, 不含回收站)。"""
    rows = _connect().execute(
        "SELECT status, COUNT(*) AS n FROM tasks WHERE deleted_at IS NULL GROUP BY status"
    ).fetchall()
    return {r["status"]: r["n"] for r in rows}


def _decode_task(row: Dict[str, Any]) -> Dict[str, Any]:
    """把任务行里的 json TEXT 字段解码回 Python 对象。"""
    for k in ("model_config", "judge_config", "benchmarks", "run_params", "summary", "progress"):
        v = row.get(k)
        if isinstance(v, str):
            try:
                row[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                pass
    return row


# ----------------------------- 任务日志 -----------------------------
def add_log(task_id: int, level: str, message: str) -> None:
    conn = _connect()
    conn.execute(
        "INSERT INTO task_logs (task_id, ts, level, message) VALUES (?,?,?,?)",
        (task_id, _now_iso(), level, message),
    )
    conn.commit()


def list_logs(task_id: int, after_id: int = 0, limit: int = 1000) -> List[Dict[str, Any]]:
    rows = _connect().execute(
        "SELECT * FROM task_logs WHERE task_id = ? AND id > ? ORDER BY id LIMIT ?",
        (task_id, after_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]
