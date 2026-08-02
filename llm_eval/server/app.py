"""FastAPI 应用工厂

启动流程:
1. 加载 server 配置 (config.yaml 的 server 段)
2. 初始化 SQLite (建表)
3. 引导首个 admin (若 server.admin 配置了且该用户不存在)
4. 启动任务 worker 线程
5. 挂载路由 (auth/task/dev/admin) + 静态 SPA
6. (可选) 开 CORS

运行:
  .venv/bin/python -m llm_eval.server            # 默认从 config.yaml 读 server 段
  uvicorn llm_eval.server.app:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import auth, db, taskman
from .server_config import ServerConfig, load_server_config

HERE = os.path.dirname(__file__)
STATIC_DIR = os.path.join(HERE, "static")


def _bootstrap_admin(scfg: ServerConfig) -> None:
    """首次启动引导超级管理员: 若配置了 server.admin 且该用户不存在, 则创建为 super。

    兼容旧库: 若已有用户但没有任何 super (旧版引导的是 admin), 把配置的同名 admin
    或第一个 admin 升级为 super, 保证平台至少有一个超级管理员 (防锁死)。
    """
    if not scfg.admin_username or not scfg.admin_password:
        return
    existing = db.get_user_by_name(scfg.admin_username)
    if existing:
        # 已存在: 旧版可能建成了 admin, 升级为 super (不改密码)
        if existing["role"] != "super":
            db.update_user_role(existing["id"], "super")
        return
    if db.count_users() > 0:
        # 用户名变了但库里有用户: 确保至少有一个 super
        users = db.list_users()
        if not any(u["role"] == "super" for u in users):
            # 把第一个 admin 升级为 super (若无 admin 则第一个用户)
            cand = next((u for u in users if u["role"] == "admin"), users[0])
            db.update_user_role(cand["id"], "super")
        return
    # 冷启动: 创建超级管理员
    db.create_user(scfg.admin_username, auth.hash_password(scfg.admin_password), "super")


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """构造 FastAPI 应用。config_path 指向 config.yaml (测试可注入)。"""
    scfg = load_server_config(config_path)

    # 1. 数据库
    db.init_db(scfg.db_path)
    # 2. 引导 admin
    _bootstrap_admin(scfg)
    # 3. 启动 worker
    taskman.start_worker()
    # 4. 启动时清理回收站里超过 30 天的任务 (顺带删报告文件)
    try:
        from .routes.admin_routes import _purge_and_clean
        _purge_and_clean()
    except Exception:
        pass
    # 5. 注入代码沙箱配置 (HumanEval/MBPP 执行模型代码的隔离环境)
    from ..scoring.code_exec import configure_sandbox
    configure_sandbox(
        mode=scfg.code_exec_sandbox if scfg.code_exec_enabled else "subprocess",
        image=scfg.code_exec_image,
        memory=scfg.code_exec_memory,
        cpus=scfg.code_exec_cpus,
        timeout=scfg.code_exec_timeout,
    )

    app = FastAPI(title="大模型精度测试平台", version="1.0.0")

    # 3. GZip 压缩: 大响应 (如 /requests 明细 JSON, 报告 HTML) 压缩后传输快数倍。
    #    minimum_size=1024: 仅压缩 >1KB 的响应 (小响应不压缩, 省 CPU)。
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # 4. CORS (默认不开; 配了 origins 才加)
    if scfg.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=scfg.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 5. 路由
    from .routes import auth_routes, task_routes, dev_routes, admin_routes
    app.include_router(auth_routes.router)
    app.include_router(task_routes.router)
    app.include_router(dev_routes.router)
    app.include_router(admin_routes.router)

    # 6. 静态 SPA (前端): / 挂载 static/index.html, /static/* 静态资源
    if os.path.isdir(STATIC_DIR):
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        """前端 SPA 入口。app.js 引用带 mtime 版本号, 强制浏览器在更新后重新拉取。"""
        idx = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(idx):
            html = open(idx, encoding="utf-8").read()
            # 用 app.js 的修改时间做版本号, 避免浏览器缓存旧版前端
            js = os.path.join(STATIC_DIR, "app.js")
            css = os.path.join(STATIC_DIR, "style.css")
            rpt = os.path.join(STATIC_DIR, "rpt_table.js")
            v = str(int(os.path.getmtime(js))) if os.path.exists(js) else "1"
            cv = str(int(os.path.getmtime(css))) if os.path.exists(css) else "1"
            rv = str(int(os.path.getmtime(rpt))) if os.path.exists(rpt) else "1"
            html = html.replace("/static/app.js", f"/static/app.js?v={v}")
            html = html.replace("/static/style.css", f"/static/style.css?v={cv}")
            html = html.replace("/static/rpt_table.js", f"/static/rpt_table.js?v={rv}")
            return HTMLResponse(html)
        return JSONResponse({"message": "前端未构建 (static/index.html 缺失). API 已就绪: /docs"}, status_code=200)

    @app.get("/api/server-info")
    def server_info():
        """公开的服务端信息 (前端判断沙箱/特性用)。"""
        return {
            "version": "1.0.0",
            "code_exec_enabled": scfg.code_exec_enabled,
            "code_exec_sandbox": scfg.code_exec_sandbox,
            "benchmarks_count": len(__import__("llm_eval.benchmarks", fromlist=["list_names"]).list_names()),
        }

    return app


# 模块级 app (供 uvicorn llm_eval.server.app:app 直接用)
app = create_app()


def main():
    import uvicorn
    scfg = load_server_config()
    uvicorn.run(app, host=scfg.host, port=scfg.port)
