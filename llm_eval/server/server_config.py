"""服务端配置

从 config.yaml 的 server 段读取平台配置 (独立于现有的 models/judge/run 段),
保持原有 load_config 不变。若 server 段缺失则用安全默认值。

config.yaml 示例:
  server:
    host: "0.0.0.0"
    port: 8765
    db_path: "data/platform.db"
    results_dir: "results"
    admin:                 # 首次启动引导管理员 (之后可在后台改)
      username: "admin"
      password: "change-me"
    code_exec:
      sandbox: "disabled"  # disabled (默认最安全) | subprocess (仅开发) | docker
    cors_origins: []       # 空=同源, 不开 CORS
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml

_DEFAULTS = {
    "host": "0.0.0.0",
    "port": 8765,
    "db_path": "data/platform.db",
    "results_dir": "results",
    "code_exec_sandbox": "disabled",
    "cors_origins": [],
}
_CONFIG_PATHS = ["config.yaml", "config.example.yaml"]


@dataclass
class ServerConfig:
    host: str = _DEFAULTS["host"]
    port: int = _DEFAULTS["port"]
    db_path: str = _DEFAULTS["db_path"]
    results_dir: str = _DEFAULTS["results_dir"]
    code_exec_sandbox: str = _DEFAULTS["code_exec_sandbox"]
    # docker 沙箱配置 (sandbox=docker 时生效)
    code_exec_image: str = "python:3.11-slim"   # 执行镜像 (可换带预装包的镜像)
    code_exec_memory: str = "256m"              # 容器内存上限
    code_exec_cpus: str = "1.0"                 # 容器 CPU 上限
    code_exec_timeout: float = 15.0             # 单次执行超时 (秒)
    cors_origins: List[str] = field(default_factory=list)
    admin_username: Optional[str] = None
    admin_password: Optional[str] = None

    @property
    def code_exec_enabled(self) -> bool:
        """是否允许执行模型生成的代码 (HumanEval/MBPP)。

        disabled 时 code 类评测集会被拒绝提交 (非 admin) / 跳过执行仅判语法。
        subprocess/docker 才真正跑代码 (仅 admin 可提交)。
        """
        return self.code_exec_sandbox in ("subprocess", "docker")


def load_server_config(path: Optional[str] = None) -> ServerConfig:
    """加载 server 段。path 为空时尝试默认路径。缺失则全默认。"""
    candidates = [path] if path else _CONFIG_PATHS
    data = {}
    for p in candidates:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            break
    srv = data.get("server", {}) or {}
    admin = srv.get("admin", {}) or {}
    ce = srv.get("code_exec", {}) or {}
    return ServerConfig(
        host=srv.get("host", _DEFAULTS["host"]),
        port=int(srv.get("port", _DEFAULTS["port"])),
        db_path=srv.get("db_path", _DEFAULTS["db_path"]),
        results_dir=srv.get("results_dir", _DEFAULTS["results_dir"]),
        code_exec_sandbox=ce.get("sandbox", _DEFAULTS["code_exec_sandbox"]),
        code_exec_image=ce.get("image", "python:3.11-slim"),
        code_exec_memory=ce.get("memory", "256m"),
        code_exec_cpus=ce.get("cpus", "1.0"),
        code_exec_timeout=float(ce.get("timeout", 15.0)),
        cors_origins=srv.get("cors_origins", []) or [],
        admin_username=admin.get("username"),
        admin_password=admin.get("password"),
    )
