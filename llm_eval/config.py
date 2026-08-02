"""配置加载

从 yaml 读取配置, 转成 ModelConfig 列表 + 运行参数。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .models import ModelConfig

DEFAULT_CONFIG_PATHS = ["config.yaml", "config.example.yaml"]


@dataclass
class AppConfig:
    models: List[ModelConfig]
    judge: Optional[ModelConfig]
    run: Dict[str, Any]
    benchmarks: List[str]
    output: Dict[str, Any]
    raw: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        models = []
        for m in data.get("models", []):
            models.append(
                ModelConfig(
                    name=m["name"],
                    base_url=m["base_url"],
                    api_key=m["api_key"],
                    model=m["model"],
                    temperature=m.get("temperature", 0.0),
                    max_tokens=m.get("max_tokens", 2048),
                    extra=m.get("extra", {}),
                )
            )
        judge = None
        if data.get("judge"):
            j = data["judge"]
            judge = ModelConfig(
                name=j["name"],
                base_url=j["base_url"],
                api_key=j["api_key"],
                model=j["model"],
                temperature=j.get("temperature", 0.0),
                max_tokens=j.get("max_tokens", 256),
                extra=j.get("extra", {}),
            )
        return cls(
            models=models,
            judge=judge,
            run=data.get("run", {}),
            benchmarks=data.get("benchmarks", []),
            output=data.get("output", {"dir": "results", "save_raw": True}),
            raw=data,
        )


def load_config(path: Optional[str] = None) -> AppConfig:
    """加载配置文件。path 为空时依次尝试默认路径。"""
    candidates = [path] if path else DEFAULT_CONFIG_PATHS
    for p in candidates:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return AppConfig.from_dict(data)
    raise FileNotFoundError(
        f"未找到配置文件。请复制 config.example.yaml 为 config.yaml 并填入 API Key。"
        f" (尝试过: {candidates})"
    )


def get_run_params(config: AppConfig) -> Dict[str, Any]:
    """从 config.run 提取 runner 参数"""
    r = config.run
    return {
        "concurrency": r.get("concurrency", 4),
        "max_retries": r.get("max_retries", 3),
        "timeout": r.get("timeout", 120),
        "limit": r.get("limit"),
        "seed": r.get("seed", 42),
    }
