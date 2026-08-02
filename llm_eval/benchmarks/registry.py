"""评测集注册表

自动扫描 benchmarks/ 下所有模块里的 Benchmark 子类并注册。
通过 get(name) 获取实例, list_benchmarks() 列出全部。
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List, Optional, Type

from ..models import BenchmarkMeta, Stage
from .base import Benchmark


_REGISTRY: Dict[str, Type[Benchmark]] = {}


def register(cls: Type[Benchmark]) -> Type[Benchmark]:
    """装饰器: 注册一个 Benchmark 子类 (按 meta.name 注册)"""
    # 先实例化拿 meta (meta 是类属性也可直接拿)
    meta = cls.META
    if meta is None:
        return cls
    _REGISTRY[meta.name] = cls
    return cls


def _autoload() -> None:
    """导入 benchmarks 包下所有子模块, 触发 @register"""
    import llm_eval.benchmarks as pkg
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        if mod_info.name in ("base", "registry", "__init__"):
            continue
        try:
            importlib.import_module(f"llm_eval.benchmarks.{mod_info.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[registry] 跳过模块 {mod_info.name}: {e}")


def get(name: str) -> Benchmark:
    if not _REGISTRY:
        _autoload()
    if name not in _REGISTRY:
        raise KeyError(
            f"未知评测集 '{name}'。可用: {', '.join(sorted(_REGISTRY.keys()))}"
        )
    return _REGISTRY[name]()


def list_benchmarks() -> List[BenchmarkMeta]:
    if not _REGISTRY:
        _autoload()
    metas = []
    for cls in _REGISTRY.values():
        metas.append(cls.META)
    # 按阶段 + 名称排序
    metas.sort(key=lambda m: (m.stage.value, m.name))
    return metas


def list_names(stage: Optional[Stage] = None) -> List[str]:
    metas = list_benchmarks()
    if stage:
        metas = [m for m in metas if m.stage == stage]
    return [m.name for m in metas]
