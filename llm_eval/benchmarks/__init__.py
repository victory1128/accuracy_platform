"""评测集插件包

约定: 每个子模块定义若干 Benchmark 子类并用 @register 注册。
新增评测集只需新建一个 .py 文件即可被 registry 自动发现。
"""
from .base import Benchmark
from .registry import register, get, list_benchmarks, list_names

__all__ = ["Benchmark", "register", "get", "list_benchmarks", "list_names"]
