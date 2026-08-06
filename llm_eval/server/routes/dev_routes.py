"""评测集目录路由

- GET  /api/benchmarks           评测集目录 (前端勾选用)

开发模式路由 (dry-run / quick-sample / reload / health) 已暂时移除。
如需恢复, 见 git 历史中本文件的旧版本。
"""
from __future__ import annotations

from fastapi import APIRouter

from ...benchmarks import list_benchmarks

router = APIRouter(tags=["dev"])


def _bench_meta_dict(m) -> dict:
    # 数该评测集的全量样本条数 (留空 limit 时跑多少条)。
    # 直接数 jsonl 文件行数 (快且省内存); 失败再退回 load_samples 计数。
    # 对长文本评测集 (mrcr 单条 389 万字符) 尤其重要——load_samples 会把全部加载进内存。
    num_samples = 0
    try:
        import os
        from ...benchmarks._data import data_path
        path = data_path(f"{m.name}.jsonl")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                num_samples = sum(1 for line in f if line.strip())
        else:
            from ...benchmarks import get as get_benchmark
            b = get_benchmark(m.name)
            num_samples = len(b.load_samples(limit=None, seed=42))
    except Exception:
        num_samples = 0
    return {
        "name": m.name,
        "display_name": m.display_name,
        "stage": m.stage.value,
        "task_type": m.task_type.value,
        "description": m.description,
        "tags": m.tags,
        "needs_judge": m.needs_judge,
        "source": m.source,
        "num_samples": num_samples,
    }


@router.get("/api/benchmarks")
def benchmarks_catalog():
    """评测集目录 (无需登录, 注册页也要展示可选集)。"""
    return [_bench_meta_dict(m) for m in list_benchmarks()]
