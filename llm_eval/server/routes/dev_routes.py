"""开发模式路由 + 评测集目录

- GET  /api/benchmarks           评测集目录 (前端勾选用)
- POST /api/dev/dry-run          dry-run 模拟跑通流程 (不入任务队列, 同步返回简报)
- POST /api/dev/quick-sample     快速小样本试跑 (建任务, limit 小)
- POST /api/dev/reload-benchmarks 热加载评测集插件 (清注册表 + importlib.reload)
- GET  /api/dev/health           平台自检
"""
from __future__ import annotations

import importlib
import pkgutil
import threading
from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from .. import auth, db, taskman
from ..schemas import DryRunIn, Message, QuickSampleIn, TaskOut
from ...benchmarks import list_benchmarks, list_names
from ...benchmarks import registry as _registry
from ...models import ModelConfig

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


@router.post("/api/dev/dry-run")
def dev_dry_run(body: DryRunIn, user: dict = Depends(auth.current_user)):
    """dry-run: 同步跑通流程, 返回每集简报。不写报告 (供快速验证)。"""
    from ...cli import _dry_run
    model = ModelConfig(name=body.model_name, base_url="", api_key="", model=body.model_name)
    try:
        results = _dry_run(model, body.benchmarks, body.limit)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    brief = []
    for r in results:
        agg = r.aggregate
        score = "-"
        for k in ("accuracy", "pass_at_1", "instruction_following_rate"):
            if k in agg and agg[k] is not None:
                score = f"{agg[k]*100:.1f}%"; break
        if "score_100" in agg and agg["score_100"] is not None:
            score = f"{agg['score_100']}/100"
        brief.append({
            "benchmark": r.benchmark_meta.display_name,
            "num_samples": r.num_samples,
            "score": score,
        })
    return {"ok": True, "brief": brief}


@router.post("/api/dev/quick-sample", response_model=TaskOut)
def dev_quick_sample(body: QuickSampleIn, user: dict = Depends(auth.current_user)):
    """快速小样本试跑: 建一个 mode=quick 的任务 (limit 默认 5)。"""
    if not body.benchmarks:
        from fastapi import HTTPException
        raise HTTPException(400, "请选择评测集")
    if not body.model_cfg.api_key:
        from fastapi import HTTPException
        raise HTTPException(400, "请填写 API Key")
    model = ModelConfig(
        name=body.model_cfg.name or body.model_cfg.model,
        base_url=body.model_cfg.base_url, api_key=body.model_cfg.api_key,
        model=body.model_cfg.model, temperature=body.model_cfg.temperature,
        max_tokens=body.model_cfg.max_tokens, extra=body.model_cfg.extra,
    )
    judge = None
    if body.use_judge and body.judge_config and body.judge_config.api_key:
        judge = ModelConfig(
            name=body.judge_config.name, base_url=body.judge_config.base_url,
            api_key=body.judge_config.api_key, model=body.judge_config.model,
            max_tokens=body.judge_config.max_tokens,
        )
    run_params = {"limit": body.limit, "concurrency": 4, "streaming": body.streaming, "debug": True}
    name = f"[试跑] {model.model} · {','.join(body.benchmarks[:3])}"
    task_id = db.create_task(
        user_id=user["id"], name=name, mode="quick",
        model_config={k: v for k, v in body.model_cfg.model_dump().items() if k != "api_key"},
        judge_config=None,
        benchmarks=body.benchmarks, run_params=run_params,
    )
    taskman.enqueue(task_id, model, judge, threading.Event())
    from ..routes.task_routes import _task_out
    return _task_out(db.get_task(task_id))


@router.post("/api/dev/reload-benchmarks")
def dev_reload_benchmarks(user: dict = Depends(auth.require_admin)):
    """热加载评测集插件: 清注册表, 重新 import 所有模块。

    用于开发时新增/修改评测集后不重启服务即生效。返回新目录 + 导入错误。
    """
    import llm_eval.benchmarks as pkg
    errors = []
    # 清空注册表
    _registry._REGISTRY.clear()
    # 重新 import 每个子模块 (触发 @register)
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        if mod_info.name in ("base", "registry", "__init__"):
            continue
        mod_name = f"llm_eval.benchmarks.{mod_info.name}"
        try:
            if mod_name in sys_modules():
                importlib.reload(sys_modules()[mod_name])
            else:
                importlib.import_module(mod_name)
        except Exception as e:  # noqa: BLE001
            errors.append({"module": mod_info.name, "error": str(e)})
    return {
        "ok": True,
        "count": len(list_names()),
        "benchmarks": [_bench_meta_dict(m) for m in list_benchmarks()],
        "errors": errors,
    }


def sys_modules():
    import sys
    return sys.modules


@router.get("/api/dev/health")
def health(user: dict = Depends(auth.require_admin)):
    """平台自检 (仅 admin)。"""
    return {
        "ok": True,
        "users": db.count_users(),
        "tasks_by_status": db.count_tasks_by_status(),
        "benchmarks": len(list_names()),
        "queue_size": len(taskman._QUEUE),
        "inflight": len(taskman._INFLIGHT),
    }
