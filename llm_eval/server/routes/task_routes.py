"""任务路由: 提交/列表/详情/删除/克隆/日志/报告/进度SSE

API key 处理: 提交时含 key, 落库前 _strip_key 去掉 (DB 永不存 key);
key 只随任务对象进 taskman 内存队列, 运行结束丢弃。克隆返回的配置无 key。
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from .. import auth, db, taskman
from ..server_config import load_server_config
from ..schemas import CloneOut, Message, TaskCreateIn, TaskOut, LogLine, PageOut
from ...models import ModelConfig

router = APIRouter(prefix="/api/tasks", tags=["tasks"])

# 需要代码执行的评测集 (沙箱限制: 未启用沙箱时拒绝提交)
_CODE_BENCHES = {"humaneval", "mbpp", "bigcodebench", "livecodebench", "evalplus", "ds1000", "swebench"}


def _strip_key(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """去掉配置里的 api_key, 用于落库/回显。"""
    c = dict(cfg)
    c.pop("api_key", None)
    return c


def _to_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    """把请求里的模型配置 dict 转成 ModelConfig (含 key, 内存用)。

    max_tokens/temperature 为 None (用户留空) 时填默认值到 ModelConfig;
    真正的"覆盖评测集"逻辑在 taskman 用 run_params.max_tokens 走 override_params,
    与这里的 ModelConfig 值无关。
    """
    return ModelConfig(
        name=cfg.get("name") or cfg.get("model", ""),
        base_url=cfg.get("base_url", ""),
        api_key=cfg.get("api_key", ""),
        model=cfg.get("model", ""),
        temperature=cfg.get("temperature") if cfg.get("temperature") is not None else 0.0,
        max_tokens=cfg.get("max_tokens") if cfg.get("max_tokens") is not None else 2048,
        extra=cfg.get("extra", {}) or {},
    )


def _task_out(t: dict) -> TaskOut:
    # task dict 用 DB 列名 (model_config), TaskOut 字段是 model_cfg (alias=model_config)
    # populate_by_name=True 允许用 alias 构造, 故直接传 model_config
    return TaskOut.model_validate(t)


@router.post("", response_model=TaskOut)
def create_task(body: TaskCreateIn, user: dict = Depends(auth.current_user)):
    """提交评测任务。api_key 仅内存, 不落库。"""
    if not body.benchmarks:
        raise HTTPException(400, "请至少选择一个评测集")
    # api_key 可选: 本地/自建端点无需鉴权时可留空 (real 模式仍建议填写)。
    # 不再强制要求, 由 client 在空 key 时不发 Authorization 头。

    # 代码沙箱: HumanEval/MBPP 需执行模型生成的代码。
    # 所有登录用户均可提交 (代码在 docker/subprocess 沙箱里隔离执行, 安全);
    # 但沙箱未启用 (disabled) 时拒绝, 避免裸跑模型代码。
    scfg = load_server_config()
    if _CODE_BENCHES & set(body.benchmarks):
        if not scfg.code_exec_enabled:
            raise HTTPException(
                403,
                "代码沙箱未启用 (server.code_exec.sandbox=disabled)。"
                "请在 config.yaml 设为 docker/subprocess 后再提交 HumanEval/MBPP。",
            )

    # 构造内存模型配置 (含 key)
    model = _to_model_config(body.model_cfg.model_dump())
    judge = None
    if body.use_judge and body.judge_config and body.judge_config.api_key:
        judge = _to_model_config(body.judge_config.model_dump())

    run_params = {
        "limit": body.limit,
        "concurrency": body.concurrency,
        "streaming": body.streaming,
        "debug": body.debug,
        # 用户填的 max_tokens: 强制覆盖各评测集 parse_params() 的值。
        # None=用评测集自带值(如 IFEval 8192/代码 4096)。
        "max_tokens": body.model_cfg.max_tokens if body.model_cfg.max_tokens else None,
        "temperature": body.model_cfg.temperature if body.model_cfg.temperature is not None else None,
        # 单请求硬超时(秒): None=用平台默认 1200。跟随到 LLMClient._hard_timeout。
        "timeout": body.timeout,
    }
    name = body.name or f"{model.name} · {','.join(body.benchmarks[:3])}"
    task_id = db.create_task(
        user_id=user["id"],
        name=name,
        mode=body.mode,
        model_config=_strip_key(body.model_cfg.model_dump()),
        judge_config=_strip_key(body.judge_config.model_dump()) if body.use_judge and body.judge_config else None,
        benchmarks=body.benchmarks,
        run_params=run_params,
    )
    # 入队 (含 key 的 model/judge 进内存)
    taskman.enqueue(task_id, model, judge, threading.Event())
    t = db.get_task(task_id)
    return _task_out(t)


@router.get("", response_model=List[TaskOut])
def list_my_tasks(user: dict = Depends(auth.current_user), status: Optional[str] = None):
    return [_task_out(t) for t in db.list_tasks(user_id=user["id"], status=status)]


@router.get("/page", response_model=PageOut)
def list_my_tasks_page(
    user: dict = Depends(auth.current_user),
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
):
    """当前用户任务的分页列表 (供仪表盘分页)。page_size=0 表示全部。"""
    page = max(1, page)
    limit = 0 if page_size == 0 else max(1, page_size)
    offset = 0 if limit == 0 else (page - 1) * limit
    total = db.count_tasks(user_id=user["id"], status=status)
    items = [_task_out(t) for t in db.list_tasks(
        user_id=user["id"], status=status, limit=limit, offset=offset,
    )]
    return PageOut(items=items, total=total, page=page, page_size=page_size)


@router.get("/{task_id}", response_model=TaskOut)
def get_task(task_id: int, user: dict = Depends(auth.current_user)):
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t["user_id"] != user["id"] and auth.role_level(user.get("role")) < auth.role_level("admin"):
        raise HTTPException(403, "无权访问该任务")
    return _task_out(t)


@router.delete("/{task_id}", response_model=Message)
def delete_task(task_id: int, user: dict = Depends(auth.current_user)):
    """删除任务 (软删除, 进回收站, 30 天后彻底清除; 管理岗可恢复)。"""
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t["user_id"] != user["id"] and auth.role_level(user.get("role")) < auth.role_level("admin"):
        raise HTTPException(403, "无权删除该任务")
    # 运行中先取消
    if t["status"] in ("pending", "running"):
        taskman.cancel_task(task_id)
    db.delete_task(task_id)
    return Message(message="已移入回收站 (30天后彻底清除)")


@router.post("/{task_id}/clone", response_model=CloneOut)
def clone_task(task_id: int, user: dict = Depends(auth.current_user)):
    """克隆: 返回可编辑的提交表单数据 (无 key, 需重填)。"""
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t["user_id"] != user["id"] and auth.role_level(user.get("role")) < auth.role_level("admin"):
        raise HTTPException(403, "无权克隆该任务")
    rp = t["run_params"]
    return CloneOut(
        name=t["name"] + " (克隆)",
        mode=t["mode"],
        model_config=t["model_config"],   # 已无 key
        judge_config=t["judge_config"],
        benchmarks=t["benchmarks"],
        limit=rp.get("limit"),
        concurrency=rp.get("concurrency"),
        streaming=rp.get("streaming", False),
        debug=rp.get("debug", False),
        timeout=rp.get("timeout"),
    )


@router.post("/{task_id}/cancel", response_model=Message)
def cancel_task(task_id: int, user: dict = Depends(auth.current_user)):
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t["user_id"] != user["id"] and auth.role_level(user.get("role")) < auth.role_level("admin"):
        raise HTTPException(403, "无权操作该任务")
    if not taskman.cancel_task(task_id):
        raise HTTPException(400, "任务不在可取消状态")
    return Message(message="已请求取消")


@router.get("/{task_id}/logs", response_model=List[LogLine])
def get_logs(task_id: int, user: dict = Depends(auth.current_user), after_id: int = 0):
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t["user_id"] != user["id"] and auth.role_level(user.get("role")) < auth.role_level("admin"):
        raise HTTPException(403, "无权访问")
    return db.list_logs(task_id, after_id=after_id)


@router.get("/{task_id}/report")
def get_report(task_id: int, user: dict = Depends(auth.current_user)):
    """下载/查看任务的 HTML 报告。"""
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t["user_id"] != user["id"] and auth.role_level(user.get("role")) < auth.role_level("admin"):
        raise HTTPException(403, "无权访问")
    if not t["report_path"]:
        raise HTTPException(404, "报告尚未生成")
    scfg = load_server_config()
    path = os.path.join(scfg.results_dir, t["report_path"] + ".html")
    # 防路径穿越: 确保路径在 results_dir 内
    base = os.path.abspath(scfg.results_dir)
    real = os.path.abspath(path)
    if not real.startswith(base + os.sep) and real != base:
        raise HTTPException(400, "非法路径")
    if not os.path.exists(path):
        raise HTTPException(404, "报告文件不存在")
    return FileResponse(path, media_type="text/html")


@router.get("/{task_id}/events")
async def task_events(task_id: int, request: Request, user: dict = Depends(auth.current_user)):
    """SSE: 实时推送任务进度/日志/状态。

    连接时先补发 DB 里已有日志 + 当前状态快照, 再切到实时增量队列。
    """
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t["user_id"] != user["id"] and auth.role_level(user.get("role")) < auth.role_level("admin"):
        raise HTTPException(403, "无权访问")

    loop = asyncio.get_running_loop()
    q = taskman.subscribe(task_id, loop)

    async def event_stream():
        try:
            # 1. 补发已有日志 (DB)
            existing = db.list_logs(task_id, after_id=0)
            for lg in existing:
                yield f"data: {json.dumps({'type':'log','level':lg['level'],'message':lg['message'],'ts':lg['ts']}, ensure_ascii=False)}\n\n"
            # 2. 当前状态快照 (DB) + 进度快照 (供重新打开详情页时恢复进度)
            cur = db.get_task(task_id)
            if cur:
                yield f"data: {json.dumps({'type':'status','status':cur['status'],'summary':cur['summary'],'report_path':cur.get('report_path')}, ensure_ascii=False)}\n\n"
                # 补发各评测集进度 (从 DB progress 字段; 完成的任务也有, 显示100%)
                if cur.get("progress"):
                    yield f"data: {json.dumps({'type':'bench_progress','progress':cur['progress']}, ensure_ascii=False)}\n\n"
            # 3. 若任务已结束, 关闭
            if cur and cur["status"] in ("done", "failed", "cancelled"):
                return
            # 4. 先排空队列里在补发期间累积的事件 (避免重复/错序)
            while not q.empty():
                try:
                    ev = q.get_nowait()
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev.get("type") == "status" and ev.get("status") in ("done", "failed", "cancelled"):
                        return
                except asyncio.QueueEmpty:
                    break
            # 5. 实时增量
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    if ev.get("type") == "status" and ev.get("status") in ("done", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    yield f": keepalive\n\n"  # 心跳防代理超时
        finally:
            taskman.unsubscribe(task_id, q)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _check_task_access(task_id: int, user: dict):
    """取任务并校验访问权限 (本人或 admin)。返回 task dict 或抛 HTTPException。"""
    t = db.get_task(task_id)
    if not t:
        raise HTTPException(404, "任务不存在")
    if t["user_id"] != user["id"] and auth.role_level(user.get("role")) < auth.role_level("admin"):
        raise HTTPException(403, "无权访问")
    return t


@router.get("/{task_id}/samples")
def list_samples(
    task_id: int,
    benchmark: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
    user: dict = Depends(auth.current_user),
):
    """运行中任务的已完成单条样本摘要 (供任务详情实时查看吐字明细)。

    任务结束后 _SAMPLES 已清理, 返回 404 (改走完整 HTML 报告)。
    """
    _check_task_access(task_id, user)
    data = taskman.get_sample_summaries(task_id, benchmark=benchmark, offset=offset, limit=limit)
    if data is None:
        raise HTTPException(404, "任务未在运行中或样本明细不可用 (任务结束后请查看完整报告)")
    return data


@router.get("/{task_id}/samples/{sample_id}")
def get_sample(task_id: int, sample_id: str, user: dict = Depends(auth.current_user)):
    """运行中任务某条样本的完整明细 (含 prompt/response/reasoning/时序)。"""
    _check_task_access(task_id, user)
    data = taskman.get_sample_detail(task_id, sample_id)
    if data is None:
        raise HTTPException(404, "样本不存在或任务未在运行中")
    return data


@router.get("/{task_id}/requests")
def get_requests(task_id: int, benchmark: Optional[str] = None, user: dict = Depends(auth.current_user)):
    """任务详情"查看明细"页数据源: 报告同款请求明细 (flat_reqs + bench_infos + bench_options)。

    - 运行中任务: 返回已完成样本 (从内存 _SAMPLES), ended=False, 分数/乱码暂缺。
    - 已结束任务: 返回完整结果 (从保存的 JSON), ended=True, 含完整分数/乱码。
    与完整报告的"每条请求明细"页用同一套 rpt_table.js 渲染, 格式统一。

    benchmark 非 None 时 flat_reqs 只含该集 (bench_infos/bench_options 仍含全部集, 供切换)。
    """
    _check_task_access(task_id, user)
    data = taskman.get_request_details(task_id, benchmark=benchmark)
    if data is None:
        raise HTTPException(404, "无结果数据 (任务未运行过或结果文件不存在)")
    return data


@router.get("/{task_id}/requests/{sample_id}")
def get_request_sample(task_id: int, sample_id: str, user: dict = Depends(auth.current_user)):
    """单条请求的完整明细 (点 Hash 展开时按需拉取, 含完整 prompt/response/reasoning)。

    /requests 返回轻量摘要 (lite, 无大文本), 此端点补单条完整文本, 避免大任务一次传 60MB+。
    运行中从 _SAMPLES, 已结束从保存的 JSON 结果文件。
    """
    _check_task_access(task_id, user)
    data = taskman.get_request_sample_detail(task_id, sample_id)
    if data is None:
        raise HTTPException(404, "样本不存在或任务无结果数据")
    return data

