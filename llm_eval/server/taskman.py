"""任务管理器 (task manager)

替换原 web/app.py 的全局 _STATE: 多用户、持久化、多任务并发。

模型:
- 任务提交后入队 (pending), worker 线程从队列取, 状态 -> running
- 运行完写报告 -> done/failed, 结果存 tasks.summary
- 实时进度通过 SSE 事件总线推给订阅者 (每个任务一个 asyncio.Queue)

被测模型 API Key: 只存在 _INFLIGHT 字典 (task_id -> ModelConfig 含 key),
运行结束即删, 绝不进 DB。克隆任务时 DB 里没有 key, 需用户重填。

代码沙箱: HumanEval/MBPP 执行模型生成代码。默认 disabled (仅 admin 可启用)。
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..models import ModelConfig, RunResult, BenchmarkMeta, Stage, TaskType, SampleResult
from ..report import save_html, save_json, build_request_data
from ..runner import Runner
from . import db

# 运行中的任务: task_id -> {"model": ModelConfig(含key), "judge": ModelConfig|None,
#                          "cancel": threading.Event, "events": asyncio.Queue}
_INFLIGHT: Dict[int, dict] = {}
_INFLIGHT_LOCK = threading.Lock()

# 全局任务队列 (FIFO); worker 线程循环取
_QUEUE: List[int] = []
_QUEUE_LOCK = threading.Lock()
_QUEUE_COND = threading.Condition(_QUEUE_LOCK)

_WORKER_STARTED = False
_WORKER_THREAD: Optional[threading.Thread] = None

# 默认并发跑多少个任务 (不同用户的任务可并行; 单任务内的样本并发由 Runner 管)
MAX_CONCURRENT_TASKS = 2
_sem = threading.Semaphore(MAX_CONCURRENT_TASKS)

# 事件总线: task_id -> list[(asyncio.Queue, asyncio.AbstractEventLoop)]
# 每个订阅者带自己的事件循环, 跨线程 emit 时在其循环上 call_soon_threadsafe
_SUBSCRIBERS: Dict[int, list] = {}
_SUB_LOCK = threading.Lock()


# ----------------------------- 启动 worker -----------------------------
def start_worker():
    """启动后台 worker 线程 (进程级, 只启动一次)。"""
    global _WORKER_STARTED, _WORKER_THREAD
    if _WORKER_STARTED:
        return
    _WORKER_STARTED = True
    _WORKER_THREAD = threading.Thread(target=_worker_loop, name="taskman-worker", daemon=True)
    _WORKER_THREAD.start()


def _worker_loop():
    while True:
        with _QUEUE_COND:
            while not _QUEUE:
                _QUEUE_COND.wait()
            task_id = _QUEUE.pop(0)
        _sem.acquire()
        # 每个任务起一个线程跑, 释放信号量
        t = threading.Thread(target=_run_one, args=(task_id,), name=f"task-{task_id}", daemon=True)
        t.start()


def enqueue(task_id: int, model: ModelConfig, judge: Optional[ModelConfig], cancel: threading.Event):
    """把任务加入运行队列。model/judge 含 api_key (内存), 不进 DB。"""
    with _INFLIGHT_LOCK:
        _INFLIGHT[task_id] = {"model": model, "judge": judge, "cancel": cancel, "events": None}
    with _QUEUE_COND:
        _QUEUE.append(task_id)
        _QUEUE_COND.notify()


def cancel_task(task_id: int) -> bool:
    """请求取消任务 (设置 cancel 事件)。运行中的任务在样本边界及时退出。

    三种情况:
    1. 任务在 _INFLIGHT (正在跑): set cancel 事件, worker 会及时退出
    2. 任务在队列里 (pending): 直接标记 cancelled, 移出队列
    3. DB 是 running 但不在 _INFLIGHT (僵尸任务: 服务重启后残留): 直接标记 cancelled
    """
    with _INFLIGHT_LOCK:
        info = _INFLIGHT.get(task_id)
    if info:
        info["cancel"].set()
        return True
    # 还在队列里没跑: 直接标记 cancelled
    t = db.get_task(task_id)
    if t and t["status"] in ("pending",):
        db.update_task_status(task_id, "cancelled", error="用户取消", finished=True)
        with _QUEUE_LOCK:
            if task_id in _QUEUE:
                _QUEUE.remove(task_id)
        return True
    # DB 是 running 但不在 _INFLIGHT: 僵尸任务 (服务重启/worker崩溃后残留), 直接标记 cancelled
    if t and t["status"] in ("running",):
        db.update_task_status(task_id, "cancelled", error="用户取消 (任务已不在运行)", finished=True)
        return True
    return False


# ----------------------------- 事件总线 (SSE) -----------------------------
def subscribe(task_id: int, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
    """订阅任务事件 (SSE 用)。返回一个 asyncio.Queue, 在 loop 上创建。"""
    q: asyncio.Queue = asyncio.Queue(maxsize=256)
    with _SUB_LOCK:
        _SUBSCRIBERS.setdefault(task_id, []).append((q, loop))
    return q


def unsubscribe(task_id: int, q: asyncio.Queue):
    with _SUB_LOCK:
        subs = _SUBSCRIBERS.get(task_id, [])
        _SUBSCRIBERS[task_id] = [(sq, sl) for (sq, sl) in subs if sq is not q]


def _emit(task_id: int, event: dict, loop: Optional[asyncio.AbstractEventLoop] = None):
    """把事件推给该任务的所有订阅者。从 worker 线程调用 (跨线程安全)。

    loop 参数 (worker 自己的循环) 现已不使用: 每个订阅者记录了自己的循环,
    用 call_soon_threadsafe 投递到订阅者循环上。保留参数仅为向后兼容。
    """
    with _SUB_LOCK:
        subs = list(_SUBSCRIBERS.get(task_id, []))
    for q, sub_loop in subs:
        try:
            sub_loop.call_soon_threadsafe(_safe_put, q, event)
        except Exception:
            pass


def _safe_put(q: asyncio.Queue, event: dict):
    """在订阅者循环上安全 put (队列满则丢弃最旧, 防 SSE 阻塞)。"""
    try:
        q.put_nowait(event)
    except asyncio.QueueFull:
        try:
            q.get_nowait()  # 丢最旧
            q.put_nowait(event)
        except Exception:
            pass


# ----------------------------- 运行单个任务 -----------------------------
def _extract_score(agg: dict) -> str:
    """从 aggregate 抽取分数字符串。

    统一输出纯数值 (0-100 量纲, 无百分号), 便于界面清爽显示与跨评测集求平均:
    - score_100 (裁判评分题, 已是 0-100): 直接取
    - mean_score (裁判评分题, 0-10 量纲): ×10 归一到 0-100
    - accuracy/pass_at_1/instruction_following_rate (0-1 小数): ×100
    """
    if "score_100" in agg and agg["score_100"] is not None:
        return f"{agg['score_100']:.1f}"
    if "mean_score" in agg and agg["mean_score"] is not None:
        return f"{agg['mean_score']*10:.1f}"
    for k in ("accuracy", "pass_at_1", "instruction_following_rate"):
        if k in agg and agg[k] is not None:
            return f"{agg[k]*100:.1f}"
    return "-"


def _log(task_id: int, msg: str, level: str = "info"):
    """写一条任务日志 (DB) + 推 SSE 事件 + 控制台。"""
    db.add_log(task_id, level, msg)
    _emit(task_id, {"type": "log", "level": level, "message": msg, "ts": datetime.now().isoformat(timespec="seconds")})


# 各评测集进度: _BENCH_PROG[task_id] = {bench: {"done","total","pct","status"}}
# status: pending(未开始)/running(进行中)/done(完成)/cancelled(取消)
# 节流: 每个评测集的进度事件按 ~1% 粒度推, 避免大评测集刷屏 SSE。
_BENCH_PROG: Dict[int, dict] = {}
_PROG_LOCK = threading.Lock()

# 运行中任务的已完成单条样本明细: _SAMPLES[task_id][bench] = [sample_result_dict, ...]
# 供任务详情页实时查看已跑完的吐字明细 (不必等整集/整任务跑完)。
# 仅运行中任务存在; 任务结束 (finally) 清理, 之后改走完整 HTML 报告。
_SAMPLES: Dict[int, Dict[str, list]] = {}
_SAMPLES_LOCK = threading.Lock()

# 已结束任务结果 JSON 解析缓存: 避免每次 /requests 都重读 100MB+ JSON (json.load 1s+)。
# _RR_CACHE[task_id] = {"path": str, "mtime": float, "rrs": List[RunResult]}
# 文件 mtime 变化 (任务重跑/覆盖) 时失效重读。仅缓存已结束任务 (结果文件不再变)。
_RR_CACHE: Dict[int, dict] = {}
_RR_CACHE_LOCK = threading.Lock()


def _load_run_results_cached(task_id: int) -> Optional[List["RunResult"]]:
    """读已结束任务的结果 JSON 并重建 RunResult, 带 mtime 缓存。

    第一次读 100MB+ JSON 约 1s (json.load), 之后命中缓存毫秒级。
    文件 mtime 变化时失效。返回 None: 任务无 report_path 或文件不存在。
    """
    t = db.get_task(task_id)
    if not t or not t.get("report_path"):
        return None
    from .server_config import load_server_config as _lsc
    _scfg = _lsc()
    json_path = os.path.join(_scfg.results_dir, t["report_path"] + ".json")
    if not os.path.exists(json_path):
        return None
    mtime = os.path.getmtime(json_path)
    with _RR_CACHE_LOCK:
        cached = _RR_CACHE.get(task_id)
        if cached and cached["path"] == json_path and cached["mtime"] == mtime:
            return cached["rrs"]
    # 缓存未命中: 读 + 解析 (锁外, 避免长时间持锁阻塞其它任务)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    rrs = _run_results_from_json(data.get("results") or [])
    with _RR_CACHE_LOCK:
        _RR_CACHE[task_id] = {"path": json_path, "mtime": mtime, "rrs": rrs}
    return rrs


def _init_bench_progress(task_id: int, benchmarks: List[str]):
    """任务开始时, 初始化各评测集进度 (全部 pending)。"""
    with _PROG_LOCK:
        _BENCH_PROG[task_id] = {b: {"done": 0, "total": 0, "pct": 0, "status": "pending"} for b in benchmarks}
    _emit_bench_progress(task_id)


def _set_bench_running(task_id: int, bench: str, total: int):
    """某评测集开始跑: 标记 running, 设总数。"""
    with _PROG_LOCK:
        bp = _BENCH_PROG.setdefault(task_id, {})
        bp[bench] = {"done": 0, "total": total, "pct": 0, "status": "running"}
    _emit_bench_progress(task_id)


def _on_sample_progress(task_id: int, done: int, total: int, sid: str, bench_name: str):
    """样本进度回调 (节流): 更新该评测集进度, 推 bench_progress 事件。

    节流: 大评测集仅当百分比变化时推; 小评测集每条推。
    """
    if total <= 0:
        return
    pct = round(done / total * 100)
    with _PROG_LOCK:
        bp = _BENCH_PROG.setdefault(task_id, {})
        st = bp.get(bench_name) or {"done": 0, "total": total, "pct": 0, "status": "running"}
        last_pct = st.get("pct", -1)
        # 大评测集仅当百分比变化时推 (小评测集每条推)
        if total > 200 and pct == last_pct and done < total:
            return
        bp[bench_name] = {"done": done, "total": total, "pct": pct, "status": "running"}
    _emit_bench_progress(task_id)


def _sample_summary(sr_dict: dict) -> dict:
    """从单条 SampleResult 字典抽轻量摘要字段 (供列表展示, 不含大块 prompt/response)。"""
    return {
        "sample_id": sr_dict.get("sample_id"),
        "idx": sr_dict.get("_idx"),
        "correct": sr_dict.get("correct"),
        "score": sr_dict.get("score"),
        "error": (sr_dict.get("error") or "")[:120] or None,
        "ttft_ms": sr_dict.get("ttft_ms"),
        "tpot_ms": sr_dict.get("tpot_ms"),
        "tokens_per_sec": sr_dict.get("tokens_per_sec"),
        "completion_tokens": sr_dict.get("completion_tokens"),
        "reasoning_tokens": sr_dict.get("reasoning_tokens"),
        "response_chars": sr_dict.get("response_chars"),
        "streaming": sr_dict.get("streaming"),
        "real_stream": sr_dict.get("real_stream"),
        "extracted": (sr_dict.get("extracted") or "")[:200] if sr_dict.get("extracted") else None,
    }


def _on_sample_done(task_id: int, bench: str, done: int, total: int, idx: int, sr):
    """单条样本完成回调: 累积到内存 _SAMPLES, 节流推 sample_result SSE 事件。

    - 累积完整 SampleResult 字典 (供 /samples 端点按需拉取吐字明细)。
    - SSE 只推轻量摘要 (避免万条集每条推大对象); 大集按 ~1% 节流, 小集每条推。
    """
    try:
        sr_dict = asdict(sr)
    except Exception:  # noqa: BLE001
        return
    sr_dict["_idx"] = idx
    with _SAMPLES_LOCK:
        bucket = _SAMPLES.setdefault(task_id, {})
        bucket.setdefault(bench, []).append(sr_dict)

    # 节流推送: 大集仅当百分比变化时推; 小集每条推 (与 _on_sample_progress 一致)。
    pct = round(done / total * 100) if total > 0 else 100
    push = True
    if total > 200 and done < total:
        with _PROG_LOCK:
            bp = _BENCH_PROG.get(task_id, {})
            st = bp.get(bench) or {}
            if pct == st.get("pct", -1):
                push = False
    if push:
        _emit(task_id, {
            "type": "sample_result",
            "benchmark": bench,
            "done": done,
            "total": total,
            "idx": idx,
            "sample": _sample_summary(sr_dict),
        })


def _set_bench_done(task_id: int, bench: str, aggregate: Optional[dict] = None,
                    num_samples: Optional[int] = None):
    """某评测集完成: 标记 done, 进度100%; 可附带 aggregate (评分等)。"""
    with _PROG_LOCK:
        bp = _BENCH_PROG.setdefault(task_id, {})
        st = bp.get(bench) or {"done": 0, "total": 0, "pct": 0, "status": "running"}
        st.update({"done": st.get("total", 0), "pct": 100, "status": "done"})
        if aggregate is not None:
            score = _extract_score(aggregate)
            st["score"] = score if score != "-" else None
            # 精简 aggregate: 分数 + 关键时序/乱码指标, 供前端展示评分卡
            st["aggregate"] = _slim_aggregate(aggregate)
        if num_samples is not None:
            st["num_samples"] = num_samples
        bp[bench] = st
    _emit_bench_progress(task_id)


def get_sample_summaries(task_id: int, benchmark: Optional[str] = None,
                         offset: int = 0, limit: int = 100) -> Optional[Dict[str, Any]]:
    """读取运行中任务已累积的单条样本摘要 (供任务详情实时查看)。

    返回 None 表示任务不在运行中 (_SAMPLES 已清理或不存在)。
    返回 {benchmark, total, items} —— items 为轻量摘要 (不含完整 prompt/response)。
    """
    with _SAMPLES_LOCK:
        buckets = _SAMPLES.get(task_id)
        if buckets is None:
            return None
        if benchmark is not None:
            bench_names = [benchmark] if benchmark in buckets else []
        else:
            bench_names = list(buckets.keys())
        out_items = []
        chosen = None
        for b in bench_names:
            chosen = b
            rows = buckets.get(b, [])
            total = len(rows)
            sliced = rows[offset: offset + limit] if limit and limit > 0 else rows[offset:]
            out_items.extend([_sample_summary(r) for r in sliced])
        if benchmark is None:
            # 未指定集: 返回各集各自的计数
            return {
                "benchmarks": {b: len(rows) for b, rows in buckets.items()},
                "total": sum(len(rows) for rows in buckets.values()),
                "items": out_items,
                "offset": offset,
                "limit": limit,
            }
        return {"benchmark": chosen, "total": total, "items": out_items,
                "offset": offset, "limit": limit}


def get_sample_detail(task_id: int, sample_id: str) -> Optional[Dict[str, Any]]:
    """读取运行中任务某条样本的完整明细 (含 prompt/response/reasoning/时序)。

    返回 None 表示任务不在运行中或样本不存在。
    """
    with _SAMPLES_LOCK:
        buckets = _SAMPLES.get(task_id)
        if buckets is None:
            return None
        for rows in buckets.values():
            for r in rows:
                if str(r.get("sample_id")) == str(sample_id):
                    return r
        return None


def get_request_details(task_id: int, benchmark: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """任务详情"查看明细"页的数据源: 返回报告同款 (flat_reqs, bench_infos, bench_options)。

    - 运行中任务: 从内存 _SAMPLES 取已完成样本的完整 SampleResult 字典, 重建 RunResult 后
      调 report.build_request_data。aggregate 此时为空 (分数尚未算), bench_infos 无分数。
    - 已结束任务: _SAMPLES 已清理, 从保存的 JSON 结果文件读 RunResult.to_dict() 重建,
      再调 build_request_data (含完整分数/乱码)。
    - 返回 None: 任务不存在或无结果文件。

    始终用 lite=True: 表格行只需摘要字段 (大文本不返回), 点 Hash 展开时由
    get_request_sample_detail 按需拉单条完整 prompt/response/reasoning。避免大任务 (万条级)
    一次传 60MB+ JSON 导致前端长时间空白。

    benchmark 非 None 时只返回该集的 flat_reqs (bench_infos/bench_options 仍含全部集)。
    """
    # 1) 运行中: 从 _SAMPLES 取
    with _SAMPLES_LOCK:
        buckets = _SAMPLES.get(task_id)
        if buckets is not None:
            rows_by_bench = {b: list(rs) for b, rs in buckets.items()}
        else:
            rows_by_bench = None
    if rows_by_bench is not None:
        return _request_details_from_dicts(rows_by_bench, benchmark)

    # 2) 已结束: 从 JSON 文件取 (带 mtime 缓存, 避免每次重读 100MB+ JSON)
    rrs = _load_run_results_cached(task_id)
    if not rrs:
        return None
    flat_reqs, bench_infos, bench_options = build_request_data(rrs, lite=True)
    if benchmark is not None:
        flat_reqs = [r for r in flat_reqs if r["benchmark"] == benchmark]
    return {"flat_reqs": flat_reqs, "bench_infos": bench_infos, "bench_options": bench_options, "ended": True}


def get_request_sample_detail(task_id: int, sample_id: str) -> Optional[Dict[str, Any]]:
    """单条请求的完整明细 (点 Hash 展开时按需拉取, 含完整 prompt/response/reasoning)。

    - 运行中任务: 从 _SAMPLES 找该条 SampleResult, 用 build_request_data(lite=False) 取完整字段。
    - 已结束任务: 从保存的 JSON 结果文件找该条, 同样取完整字段。
    - 返回 None: 任务/样本不存在。
    """
    sid = str(sample_id)
    # 1) 运行中: 从 _SAMPLES 取
    with _SAMPLES_LOCK:
        buckets = _SAMPLES.get(task_id)
        if buckets is not None:
            for b, rows in buckets.items():
                for r in rows:
                    if str(r.get("sample_id")) == sid:
                        meta = _benchmark_meta_cached(b) or BenchmarkMeta(
                            name=b, display_name=b, stage=Stage("posttrain"), task_type=TaskType("generic"))
                        d2 = {k: v for k, v in r.items() if k in SampleResult.__dataclass_fields__}
                        try:
                            samples = [SampleResult(**d2)]
                        except TypeError:
                            samples = []
                        rr = RunResult(run_id=str(b), model_name="", benchmark=b, benchmark_meta=meta,
                                       num_samples=1, results=samples, aggregate={})
                        flat, _, _ = build_request_data([rr], lite=False)
                        return flat[0] if flat else None
            return None  # _SAMPLES 有该任务但无此样本

    # 2) 已结束: 从 JSON 文件取 (带 mtime 缓存)
    rrs = _load_run_results_cached(task_id)
    if not rrs:
        return None
    for rr in rrs:
        for s in rr.results:
            if str(s.sample_id) == sid:
                flat, _, _ = build_request_data([RunResult(
                    run_id=rr.run_id, model_name="", benchmark=rr.benchmark,
                    benchmark_meta=rr.benchmark_meta, num_samples=1, results=[s], aggregate=rr.aggregate,
                )], lite=False)
                return flat[0] if flat else None
    return None


def _benchmark_meta_cached(name: str) -> Optional[BenchmarkMeta]:
    """从评测集注册表取某集的 BenchmarkMeta (用于运行中任务的明细, 避免重新加载样本)。"""
    try:
        from ..benchmarks import get as get_benchmark
        meta = get_benchmark(name).meta()
        return meta
    except Exception:  # noqa: BLE001
        return None


def _request_details_from_dicts(rows_by_bench: Dict[str, list], benchmark: Optional[str]) -> Dict[str, Any]:
    """运行中任务: 把 _SAMPLES 里的 SampleResult 字典 (asdict) 重建为 RunResult, 调 build_request_data。

    aggregate 为空 (运行中尚未汇总), 故 bench_infos 无分数/乱码 (前端容错显示 '-')。
    """
    rrs = []
    for b, rows in rows_by_bench.items():
        if benchmark is not None and b != benchmark:
            continue
        meta = _benchmark_meta_cached(b)
        if meta is None:
            # 退化: 无元信息时用 generic
            meta = BenchmarkMeta(name=b, display_name=b, stage=Stage("posttrain"), task_type=TaskType("generic"))
        samples = []
        for d in rows:
            d2 = {k: v for k, v in d.items() if k in SampleResult.__dataclass_fields__}
            try:
                samples.append(SampleResult(**d2))
            except TypeError:
                continue
        rrs.append(RunResult(
            run_id=str(b), model_name="", benchmark=b, benchmark_meta=meta,
            num_samples=len(samples), results=samples, aggregate={},
        ))
    if not rrs:
        return {"flat_reqs": [], "bench_infos": {}, "bench_options": [], "ended": False}
    # 运行中也要给出全部集的 bench_infos/bench_options (供下拉切换), 即便 benchmark 指定了单集
    all_rrs = rrs if benchmark is None else _request_details_from_dicts_all(rows_by_bench)
    flat_all, bench_infos, bench_options = build_request_data(all_rrs, lite=True)
    return {"flat_reqs": flat_all, "bench_infos": bench_infos, "bench_options": bench_options, "ended": False}


def _request_details_from_dicts_all(rows_by_bench: Dict[str, list]) -> List[RunResult]:
    """运行中任务: 重建全部集的 RunResult (供下拉/切换集, 不限单集)。"""
    rrs = []
    for b, rows in rows_by_bench.items():
        meta = _benchmark_meta_cached(b) or BenchmarkMeta(
            name=b, display_name=b, stage=Stage("posttrain"), task_type=TaskType("generic"))
        samples = []
        for d in rows:
            d2 = {k: v for k, v in d.items() if k in SampleResult.__dataclass_fields__}
            try:
                samples.append(SampleResult(**d2))
            except TypeError:
                continue
        rrs.append(RunResult(run_id=str(b), model_name="", benchmark=b, benchmark_meta=meta,
                             num_samples=len(samples), results=samples, aggregate={}))
    return rrs


def _run_results_from_json(results_data: list) -> List[RunResult]:
    """从 save_json 写出的 results 列表 (RunResult.to_dict) 重建 RunResult 对象。"""
    rrs = []
    for d in results_data:
        meta = d.get("benchmark_meta") or {}
        try:
            bm = BenchmarkMeta(
                name=d.get("benchmark", ""), display_name=meta.get("display_name", d.get("benchmark", "")),
                stage=Stage(meta.get("stage", "posttrain")), task_type=TaskType(meta.get("task_type", "generic")),
                description=meta.get("description", ""), tags=meta.get("tags", []),
                num_fewshot=meta.get("num_fewshot", 0), needs_judge=meta.get("needs_judge", False),
                source=meta.get("source", ""),
            )
        except (ValueError, TypeError):
            continue
        samples = []
        for s in d.get("results") or []:
            d2 = {k: v for k, v in s.items() if k in SampleResult.__dataclass_fields__}
            try:
                samples.append(SampleResult(**d2))
            except TypeError:
                continue
        rrs.append(RunResult(
            run_id=d.get("run_id", ""), model_name=d.get("model_name", ""), benchmark=d.get("benchmark", ""),
            benchmark_meta=bm, num_samples=d.get("num_samples", len(samples)), results=samples,
            aggregate=d.get("aggregate") or {}, started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""), fake_stream=d.get("fake_stream", False),
        ))
    return rrs


def _slim_aggregate(agg: dict) -> dict:
    """从 aggregate 抽取前端评分卡需要的关键指标 (避免推整个聚合体)。"""
    out = {}
    for k in ("accuracy", "pass_at_1", "score_100", "mean_score", "instruction_following_rate"):
        if k in agg and agg[k] is not None:
            out[k] = agg[k]
    for k in ("ttft_ms", "tpot_ms", "tokens_per_sec"):
        if k in agg and agg[k] is not None:
            out[k] = agg[k]
    g = agg.get("gibberish") or {}
    if g:
        out["gibberish_rate"] = g.get("suspicious_rate")
        out["gibberish_grade"] = g.get("overall_grade")
    if "error_count" in agg:
        out["error_count"] = agg["error_count"]
    return out


def _emit_bench_progress(task_id: int):
    """推送各评测集进度 (供前端展示每个数据集的进度条) + 持久化到 DB。"""
    with _PROG_LOCK:
        prog = {b: dict(v) for b, v in _BENCH_PROG.get(task_id, {}).items()}
    if not prog:
        return
    _emit(task_id, {"type": "bench_progress", "progress": prog})
    try:
        db.update_task_progress(task_id, prog)
    except Exception:  # noqa: BLE001
        pass


def _on_sample_error(task_id: int, sid: str, err: str, bench_name: str):
    """样本出错回调: 写详细错误日志 (前端日志区可见, 便于排查报错)。

    对相同错误做计数去重: 同一评测集内, 重复的错误只记首条 + 累计次数,
    避免认证失败等场景每条样本都刷一条相同日志。
    """
    err_short = (err or "").strip()
    if len(err_short) > 300:
        err_short = err_short[:300] + "..."
    # 去重 key: 评测集 + 错误前80字符 (常见认证错误完整相同)
    key = (bench_name, err_short[:80])
    with _ERR_LOCK:
        cnt = _ERR_COUNT.get(task_id, {}).get(key, 0) + 1
        _ERR_COUNT.setdefault(task_id, {})[key] = cnt
        if cnt > 1:
            # 重复错误: 不逐条记, 仅在每 10 条时汇总一次
            if cnt % 10 != 0:
                return
            _log(task_id, f"  ⚠ [{bench_name}] 相同错误已累计 {cnt} 次 (如 {sid}): {err_short}", "warn")
            return
    _log(task_id, f"  ⚠ [{bench_name}] {sid} 出错: {err_short}", "warn")


# 错误去重计数: _ERR_COUNT[task_id] = {(bench, err_prefix): count}
_ERR_COUNT: Dict[int, dict] = {}
_ERR_LOCK = threading.Lock()


def _run_one(task_id: int):
    """worker 线程: 取出任务, 跑 Runner/_dry_run, 写报告, 更新状态。"""
    loop = asyncio.new_event_loop()  # 供 SSE emit 用
    try:
        with _INFLIGHT_LOCK:
            info = _INFLIGHT.get(task_id)
        if not info:
            return
        model: ModelConfig = info["model"]
        judge: Optional[ModelConfig] = info["judge"]
        cancel: threading.Event = info["cancel"]

        task = db.get_task(task_id)
        if not task:
            return
        benchmarks: List[str] = task["benchmarks"]
        rp: dict = task["run_params"]
        mode = task["mode"]
        streaming = rp.get("streaming", False)
        limit = rp.get("limit")
        concurrency = rp.get("concurrency")
        debug = rp.get("debug", False)

        db.update_task_status(task_id, "running", started=True, error=None)
        _emit(task_id, {"type": "status", "status": "running"}, loop)

        _log(task_id, f"▶ 开始任务: {task['name']} (模式={mode})")
        _log(task_id, f"  模型: {model.name} ({model.model}) @ {model.base_url}")
        _log(task_id, f"  评测集: {', '.join(benchmarks)}")

        if cancel.is_set():
            db.update_task_status(task_id, "cancelled", error="用户取消", finished=True)
            _emit(task_id, {"type": "status", "status": "cancelled"}, loop)
            return

        # 真实模式用 Runner; dry_run/quick 用 _dry_run 或小样本
        from ..cli import _dry_run
        results = []
        summary = []   # 增量维护已完成集的评分摘要 (每集跑完即落 DB, 刷新页面可见)
        started = datetime.now()
        try:
            if mode == "dry_run":
                _log(task_id, "⚡ DRY-RUN: 不调用真实 API, 使用模拟响应")
                _init_bench_progress(task_id, benchmarks)
                results = _dry_run(model, benchmarks, limit)
                # dry_run 一次性完成, 标记所有评测集 done
                for r in results:
                    _set_bench_done(task_id, r.benchmark)
            else:
                # real / quick: 用 Runner 真跑 (quick 只是 limit 小)
                run_params = {"concurrency": concurrency or 4, "max_retries": 3, "timeout": 1200, "seed": 42}
                if limit is not None:
                    run_params["limit"] = limit
                # 用户级生成参数覆盖 (max_tokens/temperature): 强制覆盖各评测集自带值
                override = {}
                if rp.get("max_tokens"):
                    override["max_tokens"] = rp["max_tokens"]
                if rp.get("temperature") is not None:
                    override["temperature"] = rp["temperature"]
                runner = Runner(
                    model_config=model,
                    judge_config=judge,
                    verbose=False,
                    streaming=streaming,
                    override_params=override or None,
                    progress_callback=lambda done, total_n, sid, bname: _on_sample_progress(task_id, done, total_n, sid, bname),
                    error_callback=lambda sid, err, bname: _on_sample_error(task_id, sid, err, bname),
                    sample_result_callback=lambda bench, done, total_n, idx, sr: _on_sample_done(task_id, bench, done, total_n, idx, sr),
                    cancel_event=cancel,
                    **run_params,
                )
                # 逐个评测集跑, 边跑边推进度
                from ..benchmarks import get as get_benchmark
                _init_bench_progress(task_id, benchmarks)
                total = len(benchmarks)
                # 增量落盘: 每跑完一个评测集就保存一次, 避免中途卡死/崩溃全丢。
                # 复用最终报告的目录与命名, 跑完最后一个集时即最终报告。
                from .server_config import load_server_config as _lsc
                _scfg = _lsc()
                _out_dir = os.path.join(_scfg.results_dir, f"user_{task['user_id']}")
                os.makedirs(_out_dir, exist_ok=True)
                _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                _safe = model.name.replace("/", "_")
                _stem = f"{_safe}_{_ts}"
                _json_path = os.path.join(_out_dir, f"{_stem}.json")
                _html_path = os.path.join(_out_dir, f"{_stem}.html")
                _rel_stem = f"user_{task['user_id']}/{_stem}"
                for i, name in enumerate(benchmarks):
                    if cancel.is_set():
                        _log(task_id, "⏹ 收到取消, 中止后续评测集", "warn")
                        # 未跑的评测集标记 cancelled
                        for nb in benchmarks[i:]:
                            with _PROG_LOCK:
                                bp = _BENCH_PROG.setdefault(task_id, {})
                                bp[nb] = {"done": 0, "total": 0, "pct": 0, "status": "cancelled"}
                        _emit_bench_progress(task_id)
                        break
                    _log(task_id, f"▶ [{i+1}/{total}] {name}")
                    _emit(task_id, {"type": "progress", "done": i, "total": total, "current": name}, loop)
                    # 先算该评测集样本数, 标记 running (让前端立即看到进度条)
                    try:
                        nb = len(get_benchmark(name).load_samples(limit=limit, seed=42))
                    except Exception:  # noqa: BLE001
                        nb = 0
                    _set_bench_running(task_id, name, nb)
                    _bench_agg = None
                    _bench_ns = None
                    rr = None
                    try:
                        rr = runner.run_benchmark(name)
                        results.append(rr)
                        agg = rr.aggregate
                        _bench_agg = agg
                        _bench_ns = rr.num_samples
                        _log(task_id, f"  ✓ {name}: {_extract_score(agg)} ({rr.num_samples} 条)")
                    except Exception as e:  # noqa: BLE001
                        _log(task_id, f"  ✗ {name} 失败: {e}", "error")
                    # 取消检查: run_benchmark 返回 (取消时它会快速返回部分结果) 后, 若已取消,
                    # 不再做该集的评分/落盘, 直接跳出循环走取消收尾 (标记 cancelled)。
                    if cancel.is_set():
                        _log(task_id, "⏹ 收到取消, 中止当前评测集处理", "warn")
                        with _PROG_LOCK:
                            bp = _BENCH_PROG.setdefault(task_id, {})
                            bp[name] = {"done": _bench_ns or 0, "total": 0, "pct": 0, "status": "cancelled"}
                        _emit_bench_progress(task_id)
                        break
                    # 一集跑完即标记 done 并带评分 (前端立即看到该集分数卡, 不必等整任务)
                    _set_bench_done(task_id, name, aggregate=_bench_agg, num_samples=_bench_ns)
                    # 增量写 summary: 已完成集的评分落 DB, 刷新页面也能看到
                    if _bench_agg is not None and rr is not None:
                        try:
                            summary.append({
                                "benchmark": rr.benchmark_meta.display_name,
                                "stage": rr.benchmark_meta.stage.value,
                                "score": _extract_score(rr.aggregate),
                                "num_samples": rr.num_samples,
                            })
                            db.update_task_status(task_id, "running", summary=summary)
                        except Exception:  # noqa: BLE001
                            pass
                    # 增量落盘: 每跑完一个集就写一次 json/html, 中途卡死也有已完成集的结果
                    if results:
                        try:
                            save_json(results, _json_path)
                            save_html(results, _html_path)
                        except Exception as e:  # noqa: BLE001
                            _log(task_id, f"  ⚠ 增量保存失败: {e}", "warn")
                _emit(task_id, {"type": "progress", "done": len(results), "total": total, "current": None}, loop)
        except Exception as e:  # noqa: BLE001
            _log(task_id, f"✗ 任务异常: {e}", "error")
            # 取消过程中抛的异常 (如中断后的保存失败) 应标记 cancelled, 非 failed
            estatus = "cancelled" if cancel.is_set() else "failed"
            db.update_task_status(task_id, estatus, error=None if cancel.is_set() else str(e), finished=True)
            _emit(task_id, {"type": "status", "status": estatus, "error": None if cancel.is_set() else str(e)}, loop)
            return

        if not results:
            _log(task_id, "未产生任何结果", "warn")
            db.update_task_status(task_id, "failed", error="未产生结果", finished=True)
            _emit(task_id, {"type": "status", "status": "failed", "error": "未产生结果"}, loop)
            return

        # 检测"全样本出错"情况: 所有请求都带 error (如 API Key 认证失败 401)。
        # 此时虽没抛异常, 但实际没有任何有效结果, 应标记 failed 并醒目提示,
        # 而不是显示"完成"误导用户。报告照常生成 (供查看每条错误明细)。
        all_samples = [s for r in results for s in r.results]
        err_samples = [s for s in all_samples if s.error]
        all_failed = bool(all_samples) and len(err_samples) == len(all_samples)
        if all_failed:
            # 取第一个错误作为任务错误摘要 (常见如认证错误, 所有样本同因)
            first_err = (err_samples[0].error or "")[:200]
            _log(task_id, f"✗ 全部 {len(err_samples)} 条请求出错: {first_err}", "error")
            # 继续生成报告, 但任务标记为 failed
        else:
            first_err = None

        # 写报告: 复用增量落盘已建好的路径 (循环里每个集跑完都写过一次, 这里是最终版)
        json_path = _json_path
        html_path = _html_path
        rel_stem = _rel_stem
        save_json(results, json_path)
        save_html(results, html_path)
        _log(task_id, f"📄 报告已生成: {_stem}.html")

        summary = [
            {
                "benchmark": r.benchmark_meta.display_name,
                "stage": r.benchmark_meta.stage.value,
                "score": _extract_score(r.aggregate),
                "num_samples": r.num_samples,
            }
            for r in results
        ]
        num_samples = sum(r.num_samples for r in results)
        # 用户取消: 即使有部分结果, 也标记 cancelled (而非 done/failed)。
        # 取消时 in-flight 请求被中断, 部分样本带 error=已取消, 不能据此判 failed。
        if cancel.is_set():
            final_status = "cancelled"
            first_err = None
            _log(task_id, "⏹ 任务已取消")
        else:
            # 全样本出错 -> 标记 failed (报告仍生成, 供查看明细); 否则 done
            final_status = "failed" if all_failed else "done"
        db.update_task_status(
            task_id, final_status,
            summary=summary,
            report_path=rel_stem,
            num_samples=num_samples,
            error=first_err,
            finished=True,
        )
        _emit(task_id, {"type": "status", "status": final_status, "summary": summary, "report_path": rel_stem, "error": first_err}, loop)
        if final_status == "cancelled":
            _log(task_id, "⏹ 任务已取消")
        else:
            _log(task_id, "✓ 任务完成" if not all_failed else "✗ 任务完成但全部请求出错 (已标记失败)")
    finally:
        _sem.release()
        with _INFLIGHT_LOCK:
            _INFLIGHT.pop(task_id, None)
        with _PROG_LOCK:
            _BENCH_PROG.pop(task_id, None)
        with _SAMPLES_LOCK:
            # 任务结束: 清理内存中的单条样本明细 (之后改走完整 HTML 报告)
            _SAMPLES.pop(task_id, None)
        with _ERR_LOCK:
            _ERR_COUNT.pop(task_id, None)
        loop.close()
