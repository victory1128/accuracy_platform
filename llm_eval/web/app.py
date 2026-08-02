"""Web 控制台 (FastAPI)

功能:
- GET  /          首页: 配置运行表单 + 已有报告列表
- POST /run       启动评测 (后台线程), 重定向到进度页
- GET  /progress  查看当前运行进度
- GET  /reports   已生成报告列表
- GET  /report/{name}  查看某份 HTML 报告
- GET  /benchmarks    评测集目录 (JSON)

运行:
  .venv/bin/python -m llm_eval.web
  或 uvicorn llm_eval.web.app:app --reload
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

from ..config import load_config, get_run_params, AppConfig
from ..models import ModelConfig
from ..benchmarks import list_benchmarks
from ..runner import Runner
from ..report import save_json, save_html

HERE = os.path.dirname(__file__)
TEMPLATES = Jinja2Templates(directory=os.path.join(HERE, "templates"))

app = FastAPI(title="大模型精度测试平台")

# 全局运行状态 (单机单进程, 足够)
_STATE: Dict[str, Any] = {
    "running": False,
    "log": [],
    "started_at": None,
    "finished_at": None,
    "current_model": None,
    "current_benchmarks": [],
    "total_benchmarks": 0,
    "done_benchmarks": [],
    "results": None,
    "error": None,
}
_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _LOCK:
        _STATE["log"].append(msg)
        # 限制日志长度
        if len(_STATE["log"]) > 500:
            _STATE["log"] = _STATE["log"][-300:]


def _get_config() -> AppConfig:
    return load_config()


def _results_dir(config: AppConfig) -> str:
    return config.output.get("dir", "results")


def _list_reports(config: AppConfig) -> List[Dict[str, Any]]:
    out = []
    rdir = _results_dir(config)
    if not os.path.isdir(rdir):
        return out
    for fn in sorted(os.listdir(rdir), reverse=True):
        if fn.endswith(".html"):
            stem = fn[:-5]
            stat = os.stat(os.path.join(rdir, fn))
            out.append({"name": stem, "filename": fn, "size": stat.st_size, "mtime": stat.st_mtime})
    return out


# 各厂商 OpenAI 兼容端点速查 (前端"预填"按钮用)
VENDOR_PRESETS = {
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "kimi":     {"base_url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-auto"},
    "qwen":     {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "glm":      {"base_url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4"},
    "openai":   {"base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    "custom":   {"base_url": "", "model": ""},
}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, err: Optional[str] = None):
    config = _get_config()
    metas = list_benchmarks()
    reports = _list_reports(config)
    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "models": [m.name for m in config.models],
            "benchmarks": metas,
            "reports": reports,
            "state": dict(_STATE),
            "judge_configured": config.judge is not None,
            "vendor_presets": VENDOR_PRESETS,
            "err": err,
        },
    )


def _build_model_config(
    name: str, base_url: str, api_key: str, model: str
) -> Optional[ModelConfig]:
    """从表单字段构造一个 ModelConfig; 缺关键字段返回 None"""
    if not (base_url and api_key and model):
        return None
    return ModelConfig(
        name=name or model,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )


@app.post("/run")
async def run(
    background_tasks: BackgroundTasks,
    benchmarks: List[str] = Form(...),
    limit: Optional[int] = Form(None),
    dry_run: bool = Form(False),
    streaming: bool = Form(False),
    model: str = Form(""),
    # —— 临时填写的被测模型 API (留空则用 config.yaml 里 model 名对应的配置) ——
    use_custom: bool = Form(False),
    custom_name: str = Form(""),
    custom_base_url: str = Form(""),
    custom_api_key: str = Form(""),
    custom_model: str = Form(""),
    # —— 临时填写的裁判模型 API (留空则用 config.yaml 的 judge) ——
    use_custom_judge: bool = Form(False),
    judge_name: str = Form(""),
    judge_base_url: str = Form(""),
    judge_api_key: str = Form(""),
    judge_model: str = Form(""),
):
    # 解析被测模型: 若勾了"直接填写"且字段齐全, 用临时配置; 否则回退 config
    custom_cfg = None
    if use_custom:
        custom_cfg = _build_model_config(custom_name, custom_base_url, custom_api_key, custom_model)
        if custom_cfg is None:
            return RedirectResponse(url="/?err=填了直接填写模式但 base_url/api_key/model 不全", status_code=303)
        display_model = custom_cfg.name
    else:
        display_model = model

    # 解析裁判模型(可选)
    custom_judge = None
    if use_custom_judge:
        custom_judge = _build_model_config(judge_name or "judge", judge_base_url, judge_api_key, judge_model)
        # 裁判字段不全就静默忽略(回退 config.judge)

    with _LOCK:
        if _STATE["running"]:
            return RedirectResponse(url="/progress?msg=已有任务运行中", status_code=303)
        _STATE.update(
            running=True, log=[], started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            finished_at=None, current_model=display_model, current_benchmarks=list(benchmarks),
            total_benchmarks=len(benchmarks), done_benchmarks=[], results=None, error=None,
        )
    background_tasks.add_task(
        _run_task, display_model, benchmarks, limit, dry_run, streaming, custom_cfg, custom_judge
    )
    return RedirectResponse(url="/progress", status_code=303)


def _run_task(
    model_name: str,
    benchmarks: List[str],
    limit: Optional[int],
    dry_run: bool,
    streaming: bool,
    custom_cfg: Optional[ModelConfig] = None,
    custom_judge: Optional[ModelConfig] = None,
):
    """后台执行评测

    custom_cfg: 临时填写的被测模型(优先于 config.yaml); None 则用 config 里的同名模型
    custom_judge: 临时填写的裁判模型(优先); None 则用 config.judge
    临时 key 仅存于本次运行的内存, 不写磁盘、不进报告。
    """
    try:
        config = _get_config()
        model_cfg = custom_cfg or next(
            (m for m in config.models if m.name == model_name), None
        )
        if not model_cfg:
            _STATE["error"] = f"未找到模型 {model_name}"
            _STATE["running"] = False
            return

        # 裁判模型: 临时优先, 其次 config, 最后 None
        judge_cfg = custom_judge or config.judge
        if judge_cfg is not None:
            _log(f"裁判模型: {judge_cfg.name}")
        # 安全: 日志/报告里绝不打印 api_key (ModelConfig 也无 __repr__ 暴露 key)

        if dry_run:
            _log(f"⚡ DRY-RUN 模式: 不调用真实API")
            from ..cli import _dry_run
            # dry_run 一次性返回全部, 这里逐个更新进度, 让进度条动起来
            all_results = _dry_run(model_cfg, benchmarks, limit)
            results = []
            by_name = {r.benchmark: r for r in all_results}
            for name in benchmarks:
                _log(f"▶ 开始: {name}")
                with _LOCK:
                    _STATE["current_benchmarks"] = [b for b in benchmarks if b not in _STATE["done_benchmarks"]]
                if name in by_name:
                    results.append(by_name[name])
                    _log(f"✓ 完成: {name}")
                else:
                    _log(f"✗ 跳过: {name} (无样本)")
                with _LOCK:
                    _STATE["done_benchmarks"].append(name)
        else:
            run_params = get_run_params(config)
            if limit is not None:
                run_params["limit"] = limit
            runner = Runner(
                model_config=model_cfg,
                judge_config=judge_cfg,
                verbose=False,
                streaming=streaming,
                **run_params,
            )
            # 逐个跑, 更新进度
            results = []
            from ..benchmarks import get as get_benchmark
            for name in benchmarks:
                _log(f"▶ 开始: {name}")
                with _LOCK:
                    _STATE["current_benchmarks"] = [b for b in benchmarks if b not in _STATE["done_benchmarks"]]
                try:
                    rr = runner.run_benchmark(name)
                    results.append(rr)
                    _log(f"✓ 完成: {name}")
                except Exception as e:
                    _log(f"✗ 失败: {name} - {e}")
                with _LOCK:
                    _STATE["done_benchmarks"].append(name)

        # 生成报告
        from datetime import datetime
        out_dir = _results_dir(config)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = model_name.replace("/", "_")
        json_path = os.path.join(out_dir, f"{safe_name}_{ts}.json")
        html_path = os.path.join(out_dir, f"{safe_name}_{ts}.html")
        save_json(results, json_path)
        save_html(results, html_path)
        _log(f"📄 报告: {os.path.basename(html_path)}")

        with _LOCK:
            _STATE["results"] = {
                "html": os.path.basename(html_path),
                "json": os.path.basename(json_path),
                "summary": [
                    {
                        "benchmark": r.benchmark_meta.display_name,
                        "stage": r.benchmark_meta.stage.value,
                        "score": _extract_score(r),
                        "num_samples": r.num_samples,
                    }
                    for r in results
                ],
            }
            _STATE["running"] = False
            _STATE["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        _log(f"✗ 运行异常: {e}")
        with _LOCK:
            _STATE["error"] = str(e)
            _STATE["running"] = False
            _STATE["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")


def _extract_score(r):
    agg = r.aggregate
    if "score_100" in agg and agg["score_100"] is not None:
        return f"{agg['score_100']}/100"
    if "mean_score" in agg and agg["mean_score"] is not None:
        return f"{agg['mean_score']}/10"
    for k in ("accuracy", "pass_at_1", "instruction_following_rate"):
        if k in agg and agg[k] is not None:
            return f"{agg[k]*100:.1f}%"
    return "-"


@app.get("/progress", response_class=HTMLResponse)
async def progress(request: Request):
    return TEMPLATES.TemplateResponse("progress.html", {"request": request, "state": dict(_STATE)})


@app.get("/progress.json")
async def progress_json():
    return JSONResponse(dict(_STATE))


@app.get("/reports", response_class=HTMLResponse)
async def reports(request: Request):
    config = _get_config()
    return TEMPLATES.TemplateResponse("reports.html", {"request": request, "reports": _list_reports(config)})


@app.get("/report/{filename}")
async def report(filename: str):
    config = _get_config()
    path = os.path.join(_results_dir(config), filename)
    if not os.path.exists(path) or not filename.endswith(".html"):
        return JSONResponse({"error": "报告不存在"}, status_code=404)
    return FileResponse(path, media_type="text/html")


@app.get("/benchmarks.json")
async def benchmarks_json():
    metas = list_benchmarks()
    return JSONResponse([
        {
            "name": m.name, "display_name": m.display_name,
            "stage": m.stage.value, "task_type": m.task_type.value,
            "description": m.description, "tags": m.tags,
            "needs_judge": m.needs_judge, "source": m.source,
        }
        for m in metas
    ])


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
