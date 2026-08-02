"""命令行入口

用法:
  python -m llm_eval.cli list                       # 列出所有评测集
  python -m llm_eval.cli run                        # 按配置跑全部
  python -m llm_eval.cli run -m deepseek-v4 -b mmlu gsm8k ifeval
  python -m llm_eval.cli run --limit 5              # 每个评测集只采样5条(调试)
  python -m llm_eval.cli run --dry-run              # 不真正调API, 用模拟响应跑通流程
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

from . import __version__
from .config import load_config, get_run_params, AppConfig
from .models import ModelConfig, RunResult, SampleResult
from .benchmarks import list_benchmarks, get as get_benchmark
from .runner import Runner, _make_request_hash, _dist_stats
from .report import save_json, save_html
from .analysis import analyze_gibberish, summarize_gibberish


def cmd_list(args: argparse.Namespace) -> None:
    metas = list_benchmarks()
    print(f"\n共 {len(metas)} 个评测集:\n")
    cur_stage = None
    for m in metas:
        if m.stage.value != cur_stage:
            cur_stage = m.stage.value
            label = "预训练阶段 (知识/推理/代码)" if cur_stage == "pretrain" else "后训练阶段 (指令遵循/对齐)"
            print(f"\n  ── {label} ──")
        judge = " [需裁判]" if m.needs_judge else ""
        tags = " ".join(f"#{t}" for t in m.tags)
        print(f"    {m.name:<14} {m.display_name:<18} {m.task_type.value:<6} {judge:<8} {tags}")
        if args.verbose and m.description:
            print(f"                   {m.description}")
    print()


def cmd_run(args: argparse.Namespace) -> None:
    config = load_config(args.config)

    # 选模型
    model_cfg: Optional[ModelConfig] = None
    if args.model:
        for m in config.models:
            if m.name == args.model:
                model_cfg = m
                break
        if model_cfg is None:
            print(f"错误: 配置中未找到模型 '{args.model}'。可选: {[m.name for m in config.models]}")
            sys.exit(1)
    elif config.models:
        model_cfg = config.models[0]
    else:
        print("错误: 配置中未定义任何模型。")
        sys.exit(1)

    # 选评测集
    benchmarks: List[str] = args.benchmarks or config.benchmarks
    if not benchmarks:
        print("错误: 未指定评测集。用 -b 指定, 或在 config.yaml 里配置 benchmarks。")
        sys.exit(1)

    # 运行参数
    run_params = get_run_params(config)
    if args.limit is not None:
        run_params["limit"] = args.limit
    if args.concurrency is not None:
        run_params["concurrency"] = args.concurrency

    runner = Runner(
        model_config=model_cfg,
        judge_config=config.judge,
        verbose=True,
        **run_params,
    )

    # 接管 dry-run
    if args.dry_run:
        results = _dry_run(model_cfg, benchmarks, run_params.get("limit"))
    else:
        results = runner.run_all(benchmarks)

    if not results:
        print("未产生任何结果。")
        return

    # 写报告
    out_dir = config.output.get("dir", "results")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{model_cfg.name}_{ts}.json")
    html_path = os.path.join(out_dir, f"{model_cfg.name}_{ts}.html")
    save_json(results, json_path)
    save_html(results, html_path)
    print(f"\n报告已生成:\n  JSON: {json_path}\n  HTML: {html_path}")

    # 控制台简报
    _print_summary(results)


def _sample_is_chinese(sample, meta) -> bool:
    """判断样本题目是否中文 (dry-run 用)"""
    if meta.name in ("ceval", "cmmlu"):
        return True
    text = sample.question or ""
    han = sum(1 for c in text if "一" <= c <= "鿿")
    return han > max(1, len(text) * 0.1)


def _dry_run(model_cfg: ModelConfig, benchmarks: List[str], limit: Optional[int]) -> List[RunResult]:
    """不调真实API, 用模拟响应跑通评分+乱码分析+报告全流程。用于无Key时验证平台。"""
    import uuid
    from .models import Stage
    print("⚡ DRY-RUN 模式: 不调用真实API, 使用模拟响应验证流程。\n")
    results = []
    for name in benchmarks:
        bench = get_benchmark(name)
        meta = bench.meta()
        samples = bench.load_samples(limit=limit, seed=42)
        if not samples:
            print(f"  ⚠ {meta.display_name}: 无样本, 跳过")
            continue
        print(f"▶ {meta.display_name} ({len(samples)} 条, 模拟)")
        sim_results = []
        for s in samples:
            # 判断题目语言, 构造匹配语言的模拟响应
            zh = _sample_is_chinese(s, meta)
            # 构造一个模拟响应: 一半正确一半"乱码"
            if meta.task_type.value == "mcq":
                gold = s.gold or "A"
                resp = (f"让我想想...\n答案是 {gold}。" if zh else f"Let me think...\nThe answer is {gold}.")
            elif meta.task_type.value in ("gen",):
                gold = s.gold or "42"
                resp = (f"解答过程如下: 逐步推导可得答案。\n答案是 {gold}" if zh else f"Solving step by step.\nThe answer is {gold}")
            elif meta.task_type.value == "code":
                resp = "```python\n" + (s.question or "pass") + "\n    return 0\n```"
            elif meta.task_type.value == "rule":
                resp = ("这是一个模拟的回答, 包含一些内容用于测试指令遵循。" if zh else "This is a simulated response to test instruction following.")
            else:
                resp = ("模拟回答内容。" if zh else "Simulated response content.")

            # 每3条注入一个乱码样本, 验证乱码分析
            if len(sim_results) % 3 == 2:
                resp = ("啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊 " * 20) if zh else ("aaaa aaaa aaaa " * 20)
            prompt = bench.build_prompt(s)
            sr = bench.evaluate(s, resp, None)
            sr.analysis = analyze_gibberish(resp, expected_lang="zh" if zh else "en")
            # 填充可追溯/速度字段 (模拟数据, 让报告/浏览器有内容可展示)
            sr.benchmark = meta.name
            sr.prompt = prompt
            sr.question = s.question
            sr.gold = s.gold
            sr.gen_params = bench.parse_params()
            sr.request_hash = _make_request_hash(meta.name, s.sample_id, prompt, resp, model_cfg.name)
            # 模拟延迟 200~2000ms, token 数按响应长度估
            import random as _r
            sr.latency_ms = float(_r.Random(hash(s.sample_id) & 0xffff).randint(200, 2000))
            sr.prompt_chars = len(prompt)
            sr.response_chars = len(resp)
            # 模拟思维链 (gen/math 类模拟有思考过程)
            if meta.task_type.value in ("gen", "mcq"):
                sr.reasoning_content = "这是模拟的思维链过程: 先分析题目, 再推导得出结论。(dry-run 模拟)"
            sr.prompt_tokens = max(10, len(prompt) // 4)
            sr.completion_tokens = max(5, len(resp) // 4)
            sr.tokens_per_sec = round(sr.completion_tokens / (sr.latency_ms / 1000), 2)
            sr.ttft_ms = sr.latency_ms
            sr.streaming = False  # dry-run 默认非流式 (TTFT 为近似)
            sr.usage = {"latency_ms": sr.latency_ms, "prompt_tokens": sr.prompt_tokens,
                        "completion_tokens": sr.completion_tokens, "finish_reason": "stop"}
            sim_results.append(sr)
        agg = bench.aggregate(sim_results)
        # 模拟速度/长度分布统计 (复用 runner 的 _dist_stats)
        agg["gibberish"] = summarize_gibberish([r.analysis for r in sim_results if r.analysis])
        agg["latency"] = _dist_stats([r.latency_ms for r in sim_results if r.latency_ms], "ms")
        agg["ttft_ms"] = {**_dist_stats([r.ttft_ms for r in sim_results if r.ttft_ms], "ms"), "approx": True}
        agg["tokens_per_sec"] = _dist_stats([r.tokens_per_sec for r in sim_results if r.tokens_per_sec], "tok/s")
        agg["token_usage"] = {
            "prompt": _dist_stats([r.prompt_tokens for r in sim_results if r.prompt_tokens], "tok"),
            "completion": _dist_stats([r.completion_tokens for r in sim_results if r.completion_tokens], "tok"),
            "reasoning": None,
        }
        agg["length_chars"] = {
            "prompt": _dist_stats([r.prompt_chars for r in sim_results if r.prompt_chars is not None], "char"),
            "response": _dist_stats([r.response_chars for r in sim_results if r.response_chars is not None], "char"),
        }
        rr = RunResult(
            run_id=str(uuid.uuid4())[:8],
            model_name=model_cfg.name + " (dry-run)",
            benchmark=name,
            benchmark_meta=meta,
            num_samples=len(samples),
            results=sim_results,
            aggregate=agg,
            started_at=datetime.now().isoformat(),
            finished_at=datetime.now().isoformat(),
        )
        results.append(rr)
    return results


def _print_summary(results: List[RunResult]) -> None:
    print("\n" + "=" * 64)
    print(f"{'评测集':<18}{'阶段':<10}{'分数':<14}{'乱码率':<10}{'样本':<8}")
    print("-" * 64)
    for r in results:
        meta = r.benchmark_meta
        agg = r.aggregate
        # 分数
        if "score_100" in agg and agg["score_100"] is not None:
            score = f"{agg['score_100']}/100"
        elif "mean_score" in agg and agg["mean_score"] is not None:
            score = f"{agg['mean_score']}/10"
        elif "accuracy" in agg and agg["accuracy"] is not None:
            score = f"{agg['accuracy']*100:.1f}%"
        elif "pass_at_1" in agg and agg["pass_at_1"] is not None:
            score = f"{agg['pass_at_1']*100:.1f}%"
        elif "instruction_following_rate" in agg:
            score = f"{agg['instruction_following_rate']*100:.1f}%"
        else:
            score = "-"
        g = agg.get("gibberish", {})
        gib = f"{g.get('suspicious_rate',0)*100:.1f}%" if g else "-"
        print(f"{meta.display_name:<18}{meta.stage.value:<10}{score:<14}{gib:<10}{r.num_samples:<8}")
    print("=" * 64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llm-eval",
        description="大模型精度测试平台 — 评测预训练/后训练能力 + 乱码分析",
    )
    p.add_argument("--version", action="version", version=f"llm-eval {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出所有评测集")
    p_list.add_argument("-v", "--verbose", action="store_true", help="显示描述")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="运行评测")
    p_run.add_argument("-c", "--config", help="配置文件路径 (默认 config.yaml)")
    p_run.add_argument("-m", "--model", help="模型名 (config.models 里的 name)")
    p_run.add_argument("-b", "--benchmarks", nargs="+", help="评测集名列表")
    p_run.add_argument("--limit", type=int, help="每个评测集最多采样条数 (调试用)")
    p_run.add_argument("--concurrency", type=int, help="并发数")
    p_run.add_argument("--dry-run", action="store_true", help="不调真实API, 用模拟响应跑通流程")
    p_run.set_defaults(func=cmd_run)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
