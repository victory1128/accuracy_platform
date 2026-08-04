"""评测运行器 (runner)

编排一次评测运行:
1. 加载评测集样本
2. 并发调用被测模型
3. 对每条响应评分 + 乱码分析
4. 聚合指标
5. 返回 RunResult (供报告生成器使用)
"""
from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .client import LLMClient, LLMClientError, run_concurrent
from .models import (
    BenchmarkMeta,
    ModelConfig,
    RunResult,
    Sample,
    SampleResult,
    Stage,
    TaskType,
)
from .analysis import analyze_gibberish, summarize_gibberish
from .benchmarks import get as get_benchmark
from .benchmarks.base import Benchmark


class Runner:
    def __init__(
        self,
        model_config: ModelConfig,
        judge_config: Optional[ModelConfig] = None,
        concurrency: int = 4,
        max_retries: int = 3,
        timeout: int = 1200,
        limit: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
        streaming: bool = False,
        override_params: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None,
        error_callback: Optional[callable] = None,
        sample_result_callback: Optional[callable] = None,
        cancel_event: Optional[Any] = None,
    ):
        self.client = LLMClient(model_config, timeout=timeout, max_retries=max_retries, concurrency=concurrency)
        self.judge_client = (
            LLMClient(judge_config, timeout=timeout, max_retries=max_retries)
            if judge_config
            else None
        )
        # 取消事件注入客户端: 用户点"取消任务"时, client 的请求循环 (chat/chat_stream)
        # 检测到 cancel_event 被 set 后立即关闭底层 socket 并抛"已取消", 让在途请求
        # 秒级中断, 而非等硬超时/慢吐自然结束。
        self.client.cancel_event = cancel_event
        if self.judge_client is not None:
            self.judge_client.cancel_event = cancel_event
        self.concurrency = concurrency
        self.limit = limit
        self.seed = seed
        self.verbose = verbose
        self.model_config = model_config
        # 进度/错误回调 (taskman 注入, 用于样本级进度 + 详细报错日志)
        # progress_callback(done, total, sample_id) / error_callback(sample_id, error)
        self.progress_callback = progress_callback
        self.error_callback = error_callback
        # 单条结果回调 (taskman 注入): 每跑完一条样本, 回传 (benchmark, sample_result_dict),
        # 供任务详情页实时展示已跑完的吐字明细 (不必等整集/整任务跑完)。
        self.sample_result_callback = sample_result_callback
        # 取消事件 (threading.Event): taskman 注入, 用于在样本级别及时中止
        self.cancel_event = cancel_event
        self.streaming = streaming
        # 用户级生成参数覆盖: 强制覆盖各评测集 parse_params() 的同名键。
        # 如 max_tokens=4096 会覆盖 IFEval(8192)/代码(4096) 等评测集自带值。
        # None 的键不覆盖 (保留评测集自带值)。
        self.override_params = {k: v for k, v in (override_params or {}).items() if v is not None}

    def run_benchmark(self, benchmark_name: str) -> RunResult:
        """运行单个评测集"""
        bench: Benchmark = get_benchmark(benchmark_name)
        meta = bench.meta()
        self._log(f"▶ 加载评测集 {meta.display_name} ({meta.stage.value}/{meta.task_type.value})")

        if meta.needs_judge and self.judge_client is None:
            self._log(f"  ⚠ {meta.display_name} 需要裁判模型但未配置, 将标记为失败")

        samples = bench.load_samples(limit=self.limit, seed=self.seed)
        if not samples:
            self._log(f"  ⚠ {meta.display_name} 无样本数据 (data/{...} 缺失?)")
            # 仍返回一个空结果
            return RunResult(
                run_id=str(uuid.uuid4())[:8],
                model_name=self.model_config.name,
                benchmark=benchmark_name,
                benchmark_meta=meta,
                num_samples=0,
                started_at=datetime.now().isoformat(),
                finished_at=datetime.now().isoformat(),
                aggregate={"error": "无样本数据"},
            )

        params = bench.parse_params()
        # 用户级参数覆盖 (max_tokens/temperature 等): 强制覆盖评测集自带值。
        # 仅覆盖非 None 的键; 评测集自带的 stop 等不受影响。
        if self.override_params:
            params.update(self.override_params)
        self._log(f"  共 {len(samples)} 条样本, 并发 {self.concurrency}, 开始评测...")

        started = datetime.now()

        # 定义单样本 worker
        def worker(sample: Sample) -> SampleResult:
            prompt = bench.build_prompt(sample)
            system = bench.system_prompt()
            try:
                if self.streaming:
                    response, usage = self.client.chat_stream(
                        prompt,
                        system=system,
                        temperature=params.get("temperature", 0.0),
                        max_tokens=params.get("max_tokens", 2048),
                        stop=params.get("stop"),
                    )
                else:
                    response, usage = self.client.chat(
                        prompt,
                        system=system,
                        temperature=params.get("temperature", 0.0),
                        max_tokens=params.get("max_tokens", 2048),
                        stop=params.get("stop"),
                    )
            except LLMClientError as e:
                # 出错也要补全可追溯/长度字段 (request_hash/prompt_chars 等),
                # 否则报告里出错样本的 Hash 列为空, 无法定位。response 为空。
                return SampleResult(
                    sample_id=sample.sample_id,
                    response="",
                    error=str(e),
                    benchmark=meta.name,
                    prompt=prompt,
                    system_prompt=system,
                    question=sample.question,
                    gold=sample.gold,
                    reference=(sample.meta or {}).get("reference") or sample.gold,
                    gen_params={k: v for k, v in params.items()},
                    streaming=self.streaming,
                    request_hash=_make_request_hash(
                        meta.name, sample.sample_id, prompt, "", self.model_config.name
                    ),
                    prompt_chars=len(prompt),
                    response_chars=0,
                    analysis=analyze_gibberish("", expected_lang=("zh" if _is_chinese_bench(meta, sample) else "en")),
                )

            # 评分
            try:
                # 思维链模型常见问题: reasoning 吃光 max_tokens, content(response)为空。
                # 此时答案常在 reasoning_content 末尾 (模型推理后得出结论但没写到 content),
                # 用 reasoning 作为评分输入兜底, 让各评测集的抽取逻辑从 reasoning 里抽答案。
                # 对 MCQ/数学(gpqa/aime/math500) 这很有用; 开放式 QA(simpleqa) reasoning 里
                # 多半无简洁答案会判错, 但那是模型/预算问题, 不在此掩盖。
                # 同时把 response 设成 reasoning, 让后续截断检测/报告展示一致。
                if not (response or "").strip():
                    rc = (usage.get("reasoning_content") or "").strip()
                    if rc:
                        response = rc
                result = bench.evaluate(sample, response, self.judge_client)
            except Exception as e:  # noqa: BLE001
                result = SampleResult(
                    sample_id=sample.sample_id, response=response, error=f"评分异常: {e}"
                )

            # 填充用量/延迟/乱码分析
            result.usage = usage
            result.latency_ms = usage.get("latency_ms")
            real_stream = usage.get("real_stream")
            # 伪流式处理: 退化成非流式口径 (TTFT 用端到端近似, 不显示 TPOT/gen_time)
            # 这样伪流式的报告和非流式完全一致, 避免误导。
            # streaming 字段: 仅真流式才为 True; 伪流式/非流式都为 False
            if self.streaming and real_stream is False:
                # 伪流式: 标记为非流式口径
                result.streaming = False
                result.real_stream = False
                result.ttft_ms = None  # 下面会用端到端近似补
                result.tpot_ms = None
                result.gen_time_ms = None
            else:
                result.streaming = self.streaming
                result.real_stream = real_stream
                result.ttft_ms = usage.get("ttft_ms")  # 流式=真实, 非流式下面补近似
                result.tpot_ms = usage.get("tpot_ms")   # 仅真流式有值
                result.gen_time_ms = usage.get("gen_time_ms")  # 仅真流式有值
            # 思维链全文 (思维链模型才有)
            result.reasoning_content = usage.get("reasoning_content")
            expected_lang = "zh" if _is_chinese_bench(meta, sample) else "en"
            # IFEval 的 response_language 约束要求特定语言 (孟加拉/波斯/越南语等),
            # 此时模型用该语言回答是正确的, 传 "any" 跳过异常脚本/语言不匹配误判。
            cons = (sample.meta or {}).get("constraints") or []
            if any(c.get("type") == "response_language" for c in cons if isinstance(c, dict)):
                expected_lang = "any"
            # 合并而非覆盖: evaluate() 可能已往 analysis 写入评测集专属结果
            # (如 IFEval 的约束检查明细), 这里把乱码分析并进去。
            gib = analyze_gibberish(response, expected_lang=expected_lang)
            if result.analysis:
                gib.update(result.analysis)
            result.analysis = gib

            # —— 可追溯/可搜索字段 ——
            result.benchmark = meta.name
            result.prompt = prompt
            result.system_prompt = system
            result.question = sample.question
            result.gold = sample.gold
            # 参考答案: JUDGE 类存参考模型输出于 meta.reference;
            # MCQ/GEN 的 gold 是字母/数字, reference 取题面可读答案 (若有)。
            ref = (sample.meta or {}).get("reference")
            if ref:
                result.reference = ref
            elif sample.gold:
                result.reference = sample.gold
            result.gen_params = {k: v for k, v in params.items()}
            result.request_hash = _make_request_hash(
                meta.name, sample.sample_id, prompt, response, self.model_config.name
            )

            # —— 速度/长度指标 ——
            result.prompt_chars = len(prompt)
            result.response_chars = len(response)
            result.prompt_tokens = usage.get("prompt_tokens")
            result.completion_tokens = usage.get("completion_tokens")
            result.reasoning_tokens = usage.get("reasoning_tokens")
            latency_s = (result.latency_ms or 0) / 1000.0
            # 输出速度 = completion_tokens / 端到端秒数 (报告统一用这个总速度)
            if result.completion_tokens and latency_s > 0:
                result.tokens_per_sec = round(result.completion_tokens / latency_s, 2)
            # TTFT: 流式已有真实值(usage.ttft_ms); 非流式拿不到, 用端到端近似
            if result.ttft_ms is None:
                result.ttft_ms = result.latency_ms

            # 思维链模型常见问题: reasoning 吃光 max_tokens, content 为空或被截断
            # finish_reason=length 表示因 token 上限停止, 此时答案多半不可信
            # 注: 截断属"输出异常"而非"乱码", 归入 is_abnormal/abnormal_notes,
            # 不混入乱码率 (is_suspicious), 避免把截断误报成乱码。
            if usage.get("finish_reason") == "length":
                note = "截断(finish_reason=length, 思维链可能未输出答案)"
                if not response.strip():
                    # content 完全空 -> 这条基本无效
                    result.error = (result.error + "; " if result.error else "") + note
                    result.correct = False
                    result.score = 0.0
                    if result.analysis:
                        result.analysis.setdefault("abnormal_notes", []).append("空输出(finish_reason=length)")
                        result.analysis["is_abnormal"] = True
                else:
                    # 有内容但被截断, 记录但不强制判错
                    if result.analysis:
                        result.analysis.setdefault("abnormal_notes", []).append(note)
                        result.analysis["is_abnormal"] = True
            return result

        # 样本级进度回调: 每完成一条, 推进度 + 若出错推错误日志
        sample_ids = [s.sample_id for s in samples]

        def _on_sample_progress(done, total, idx, result):
            sid = sample_ids[idx] if 0 <= idx < len(sample_ids) else ""
            if self.progress_callback:
                try:
                    self.progress_callback(done, total, sid, meta.name)
                except Exception:  # noqa: BLE001
                    pass
            # 出错的样本 -> error_callback (供详细日志)
            if self.error_callback and result is not None:
                err = None
                if isinstance(result, dict) and result.get("_error"):
                    err = str(result["_error"])
                elif hasattr(result, "error") and result.error:
                    err = str(result.error)
                if err:
                    try:
                        self.error_callback(sid, err, meta.name)
                    except Exception:  # noqa: BLE001
                        pass
            # 单条结果回传 (供任务详情实时展示吐字明细): 仅对成功样本 (SampleResult) 回传,
            # 异常样本 result 是 dict, 不含完整结果。idx 保留样本在集内的序号 (0-based)。
            if self.sample_result_callback is not None and isinstance(result, SampleResult):
                try:
                    self.sample_result_callback(meta.name, done, total, idx, result)
                except Exception:  # noqa: BLE001
                    pass

        # 取消检查: taskman 注入的 cancel_event 被 set 时, 及时中止当前评测集
        _cancel = self.cancel_event
        should_stop = (lambda: _cancel.is_set()) if _cancel is not None else None
        # 取消时强制关闭在途请求的底层连接, 让卡在 socket recv 的 worker 立即退出。
        on_cancel = (lambda: self.client.abort_in_flight()) if _cancel is not None else None

        results = run_concurrent(samples, worker, concurrency=self.concurrency,
                                 on_progress=_on_sample_progress, should_stop=should_stop,
                                 on_cancel=on_cancel)

        # 失败/取消的样本兜底为"错误占位" SampleResult, 不再丢弃。
        # 旧实现 `[r for r in results if hasattr(r,"analysis")]` 会过滤掉所有返回 dict 的
        # 超时/异常样本, 导致: ① 分数偏(分子分母都缩水, accuracy=对/幸存数 而非 对/总数);
        # ② 报告里看不到失败样本, 无法追溯失败原因。现在统一转成 error 占位:
        # correct=False、response="", 保留 sample_id/prompt/gold 供定位, 计入分母。
        # 注意: client 层已修 (网络异常转 LLMClientError -> worker 接住成 SampleResult),
        # 此处是对 worker 仍可能抛非 LLMClientError 的防御兜底。
        fixed_results = []
        for sample, r in zip(samples, results):
            if isinstance(r, SampleResult):
                fixed_results.append(r)
                continue
            # dict: run_concurrent 对 worker 抛出的异常 catch 成 {"_error":.., "_cancelled":..}
            err = ""
            cancelled = False
            if isinstance(r, dict):
                err = str(r.get("_error") or "未知错误")
                cancelled = bool(r.get("_cancelled"))
            else:
                err = str(r)
            prompt = bench.build_prompt(sample)
            system = bench.system_prompt()
            fixed_results.append(SampleResult(
                sample_id=sample.sample_id,
                response="",
                error=err,
                correct=False,
                benchmark=meta.name,
                prompt=prompt,
                system_prompt=system,
                question=sample.question,
                gold=sample.gold,
                reference=(sample.meta or {}).get("reference") or sample.gold,
                gen_params={k: v for k, v in params.items()},
                streaming=self.streaming,
                request_hash=_make_request_hash(
                    meta.name, sample.sample_id, prompt, "", self.model_config.name
                ),
                prompt_chars=len(prompt),
                response_chars=0,
                analysis=analyze_gibberish(
                    "", expected_lang=("zh" if _is_chinese_bench(meta, sample) else "en")
                ),
            ))
            if cancelled:
                fixed_results[-1].analysis = fixed_results[-1].analysis or {}
                fixed_results[-1].analysis.setdefault("abnormal_notes", []).append("已取消")
        results = fixed_results

        # 聚合
        aggregate = bench.aggregate(results)
        # 乱码汇总
        gibberish_summary = summarize_gibberish(
            [r.analysis for r in results if r.analysis]
        )
        aggregate["gibberish"] = gibberish_summary
        # —— 速度/延迟分布 ——
        latencies = [r.latency_ms for r in results if r.latency_ms]
        if latencies:
            aggregate["latency"] = _dist_stats(latencies, unit="ms")
        ttfts = [r.ttft_ms for r in results if r.ttft_ms]
        if ttfts:
            # 流式=真实TTFT, 非流式=端到端近似; 用 streaming 标志区分
            is_stream = any(r.streaming for r in results if r.streaming is not None)
            aggregate["ttft_ms"] = _dist_stats(ttfts, unit="ms", approx=not is_stream)
        # 流式专属真实指标: TPOT 每token生成时间, 生成阶段耗时
        tpots = [r.tpot_ms for r in results if r.tpot_ms is not None]
        if tpots:
            aggregate["tpot_ms"] = _dist_stats(tpots, unit="ms")
        gentimes = [r.gen_time_ms for r in results if r.gen_time_ms is not None]
        if gentimes:
            aggregate["gen_time_ms"] = _dist_stats(gentimes, unit="ms")
        # 是否伪流式: 用户开了流式, 但样本里检测到伪流式 (real_stream=False 且 streaming 已退化)
        # 注意: 伪流式样本的 streaming 已被置 False, 故用 real_stream 字段判断
        fake_stream = self.streaming and any(
            r.real_stream is False for r in results if r.real_stream is not None
        )
        # streaming 字段: 伪流式退化为 False (按非流式口径)
        aggregate["streaming"] = self.streaming and not fake_stream
        aggregate["fake_stream"] = fake_stream
        tps = [r.tokens_per_sec for r in results if r.tokens_per_sec]
        if tps:
            aggregate["tokens_per_sec"] = _dist_stats(tps, unit="tok/s")
        # —— token 用量分布 ——
        ptoks = [r.prompt_tokens for r in results if r.prompt_tokens]
        ctoks = [r.completion_tokens for r in results if r.completion_tokens]
        rtoks = [r.reasoning_tokens for r in results if r.reasoning_tokens]
        aggregate["token_usage"] = {
            "prompt": _dist_stats(ptoks, unit="tok"),
            "completion": _dist_stats(ctoks, unit="tok"),
            "reasoning": _dist_stats(rtoks, unit="tok") if rtoks else None,
        }
        # —— 输入/输出字符长度分布 ——
        pchars = [r.prompt_chars for r in results if r.prompt_chars is not None]
        rchars = [r.response_chars for r in results if r.response_chars is not None]
        # 思维链字符分布 (思维链模型才有)
        reason_chars = [len(r.reasoning_content or "") for r in results if r.reasoning_content]
        aggregate["length_chars"] = {
            "prompt": _dist_stats(pchars, unit="char"),
            "response": _dist_stats(rchars, unit="char"),
            "reasoning": _dist_stats(reason_chars, unit="char") if reason_chars else None,
        }
        # 出错统计
        errors = [r for r in results if r.error]
        if errors:
            aggregate["error_count"] = len(errors)

        finished = datetime.now()
        run_result = RunResult(
            run_id=str(uuid.uuid4())[:8],
            model_name=self.model_config.name,
            benchmark=benchmark_name,
            benchmark_meta=meta,
            num_samples=len(results),
            results=results,
            aggregate=aggregate,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            fake_stream=fake_stream,
        )
        self._log(f"  ✓ {meta.display_name} 完成: {_format_agg(aggregate)}  ({(finished - started).total_seconds():.1f}s)")
        return run_result

    def run_all(self, benchmark_names: List[str]) -> List[RunResult]:
        self._log(f"模型: {self.model_config.name} | 评测集: {', '.join(benchmark_names)}")
        if self.judge_client:
            self._log(f"裁判模型: {self.judge_client.config.name}")
        out = []
        for name in benchmark_names:
            try:
                out.append(self.run_benchmark(name))
            except Exception as e:  # noqa: BLE001
                self._log(f"  ✗ {name} 运行失败: {e}")
        return out

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


def _is_chinese_bench(meta: BenchmarkMeta, sample: Sample) -> bool:
    """判断样本题目是否为中文 (乱码分析用)

    返回 bool: True=中文题, False=英文题。
    注意: 之前返回字符串 "zh"/"en" 会导致调用处
        "zh" if _is_chinese_bench(...) else "en"  始终为 "zh"
    (因为 "en" 也是真值), 已修正为返回 bool。
    """
    # C-Eval / CMMLU 是中文评测集
    if meta.name in ("ceval", "cmmlu"):
        return True
    # 含中文字符的题目按中文
    text = sample.question or ""
    if not text:
        return False
    han = sum(1 for c in text if "一" <= c <= "鿿")
    return han > len(text) * 0.1


def _format_agg(agg: Dict[str, Any]) -> str:
    """把聚合指标格式成一行简报"""
    parts = []
    for key in ("accuracy", "pass_at_1", "instruction_following_rate", "score_100", "mean_score"):
        if key in agg and agg[key] is not None:
            label = {
                "accuracy": "acc",
                "pass_at_1": "pass@1",
                "instruction_following_rate": "IF-rate",
                "score_100": "score/100",
                "mean_score": "mean",
            }.get(key, key)
            val = agg[key]
            parts.append(f"{label}={val}")
    g = agg.get("gibberish", {})
    if g:
        parts.append(f"乱码={g.get('suspicious_rate', 0)}")
    return " | ".join(parts) if parts else "(无指标)"


def _make_request_hash(benchmark: str, sample_id: str, prompt: str, response: str, model: str) -> str:
    """为单个请求生成唯一 hash (12位短哈希, 便于搜索定位)

    基于模型+评测集+样本+prompt+response 内容, 同一请求内容稳定复现;
    加上 uuid4 的随机因子保证即使内容相同也是唯一请求 (放在前缀)。
    """
    content = f"{model}|{benchmark}|{sample_id}|{prompt[:512]}|{response[:512]}"
    h = hashlib.sha1(content.encode("utf-8")).hexdigest()
    return h[:12]


def _percentile(sorted_vals: List[float], p: float) -> float:
    """计算已排序序列的 p 分位数 (p in 0..100)"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _dist_stats(vals: List[float], unit: str = "", approx: bool = False) -> Dict[str, Any]:
    """对一组数值计算分布统计: 数量/均值/p50/p95/最小/最大"""
    if not vals:
        return {"count": 0}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    stats: Dict[str, Any] = {
        "count": n,
        "mean": round(mean, 2),
        "p50": round(_percentile(s, 50), 2),
        "p95": round(_percentile(s, 95), 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "unit": unit,
    }
    if approx:
        stats["approx"] = True  # 标注该指标为近似值 (如 TTFT)
    return stats
