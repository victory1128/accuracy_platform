"""评测集插件基类

每个评测集是一个 Benchmark 子类, 负责:
1. meta(): 返回元信息 (名称/阶段/任务类型/标签)
2. load_samples(): 加载/构造样本列表 (返回 List[Sample])
3. build_prompt(sample): 把样本拼成最终 prompt (含 few-shot)
4. parse_params(): 该评测集的生成参数 (temperature/max_tokens/stop 等)
5. evaluate(sample, response): 给单条响应打分, 填充 SampleResult

基类提供 MCQ / GEN / CODE / JUDGE / RULE 的常用默认实现, 子类按需覆盖。
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from ..models import BenchmarkMeta, Sample, SampleResult, Stage, TaskType
from ..client import LLMClient
from ..scoring import (
    extract_choice,
    extract_number,
    numbers_equal,
    run_code_tests,
    extract_code,
    pass_at_1,
    check_ifeval,
)
from ..scoring.judging import judge_single


class Benchmark:
    """评测集基类"""

    # 子类覆盖
    META: BenchmarkMeta = None  # type: ignore[assignment]

    def meta(self) -> BenchmarkMeta:
        return self.META

    # ---- 数据加载 -------------------------------------------------
    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        """加载样本。子类实现。base 提供从本地 jsonl 读取的便捷方法。"""
        raise NotImplementedError

    def _load_jsonl(self, path: str) -> List[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _maybe_limit(self, samples: List[Sample], limit: Optional[int], seed: int) -> List[Sample]:
        if limit is not None and limit > 0 and limit < len(samples):
            rng = random.Random(seed)
            samples = rng.sample(samples, limit)
        return samples

    # ---- Prompt 构造 ---------------------------------------------
    def build_prompt(self, sample: Sample) -> str:
        return sample.prompt

    def system_prompt(self) -> Optional[str]:
        return None

    # ---- 生成参数 -------------------------------------------------
    def parse_params(self) -> Dict[str, Any]:
        """返回该评测集推荐的生成参数"""
        return {"temperature": 0.0, "max_tokens": 2048, "stop": None}

    # ---- 评测 (默认实现按 task_type 分发) -------------------------
    def evaluate(
        self,
        sample: Sample,
        response: str,
        judge_client: Optional[LLMClient] = None,
    ) -> SampleResult:
        tt = self.META.task_type
        if tt == TaskType.MCQ:
            return self._eval_mcq(sample, response)
        if tt == TaskType.GEN:
            return self._eval_gen(sample, response)
        if tt == TaskType.CODE:
            return self._eval_code(sample, response)
        if tt == TaskType.JUDGE:
            return self._eval_judge(sample, response, judge_client)
        if tt == TaskType.RULE:
            return self._eval_rule(sample, response)
        raise ValueError(f"未知任务类型: {tt}")

    # ---- MCQ: 选项精确匹配 ---------------------------------------
    def _eval_mcq(self, sample: Sample, response: str) -> SampleResult:
        valid = [chr(ord("A") + i) for i in range(len(sample.choices or []))]
        pred = extract_choice(response, valid)
        gold = (sample.gold or "").strip().upper()
        correct = (pred == gold) if pred else False
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=pred,
            correct=correct,
            score=1.0 if correct else 0.0,
        )

    # ---- GEN: 数字/答案抽取匹配 ----------------------------------
    def _eval_gen(self, sample: Sample, response: str) -> SampleResult:
        pred = extract_number(response)
        gold = sample.gold or ""
        correct = _math_answer_equal(pred, gold) if pred else False
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=pred,
            correct=correct,
            score=1.0 if correct else 0.0,
        )

    # ---- CODE: 执行测试用例 --------------------------------------
    def _eval_code(self, sample: Sample, response: str) -> SampleResult:
        completion = extract_code(response)
        passed = False
        err = ""
        if completion and sample.test_code:
            test_code = _ensure_check_called(sample.test_code, sample.meta.get("entry_point", ""))
            # HumanEval 风格: prompt(question)里可能定义了辅助函数 (如 encode_shift),
            # 模型只补全目标函数, 执行时辅助函数丢失 -> test 调用报 NameError。
            # 把 prompt 上下文作为前缀拼上: prompt + completion + test。
            # 仅当 prompt 本身是代码 (含 def) 时才拼, 避免 MBPP 的自然语言题面被当代码。
            # 模型补全常重新 def 目标函数, Python 取后定义 (completion 的), 不冲突。
            prompt_ctx = (sample.question or "").strip()
            if "def " in prompt_ctx:
                full_completion = prompt_ctx + "\n\n" + completion
            else:
                full_completion = completion
            passed, err = run_code_tests(full_completion, test_code, entry_point=sample.meta.get("entry_point", "") or None)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            extracted=completion[:200] if completion else "",
            correct=passed,
            score=1.0 if passed else 0.0,
            error=err or None,
        )

    # ---- JUDGE: LLM 裁判打分 -------------------------------------
    def _eval_judge(
        self, sample: Sample, response: str, judge_client: Optional[LLMClient]
    ) -> SampleResult:
        if judge_client is None:
            return SampleResult(
                sample_id=sample.sample_id,
                response=response,
                error="缺少裁判模型 (judge_client=None)",
            )
        reference = sample.meta.get("reference", "")
        score, reason = judge_single(
            judge_client, sample.question, response, reference
        )
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            score=score,
            error=None if score is not None else f"裁判解析失败: {reason}",
        )

    # ---- RULE: IFEval 规则检查 -----------------------------------
    def _eval_rule(self, sample: Sample, response: str) -> SampleResult:
        constraints = sample.meta.get("constraints", [])
        report = check_ifeval(constraints, response)
        return SampleResult(
            sample_id=sample.sample_id,
            response=response,
            score=report["rate"],
            correct=report["rate"] == 1.0,
            analysis={"ifeval": report},
        )

    # ---- 汇总 ----------------------------------------------------
    def aggregate(self, results: List[SampleResult]) -> Dict[str, Any]:
        """聚合一个评测集所有样本结果 -> 汇总指标"""
        n = len(results) or 1
        tt = self.META.task_type
        agg: Dict[str, Any] = {"num_samples": len(results), "task_type": tt.value}

        if tt == TaskType.JUDGE:
            scores = [r.score for r in results if r.score is not None]
            if scores:
                # 归一到 0~100 (MT-Bench 风格 1-10 → 乘 10)
                agg["mean_score"] = round(sum(scores) / len(scores), 3)
                agg["score_100"] = round(sum(scores) / len(scores) * 10, 2)
            else:
                agg["mean_score"] = None
                agg["score_100"] = None
        elif tt == TaskType.CODE:
            passed = [r for r in results if r.correct]
            agg["pass_at_1"] = round(len(passed) / n, 4)
            agg["accuracy"] = agg["pass_at_1"]
        elif tt == TaskType.RULE:
            rates = [r.score for r in results if r.score is not None]
            agg["instruction_following_rate"] = round(sum(rates) / len(rates), 4) if rates else 0.0
            agg["accuracy"] = agg["instruction_following_rate"]
        else:
            # MCQ / GEN
            correct = [r for r in results if r.correct]
            agg["accuracy"] = round(len(correct) / n, 4)
        return agg


def _ensure_check_called(test_code: str, entry_point: str) -> str:
    """HumanEval/EvalPlus 风格的 test 定义了 def check(candidate): 但不调用它,
    导致断言不执行、误判通过。若检测到此模式, 追加 check(entry_point) 调用。

    MBPP 等顶层裸 assert 不受影响 (无 def check)。
    """
    if not test_code or "def check" not in test_code:
        return test_code
    # 已有 check(...) 调用 (函数定义外) 则不重复
    import re as _re
    # 找 check( 调用, 排除 "def check(" 定义行
    calls = [m for m in _re.finditer(r"(?<!def )\bcheck\s*\(", test_code)
             if "def " not in test_code[max(0, m.start()-8):m.start()]]
    if calls:
        return test_code
    ep = entry_point or "candidate"
    return test_code + f"\n\ncheck({ep})\n"


def _math_answer_equal(pred: str, gold: str) -> bool:
    """数学题答案匹配 (GSM8K / MATH-500 / AIME)。

    在 numbers_equal 基础上兜底两类抽取/口径问题:
    1. LaTeX 百分号: pred='60\\%' / '\\92' (思维链模型常带 LaTeX 转义),
       numbers_equal 因含 '\\' 判非纯数值直接字符串比较 -> 不等。清理 '\\' 后重比。
    2. 百分号口径: pred='60%' gold='60'。GSM8K 里答案就是数字 60, 百分号只是表达方式,
       应判等; 但 numbers_equal 把 '60%' 换算成 0.6 与 60 比 -> 不等。
       故当 gold 是纯数字、pred 是 'N%' 时, 去 '%' 后比较 N==gold。
    """
    if not pred or not gold:
        return False
    # 主路径
    if numbers_equal(pred, gold):
        return True
    import re as _re
    # 兜底1: 清理 LaTeX 转义反斜杠 (\\% -> %, \\92 -> 92) 后重比
    cleaned = pred.replace("\\", "").strip()
    if cleaned != pred and numbers_equal(cleaned, gold):
        return True
    # 兜底2: gold 是纯数字、pred 含 % -> 去掉 % 比较数值 (60% == 60)
    g = gold.strip()
    p_no_pct = cleaned.rstrip("%").strip()
    if "%" in cleaned and p_no_pct and p_no_pct != cleaned:
        if numbers_equal(p_no_pct, g):
            return True
    return False
