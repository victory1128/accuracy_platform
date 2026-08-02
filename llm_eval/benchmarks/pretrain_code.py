"""预训练阶段 - 代码生成评测集

HumanEval / MBPP
给函数签名+docstring, 模型补全, 执行测试用例 pass@1。
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..models import BenchmarkMeta, Sample, Stage, TaskType
from .base import Benchmark
from .registry import register
from ._data import data_path

CODE_HEADER = "Complete the following Python function. Only output the code, no explanation. Use ```python ... ``` code fences."


class _CodeBenchmark(Benchmark):
    META: BenchmarkMeta = None  # type: ignore[assignment]
    DATA_FILE: str = ""

    def load_samples(self, limit: Optional[int] = None, seed: int = 42) -> List[Sample]:
        path = data_path(self.DATA_FILE)
        if not os.path.exists(path):
            return []
        rows = self._load_jsonl(path)
        samples = []
        for r in rows:
            sid = r.get("sample_id") or r.get("task_id") or ""
            samples.append(
                Sample(
                    sample_id=sid,
                    prompt="",
                    question=r.get("prompt", ""),
                    gold=r.get("canonical_solution", ""),
                    test_code=r.get("test_code", ""),
                    meta={"entry_point": r.get("entry_point", "")},
                )
            )
        return self._maybe_limit(samples, limit, seed)

    def build_prompt(self, sample: Sample) -> str:
        # 若有 entry_point (MBPP: 函数名), 明确告诉模型用这个名字, 否则 test 调用会 NameError。
        # HumanEval 的 question 已含函数签名, 无需额外加。
        ep = sample.meta.get("entry_point", "")
        if ep and "def " not in sample.question:
            return (f"{CODE_HEADER}\n\nTask: {sample.question}\n\n"
                    f"The function must be named `{ep}`. Define it as `def {ep}(...):`.\n\n```python\n")
        return f"{CODE_HEADER}\n\n```python\n{sample.question}```"

    def parse_params(self) -> dict:
        # 代码生成用低温度。思维链模型思考+代码, 给足到 4096。
        # 重要: 不设 stop。思维链模型会先在 reasoning_content 思考, 再输出 ```python 代码块,
        # 若设 stop=["\ndef "], 模型思考里提到 def 就会提前触发, 导致 content 为空。
        # 代码抽取改由 extract_code() 从 ```python 块里取, 更鲁棒。
        return {"temperature": 0.0, "max_tokens": 4096, "stop": None}


@register
class HumanEval(_CodeBenchmark):
    DATA_FILE = "humaneval.jsonl"
    META = BenchmarkMeta(
        name="humaneval",
        display_name="HumanEval",
        stage=Stage.PRETRAIN,
        task_type=TaskType.CODE,
        description="函数级代码补全, 164题, pass@1",
        tags=["modern", "code", "classic"],
        num_fewshot=0,
        source="Chen et al. 2021 (OpenAI)",
    )


@register
class MBPP(_CodeBenchmark):
    DATA_FILE = "mbpp.jsonl"
    META = BenchmarkMeta(
        name="mbpp",
        display_name="MBPP",
        stage=Stage.PRETRAIN,
        task_type=TaskType.CODE,
        description="基础Python编程, 974题, pass@1",
        tags=["modern", "code"],
        num_fewshot=0,
        source="Austin et al. 2021",
    )


@register
class EvalPlus(_CodeBenchmark):
    """EvalPlus HumanEval+: HumanEval 增强版, 测试用例更多更严 (pass@1)"""
    DATA_FILE = "evalplus.jsonl"
    META = BenchmarkMeta(
        name="evalplus",
        display_name="HumanEval+ (EvalPlus)",
        stage=Stage.PRETRAIN,
        task_type=TaskType.CODE,
        description="HumanEval增强版, 80x测试用例, 更严的pass@1",
        tags=["modern", "code", "classic"],
        num_fewshot=0,
        source="Liu et al. 2024",
    )
